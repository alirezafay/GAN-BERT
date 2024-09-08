#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U adapter-transformers')
get_ipython().system('pip install gdown')


# In[2]:


import json
import math
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoModelWithHeads


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[4]:


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)


# In[5]:


get_ipython().system('gdown --id 1TcmFzF6d17dcg0DyR59rmRwLWuxSEXIn')
get_ipython().system('gdown --id 1-xMDOvuxuH2zW5qcgBGxawpkN1xZ8Ju-')
dataframe = pd.read_json("/kaggle/working/subtaskB_train.jsonl", lines=True)
dataframe_test = pd.read_json("/kaggle/working/subtaskB_dev.jsonl", lines=True)


# In[6]:


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.labels = dataframe.label
        self.text = dataframe.text


    def __len__(self):

        return len(self.text)

    def __getitem__(self, index):

        text = str(self.text[index])
        text = " ".join(text.split())
        token_dictionary = self.tokenizer.encode_plus(text,None,add_special_tokens=True,max_length=128,pad_to_max_length=True,return_token_type_ids=True,truncation=True)
        ids = token_dictionary['input_ids']
        attention_mask = token_dictionary['attention_mask']
        token_type_ids = token_dictionary["token_type_ids"]

        return {'ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(attention_mask, dtype=torch.long), 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long), 'labels': torch.tensor(self.labels[index], dtype=torch.float) }


# In[7]:


def DataframeSetting(dataframe, labeld_rate):
    df = dataframe.copy()
    unlabeled_df = df.sample(frac=(1 - (labeld_rate / 100)), random_state=42)
    unlabeled_df['label'] = -1
    df.loc[unlabeled_df.index, 'label'] = -1
    labeld_df = df.loc[df['label'] != -1]
    repeat_rate = int(1/(labeld_rate / 100))
    repeat_rate = int(math.log(repeat_rate,2))
    labeld_df = labeld_df.loc[np.repeat(labeld_df.index, repeat_rate)].reset_index(drop=True)
    df_new = pd.DataFrame()
    df_new = pd.concat([labeld_df, unlabeled_df], ignore_index=True, sort=False)
    return df_new


# In[8]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df50 = dataframe.copy()
dataframe_50 = DataframeSetting(df50, 50)
training_set50 = CustomDataset(dataframe_50, tokenizer)
training_data50 = DataLoader(training_set50, batch_size=128, shuffle=True)

test_set = CustomDataset(dataframe_test, tokenizer)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


# In[9]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048,3072)
        self.fc5 = nn.Linear(3072,4096)
        self.fc6 = nn.Linear(4096,7168)
        self.fc7 = nn.Linear(7168, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)


    def forward(self, noise):
        output1 = self.fc1(noise)
        output1 = self.relu(output1)
        output1 = self.dropout(output1)
        output2 = self.fc2(output1)
        output2 = self.relu(output2)
        output2 = self.dropout(output2)
        input3 = torch.cat((output1,output2), dim=1)
        output3 = self.fc3(input3)
        output3 = self.relu(output3)
        output3 = self.dropout(output3)
        output4 = self.fc4(output3)#3072
        output4 = self.relu(output4)
        output4 = self.dropout(output4)
        input5 = torch.cat((output1,output2,output3), dim=1) + output4
        output5 = self.fc5(input5)#4096
        output5 = self.relu(output5)
        output5 = self.dropout(output5)
        output6 = self.fc6(output5)#7168
        output6 = self.relu(output6)
        output6 = self.dropout(output6)
        input7 = torch.cat((output4,output5), dim=1) + output6
        output7 = self.fc7(input7)
        output7 = self.relu(output7)
        output = self.dropout(output7)
        return output

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 768),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        ) 
        self.logit = nn.Linear(768,7)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        features = self.layers(x)
        logits = self.logit(features)
        probs = self.softmax(logits)
        return features, logits, probs


# In[10]:


def DataframeSetting(dataframe, labeld_rate):
    df = dataframe.copy()
    unlabeled_df = df.sample(frac=(1 - (labeld_rate / 100)), random_state=42)
    unlabeled_df['label'] = -1
    df.loc[unlabeled_df.index, 'label'] = -1
    labeld_df = df.loc[df['label'] != -1]
    repeat_rate = int(1/(labeld_rate / 100))
    repeat_rate = int(math.log(repeat_rate,2))
    labeld_df = labeld_df.loc[np.repeat(labeld_df.index, repeat_rate)].reset_index(drop=True)
    df_new = pd.DataFrame()
    df_new = pd.concat([labeld_df, unlabeled_df], ignore_index=True, sort=False)
    return df_new


# In[12]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df50 = dataframe.copy()
dataframe_50 = DataframeSetting(df50, 50)
training_set50 = CustomDataset(dataframe_50, tokenizer)
training_data50 = DataLoader(training_set50, batch_size=128, shuffle=True)

test_set = CustomDataset(dataframe_test, tokenizer)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


# In[13]:


def training_data(transformer, generator, discriminator, num_epochs, training_loader, testing_loader, learning_rate_discriminator, learning_rate_generator):

    for param in transformer.parameters():
        param.requires_grad = False
    for param in transformer.module.heads.parameters():
        param.requires_grad = True
    for i in range(12):
        for param in transformer.module.bert.encoder.layer[i].attention.output.adapter_fusion_layer.parameters():
            param.requires_grad = True
        for param in transformer.module.bert.encoder.layer[i].attention.output.adapters.parameters():
            param.requires_grad = True
        for param in transformer.module.bert.encoder.layer[i].output.adapters.parameters():
            param.requires_grad = True
        for param in transformer.module.bert.encoder.layer[i].output.adapter_fusion_layer.parameters():
            param.requires_grad = True
    d_optimizer = torch.optim.AdamW(params= list(discriminator.parameters())+list(transformer.parameters()), lr=learning_rate_discriminator)
    g_optimizer = torch.optim.AdamW(params= generator.parameters(), lr=learning_rate_generator)

    for epoch in range (num_epochs):

        epsilon = 1e-8
        d_loss = 0
        g_loss = 0
        epoch_train_d_loss = 0.0
        epoch_train_g_loss = 0.0
        transformer.train()
        generator.train()
        discriminator.train()

        for i,dataset in enumerate(training_loader):

            ids = dataset['ids'].to(device, dtype = torch.long)
            mask = dataset['attention_mask'].to(device, dtype = torch.long)
            labels = dataset['labels'].to(device, dtype = torch.long)
            token_type_ids = dataset['token_type_ids'].to(device, dtype = torch.long)

            real_labeled_index      = (labels != -1).nonzero(as_tuple=True)
            real_unlabeled_index    = (labels == -1).nonzero(as_tuple=True)

            real_batch_size = ids.shape[0]
            noise = torch.zeros(real_batch_size, 100, device=device).uniform_(0, 1)

            transformer_outputs = transformer(ids, attention_mask = mask)
            generator_outputs = generator(noise)
            disciminator_input = torch.cat([transformer_outputs[-1], generator_outputs], dim=0)
            features, logits, probs = discriminator(disciminator_input)

            features_list = torch.split(features, real_batch_size)
            real_features = features_list[0]
            fake_features = features_list[1]
            
            logits_list = torch.split(logits, real_batch_size)
            real_logits = logits_list[0]
            fake_logits = logits_list[1]
            real_labeled_logits = real_logits[real_labeled_index]
            real_labeled_logits = real_labeled_logits[:,0:-1]
            
            probs_list = torch.split(probs, real_batch_size)
            real_probs = probs_list[0]
            fake_probs = probs_list[1]

            real_labeled_probs = real_probs[real_labeled_index]
            real_labeled_probs = real_labeled_probs[:,0:-1]

            g_loss_unsup = -1 * torch.mean(torch.log(1 - fake_probs[:,-1] + epsilon))
            g_loss_feat = torch.mean(torch.pow(torch.mean(real_features, dim=0) - torch.mean(fake_features, dim=0), 2))
            g_loss = g_loss_unsup + g_loss_feat
            
            log_probs = F.log_softmax(real_labeled_logits, dim=-1)
            label2one_hot = torch.nn.functional.one_hot(labels[real_labeled_index], 6)
            labeled_loss = -torch.sum(label2one_hot * log_probs, dim=-1)

            if real_labeled_index[0].shape[0] == 0 :
                d_loss_sup = 0
            else:
                d_loss_sup = torch.div(torch.sum(labeled_loss.to(device)), real_labeled_index[0].shape[0])

            d_loss_unsup = -1 * torch.mean(torch.log(1 - real_probs[:, -1] + epsilon))
            d_loss_unsup += -1 * torch.mean(torch.log(fake_probs[:, -1] + epsilon))
            d_loss = d_loss_sup + d_loss_unsup

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            g_loss.backward(retain_graph=True)
            d_loss.backward()
            
            g_optimizer.step()
            d_optimizer.step()
            
            epoch_train_g_loss += g_loss.item()
            epoch_train_d_loss += d_loss.item()
            
        epoch_train_g_loss = epoch_train_g_loss / len(training_loader)
        epoch_train_d_loss = epoch_train_d_loss / len(training_loader)
        
        with torch.no_grad():
            test_acc = testing_data(transformer, discriminator,testing_loader)
    
        print(f"Epoch {epoch+1} - Train Loss Generator: {epoch_train_g_loss:.4f}, Train Loss Discriminator: {epoch_train_d_loss:.4f}, Validation Accuracy: {test_acc:.4f}")


# In[14]:


def testing_data(transformer,discriminator,testing_loader):
    
    transformer.eval()
    discriminator.eval()

    total_samples = 0
    correct_pred = 0
    
    for i,dataset in enumerate(testing_loader):
        
        ids = dataset['ids'].to(device, dtype = torch.long)
        mask = dataset['attention_mask'].to(device, dtype = torch.long)
        labels = dataset['labels'].to(device, dtype = torch.long)
        token_type_ids = dataset['token_type_ids'].to(device, dtype = torch.long)
        
        model_outputs = transformer(ids, attention_mask = mask)
        _, logits, probs = discriminator(model_outputs[-1])
        
        real_probs = probs[:,0:-1]
        
        total_samples += labels.size(0)
        preds = torch.argmax (real_probs, dim=1)
        correct_pred += (preds == labels).sum().item()
        
    test_accuracy = 100 * correct_pred / total_samples
    return test_accuracy


# In[15]:


test_set = CustomDataset(dataframe_test, tokenizer)
test_data = DataLoader(test_set, batch_size=8, shuffle=False)


# In[17]:


transformer_50 = AutoModelWithHeads.from_pretrained("bert-base-uncased")
transformer_50.add_adapter("custom")
transformer_50.set_active_adapters("custom")
transformer_50 = torch.nn.DataParallel(transformer_50)
transformer_50 = transformer_50.to(device)

generator_50 = Generator()
generator_50 = generator_50.to(device)

discriminator_50 = Discriminator()
discriminator_50 = discriminator_50.to(device)

num_epochs = 5
learning_rate_discriminator = 5e-5
learning_rate_generator = 5e-5


# In[18]:


training_data(transformer_50, generator_50, discriminator_50, num_epochs, training_data50, test_data, learning_rate_discriminator, learning_rate_generator)

torch.save(transformer_50.state_dict(), "transformer_50.pth")
torch.save(generator_50.state_dict(), "generator_50.pth")
torch.save(discriminator_50.state_dict(), "discriminator_50.pth")


# In[24]:


def training_data(trans_freez, transformer, generator, discriminator, num_epochs, training_loader, testing_loader, learning_rate_discriminator, learning_rate_generator):

    for param in transformer.parameters():
        param.requires_grad = False
    for param in transformer.module.heads.parameters():
        param.requires_grad = True
    for i in range(12):
        for param in transformer.module.bert.encoder.layer[i].attention.output.adapter_fusion_layer.parameters():
            param.requires_grad = True
        for param in transformer.module.bert.encoder.layer[i].attention.output.adapters.parameters():
            param.requires_grad = True
        for param in transformer.module.bert.encoder.layer[i].output.adapters.parameters():
            param.requires_grad = True
        for param in transformer.module.bert.encoder.layer[i].output.adapter_fusion_layer.parameters():
            param.requires_grad = True
    d_optimizer = torch.optim.AdamW(params= list(discriminator.parameters())+list(transformer.parameters()), lr=learning_rate_discriminator)
    g_optimizer = torch.optim.AdamW(params= generator.parameters(), lr=learning_rate_generator)

    for epoch in range (num_epochs):

        epsilon = 1e-8
        d_loss = 0
        g_loss = 0
        epoch_train_d_loss = 0.0
        epoch_train_g_loss = 0.0
        transformer.train()
        generator.train()
        discriminator.train()

        for i,dataset in enumerate(training_loader):

            ids = dataset['ids'].to(device, dtype = torch.long)
            mask = dataset['attention_mask'].to(device, dtype = torch.long)
            labels = dataset['labels'].to(device, dtype = torch.long)
            token_type_ids = dataset['token_type_ids'].to(device, dtype = torch.long)

            real_labeled_index      = (labels != -1).nonzero(as_tuple=True)
            real_unlabeled_index    = (labels == -1).nonzero(as_tuple=True)

            real_batch_size = ids.shape[0]

            transformer_outputs = transformer(ids, attention_mask = mask)
            generator_input1 = trans_freez(ids, attention_mask = mask)
            generator_outputs1 = Generator(generator_input1.tensor[2])
            
            generator_input2 = trans_freez(ids, attention_mask = mask)
            generator_outputs2 = Generator(generator_input2.tensor[5])
            
            generator_input3 = trans_freez(ids, attention_mask = mask)
            generator_outputs3 = Generator(generator_input3.tensor[7])
            
            generator_input4 = trans_freez(ids, attention_mask = mask)
            generator_outputs4 = Generator(generator_input4.tensor[10])
            
            generator_input5 = trans_freez(ids, attention_mask = mask)
            generator_outputs5 = Generator(generator_input5.tensor[3])
            
            disciminator_input = torch.cat([transformer_outputs[-1], generator_outputs1[-1], generator_outputs2[-1], generator_outputs3[-1], generator_outputs4[-1], generator_outputs5[-1]], dim=0)
            features, logits, probs = discriminator(disciminator_input)

            features_list = torch.split(features, real_batch_size)
            real_features = features_list[0]
            fake_features = features_list[1]
            
            logits_list = torch.split(logits, real_batch_size)
            real_logits = logits_list[0]
            fake_logits = logits_list[1]
            real_labeled_logits = real_logits[real_labeled_index]
            real_labeled_logits = real_labeled_logits[:,0:-1]
            
            probs_list = torch.split(probs, real_batch_size)
            real_probs = probs_list[0]
            fake_probs = probs_list[1]

            real_labeled_probs = real_probs[real_labeled_index]
            real_labeled_probs = real_labeled_probs[:,0:-1]

            g_loss_unsup = -1 * torch.mean(torch.log(1 - fake_probs[:,-1] + epsilon))
            g_loss_feat = torch.mean(torch.pow(torch.mean(real_features, dim=0) - torch.mean(fake_features, dim=0), 2))
            g_loss = g_loss_unsup + g_loss_feat
            
            log_probs = F.log_softmax(real_labeled_logits, dim=-1)
            label2one_hot = torch.nn.functional.one_hot(labels[real_labeled_index], 6)
            labeled_loss = -torch.sum(label2one_hot * log_probs, dim=-1)

            if real_labeled_index[0].shape[0] == 0 :
                d_loss_sup = 0
            else:
                d_loss_sup = torch.div(torch.sum(labeled_loss.to(device)), real_labeled_index[0].shape[0])

            d_loss_unsup = -1 * torch.mean(torch.log(1 - real_probs[:, -1] + epsilon))
            d_loss_unsup += -1 * torch.mean(torch.log(fake_probs[:, -1] + epsilon))
            d_loss = d_loss_sup + d_loss_unsup

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            g_loss.backward(retain_graph=True)
            d_loss.backward()
            
            g_optimizer.step()
            d_optimizer.step()
            
            epoch_train_g_loss += g_loss.item()
            epoch_train_d_loss += d_loss.item()
            
        epoch_train_g_loss = epoch_train_g_loss / len(training_loader)
        epoch_train_d_loss = epoch_train_d_loss / len(training_loader)
        
        with torch.no_grad():
            test_acc = testing_data(transformer, discriminator,testing_loader)
    
        print(f"Epoch {epoch+1} - Train Loss Generator: {epoch_train_g_loss:.4f}, Train Loss Discriminator: {epoch_train_d_loss:.4f}, Validation Accuracy: {test_acc:.4f}")


# In[25]:


transformer_50_2 = AutoModelWithHeads.from_pretrained("bert-base-uncased")
transformer_50_2.add_adapter("custom")
transformer_50_2.set_active_adapters("custom")
transformer_50_2 = torch.nn.DataParallel(transformer_50_2)
transformer_50_2 = transformer_50_2.to(device)

generator_50_2 = Generator()
generator_50_2 = generator_50_2.to(device)

discriminator_50_2 = Discriminator()
discriminator_50_2 = discriminator_50.to(device)

num_epochs = 5
learning_rate_discriminator = 5e-5
learning_rate_generator = 5e-5


# In[26]:


training_data(transformer_50, transformer_50_2, generator_50_2, discriminator_50_2, num_epochs, training_data50, test_data, learning_rate_discriminator, learning_rate_generator)

torch.save(transformer_50.state_dict(), "transformer_50.pth")
torch.save(generator_50.state_dict(), "generator_50.pth")
torch.save(discriminator_50.state_dict(), "discriminator_50.pth")


# In[ ]:


with torch.no_grad():
    test_50 = testing_data(transformer_50, discriminator_50,training_data50)
    
print(f"Accuracy on validation set for 50 percent of data: {test_50}")

