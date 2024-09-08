#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install -q transformers
get_ipython().system('pip install gdown')


# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, DataLoader


# In[2]:


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[5]:


get_ipython().system('gdown --id 1k5LMwmYF7PF-BzYQNE2ULBae79nbM268')
get_ipython().system('gdown --id 1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE')


# In[4]:


# dataframe = pd.read_json("/kaggle/working/subtaskB_train.jsonl", lines=True)
# dataframe_test = pd.read_json("/kaggle/working/subtaskB_dev.jsonl", lines=True)
dataframe = pd.read_json("subtaskB_train.jsonl", lines=True)
dataframe_test = pd.read_json("subtaskB_dev.jsonl", lines=True)


# In[5]:


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


# In[6]:


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


# In[7]:


tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

df1 = dataframe.copy()
df5 = dataframe.copy()
df10 = dataframe.copy()
df50 = dataframe.copy()
dataframe_1 = DataframeSetting(df1, 1)
dataframe_5 = DataframeSetting(df5, 5)
dataframe_10 = DataframeSetting(df10, 10)
dataframe_50 = DataframeSetting(df50, 50)

training_set1 = CustomDataset(dataframe_1, tokenizer)
training_set5 = CustomDataset(dataframe_5, tokenizer)
training_set10 = CustomDataset(dataframe_10, tokenizer)
training_set50 = CustomDataset(dataframe_50, tokenizer)

training_data1 = DataLoader(training_set1, batch_size=128, shuffle=True)
training_data5 = DataLoader(training_set5, batch_size=128, shuffle=True)
training_data10 = DataLoader(training_set10, batch_size=128, shuffle=True)
training_data50 = DataLoader(training_set50, batch_size=128, shuffle=True)

test_set = CustomDataset(dataframe_test, tokenizer)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


# In[8]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 768),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(768, 768),
        )
    def forward(self, noise):
        output = self.layers(noise)
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


# In[9]:


def training_data(transformer, generator, discriminator, num_epochs, training_loader, testing_loader, learning_rate_discriminator, learning_rate_generator):

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


# In[10]:


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


# In[20]:


transformer_1 = transformers.AutoModel.from_pretrained('bert-base-uncased')
transformer_1 = torch.nn.DataParallel(transformer_1)
transformer_1 = transformer_1.to(device)

generator_1 = Generator()
generator_1 = generator_1.to(device)

discriminator_1 = Discriminator()
discriminator_1 = discriminator_1.to(device)

num_epochs = 5
learning_rate_discriminator = 5e-5
learning_rate_generator = 5e-5


# In[14]:


training_data(transformer_1, generator_1, discriminator_1, num_epochs, training_data1, test_data, learning_rate_discriminator, learning_rate_generator)

torch.save(transformer_1.state_dict(), "transformer_1.pth")
torch.save(generator_1.state_dict(), "generator_1.pth")
torch.save(discriminator_1.state_dict(), "discriminator_1.pth")


# In[22]:


with torch.no_grad():
    test_1 = testing_data(transformer_1, discriminator_1,test_data)
    
print(f"Accuracy on validation set for 1 percent of data: {test_1}")


# In[16]:


transformer_5 = transformers.AutoModel.from_pretrained('bert-base-uncased')
transformer_5 = torch.nn.DataParallel(transformer_5)
transformer_5 = transformer_5.to(device)

generator_5 = Generator()
generator_5 = generator_5.to(device)

discriminator_5 = Discriminator()
discriminator_5 = discriminator_5.to(device)

num_epochs = 5
learning_rate_discriminator = 5e-5
learning_rate_generator = 5e-5


# In[17]:


training_data(transformer_5, generator_5, discriminator_5, num_epochs, training_data5, test_data, learning_rate_discriminator, learning_rate_generator)

torch.save(transformer_5.state_dict(), "transformer_5.pth")
torch.save(generator_5.state_dict(), "generator_5.pth")
torch.save(discriminator_5.state_dict(), "discriminator_5.pth")


# In[18]:


with torch.no_grad():
    test_5 = testing_data(transformer_5, discriminator_5,test_data)
    
print(f"Accuracy on validation set for 5 percent of data: {test_5}")


# In[12]:


transformer_10 = transformers.AutoModel.from_pretrained('bert-base-uncased')
transformer_10 = torch.nn.DataParallel(transformer_10)
transformer_10 = transformer_10.to(device)

generator_10 = Generator()
generator_10 = generator_10.to(device)

discriminator_10 = Discriminator()
discriminator_10 = discriminator_10.to(device)

num_epochs = 5
learning_rate_discriminator = 5e-5
learning_rate_generator = 5e-5


# In[20]:


training_data(transformer_10, generator_10, discriminator_10, num_epochs, training_data10, test_data, learning_rate_discriminator, learning_rate_generator)

torch.save(transformer_10.state_dict(), "transformer_10.pth")
torch.save(generator_10.state_dict(), "generator_10.pth")
torch.save(discriminator_10.state_dict(), "discriminator_10.pth")


# In[14]:


with torch.no_grad():
    test_10 = testing_data(transformer_10, discriminator_10,test_data)
    
print(f"Accuracy on validation set for 10 percent of data: {test_10}")


# In[15]:


transformer_50 = transformers.AutoModel.from_pretrained('bert-base-uncased')
transformer_50 = torch.nn.DataParallel(transformer_50)
transformer_50 = transformer_50.to(device)

generator_50 = Generator()
generator_50 = generator_50.to(device)

discriminator_50 = Discriminator()
discriminator_50 = discriminator_50.to(device)

num_epochs = 5
learning_rate_discriminator = 5e-5
learning_rate_generator = 5e-5


# In[23]:


training_data(transformer_50, generator_50, discriminator_50, num_epochs, training_data50, test_data, learning_rate_discriminator, learning_rate_generator)

torch.save(transformer_50.state_dict(), "transformer_50.pth")
torch.save(generator_50.state_dict(), "generator_50.pth")
torch.save(discriminator_50.state_dict(), "discriminator_50.pth")


# In[17]:


with torch.no_grad():
    test_50 = testing_data(transformer_50, discriminator_50,test_data)
    
print(f"Accuracy on validation set for 50 percent of data: {test_50}")


# In[24]:


Accuracies_Bert = [33.56666666666667, 49.46666666666667, 47.63333333333333, 50.5]
Accuracies_GAN_Bert = [47.3, 45, 46.5, 53.6]
data_percentage = [0.01, 0.05, 0.1, 0.5]


# In[26]:


plt.plot(data_percentage, Accuracies_Bert, color='r', label='Bert') 
plt.plot(data_percentage, Accuracies_GAN_Bert, color='g', label='GAN-Bert(G1)') 
plt.xlabel('Data Percentage')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Data Percentage')
plt.legend() 
plt.show() 

