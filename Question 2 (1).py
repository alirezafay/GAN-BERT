#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U adapter-transformers')


# In[2]:


import numpy as np
import pandas as pd
import torch.nn as nn
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import json
from transformers import AutoModelWithHeads


# In[3]:


get_ipython().system('pip install gdown')


# In[4]:


get_ipython().system('gdown --id 1k5LMwmYF7PF-BzYQNE2ULBae79nbM268')
#https://drive.google.com/file/d/1k5LMwmYF7PF-BzYQNE2ULBae79nbM268/view?usp=drive_link


# In[ ]:


get_ipython().system('gdown --id 16btYEgpxxvn1of04nsg02HpFUT9INYVb')


# In[5]:


get_ipython().system('gdown --id 1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE')
#https://drive.google.com/file/d/1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE/view?usp=drive_link


# In[17]:


dataframe = pd.read_json("/kaggle/working/subtaskB_train.jsonl", lines=True)
dataframe_test = pd.read_json("/kaggle/working/subtaskB_dev.jsonl", lines=True)


# In[18]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[19]:


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
        token_dictionary = self.tokenizer.encode_plus(text,None,add_special_tokens=True,max_length=256,pad_to_max_length=True,return_token_type_ids=True)
        ids = token_dictionary['input_ids']
        attention_mask = token_dictionary['attention_mask']
        token_type_ids = token_dictionary["token_type_ids"]

        return {'ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(attention_mask, dtype=torch.long), 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long), 'labels': torch.tensor(self.labels[index], dtype=torch.float) }


# In[20]:


dataframe_1 = dataframe.sample(frac=0.01, random_state=42)
dataframe_1 = dataframe_1.reset_index(drop=True)
dataframe_5 = dataframe.sample(frac=0.05, random_state=42)
dataframe_5 = dataframe_5.reset_index(drop=True)
dataframe_10 = dataframe.sample(frac=0.1, random_state=42)
dataframe_10 = dataframe_10.reset_index(drop=True)
dataframe_50 = dataframe.sample(frac=0.5, random_state=42)
dataframe_50 = dataframe_50.reset_index(drop=True)

training_set1 = CustomDataset(dataframe_1, tokenizer)
training_set5 = CustomDataset(dataframe_5, tokenizer)
training_set10 = CustomDataset(dataframe_10, tokenizer)
training_set50 = CustomDataset(dataframe_50, tokenizer)

training_data1 = DataLoader(training_set1, batch_size=128, shuffle=True)
training_data5 = DataLoader(training_set5, batch_size=128, shuffle=True)
training_data10 = DataLoader(training_set10, batch_size=128, shuffle=True)
training_data50 = DataLoader(training_set50, batch_size=128, shuffle=True)


# In[21]:


test_set = CustomDataset(dataframe_test, tokenizer)
test_data = DataLoader(test_set, shuffle=False)


# In[22]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[23]:


model = AutoModelWithHeads.from_pretrained("bert-base-uncased")
model.add_classification_head("custom", num_labels=6)
model.add_adapter("custom")


# In[24]:


model


# In[25]:


model.bert.encoder.layer[1].output.adapters


# In[26]:


# برای تولید فانکشن ترینینگ در اینجا، ابتدا تمامی پارامترهای شبکه را فریز میکنیم. سپس بخش های مختلف شبکه را که مربوط به پارامترهای اداپتر است را مشتق پذیر در نظر میگیریم تا پارامترهای مربوطه فقط آپدیت شوند. 
def training_data(model, num_epochs, training_loader, learning_rate):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.module.heads.parameters():
        param.requires_grad = True
    for i in range(12):
        for param in model.module.bert.encoder.layer[i].attention.output.adapter_fusion_layer.parameters():
            param.requires_grad = True
        for param in model.module.bert.encoder.layer[i].attention.output.adapters.parameters():
            param.requires_grad = True
        for param in model.module.bert.encoder.layer[i].output.adapters.parameters():
            param.requires_grad = True
        for param in model.module.bert.encoder.layer[i].output.adapter_fusion_layer.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    Loss_train=[]
    accuracy_list=[]
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range (num_epochs):
        epoch_train_loss = 0.0
        correct_pred = 0
        total_samples = 0
        for i,dataset in enumerate(training_loader):
            ids = dataset['ids'].to(device, dtype = torch.long)
            mask = dataset['attention_mask'].to(device, dtype = torch.long)
            labels = dataset['labels'].to(device, dtype = torch.long)
            token_type_ids = dataset['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).logits
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            total_samples += labels.size(0)
            preds = torch.argmax (outputs, dim=1)
            correct_pred += (preds == labels).sum().item()
        epoch_train_accuracy =  100 * correct_pred / total_samples
        accuracy_list.append(epoch_train_accuracy)
        Loss_train.append(epoch_train_loss)
        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss/len(training_loader):.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")


# In[27]:


def testing_data(model,testing_loader):
    total_samples = 0
    correct_pred = 0
    for i,dataset in enumerate(testing_loader):
        ids = dataset['ids'].to(device, dtype = torch.long)
        mask = dataset['attention_mask'].to(device, dtype = torch.long)
        labels = dataset['labels'].to(device, dtype = torch.long)
        token_type_ids = dataset['token_type_ids'].to(device, dtype = torch.long)
        outputs = model(ids, mask, token_type_ids).logits
        total_samples += labels.size(0)
        preds = torch.argmax (outputs, dim=1)
        correct_pred += (preds == labels).sum().item()
    test_accuracy = 100 * correct_pred / total_samples
    return test_accuracy


# In[28]:


model_1 = AutoModelWithHeads.from_pretrained("bert-base-uncased")
model_1.add_classification_head("custom", num_labels=6)
model_1.add_adapter("custom")
model_1.set_active_adapters("custom")
model_1 = nn.DataParallel(model_1, device_ids=[0, 1])
model_1 = model_1.to(device)


# In[29]:


training_data(model_1, 15, training_data1, learning_rate= 2e-05)


# In[30]:


with torch.no_grad():
    test_1 = testing_data(model_1,test_data)    


# In[31]:


print(f"Accuracy on validation set for 1 percent of data: {test_1}")


# In[32]:


num_params = sum(p.numel() for p in model_1.parameters() if p.requires_grad)
print(f"Number of trainable parameters with adapter and bert: {num_params}")


# In[33]:


main_model = AutoModelWithHeads.from_pretrained("bert-base-uncased")
main_model.add_classification_head("custom", num_labels=6)
num_params_main_model = sum(p.numel() for p in main_model.parameters())
print(f"Number of trainable parameters without bert: {num_params_main_model}")


# In[34]:


model_5 = AutoModelWithHeads.from_pretrained("bert-base-uncased")
model_5.add_classification_head("custom", num_labels=6)
model_5.add_adapter("custom")
model_5.set_active_adapters("custom")
model_5 = nn.DataParallel(model_5, device_ids=[0, 1])
model_5 = model_5.to(device)


# In[35]:


training_data(model_5, 15, training_data5, learning_rate= 2e-05)


# In[36]:


with torch.no_grad():
    test_5 = testing_data(model_5,test_data)    


# In[37]:


print(f"Accuracy on validation set for 5 percent of data: {test_5}")


# In[38]:


model_10 = AutoModelWithHeads.from_pretrained("bert-base-uncased")
model_10.add_classification_head("custom", num_labels=6)
model_10.add_adapter("custom")
model_10.set_active_adapters("custom")
model_10 = nn.DataParallel(model_10, device_ids=[0, 1])
model_10 = model_10.to(device)


# In[39]:


training_data(model_10, 15, training_data5, learning_rate= 2e-05)


# In[40]:


with torch.no_grad():
    test_10 = testing_data(model_10,test_data)    


# In[41]:


print(f"Accuracy on validation set for 10 percent of data: {test_10}")


# In[42]:


model_50 = AutoModelWithHeads.from_pretrained("bert-base-uncased")
model_50.add_classification_head("custom", num_labels=6)
model_50.add_adapter("custom")
model_50.set_active_adapters("custom")
model_50 = nn.DataParallel(model_50, device_ids=[0, 1])
model_50 = model_50.to(device)


# In[44]:


training_data(model_50, 5, training_data50, learning_rate= 2e-05)


# In[45]:


with torch.no_grad():
    test_50 = testing_data(model_50,test_data)    


# In[46]:


print(f"Accuracy on validation set for 50 percent of data: {test_50}")


# In[47]:


Accuracies = [33.56666666666667, 49.46666666666667, 47.63333333333333, 50.5]
data_percentage = [0.01, 0.05, 0.1, 0.5]


# In[49]:


import matplotlib.pyplot as plt
plt.plot(data_percentage, Accuracies, marker='o')  
plt.xlabel('Data Percentage')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Data Percentage')
plt.show() 


# In[ ]:




