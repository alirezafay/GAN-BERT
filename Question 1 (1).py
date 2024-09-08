#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q transformers')


# In[2]:


import numpy as np
import pandas as pd
import torch.nn as nn
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import json
import matplotlib.pyplot as plt


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


get_ipython().system('pip install gdown')


# In[5]:


get_ipython().system('gdown --id 1k5LMwmYF7PF-BzYQNE2ULBae79nbM268')
#https://drive.google.com/file/d/1k5LMwmYF7PF-BzYQNE2ULBae79nbM268/view?usp=drive_link


# In[6]:


get_ipython().system('gdown --id 1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE')
#https://drive.google.com/file/d/1oh9c-d0fo3NtETNySmCNLUc6H1j4dSWE/view?usp=drive_link


# In[7]:


dataframe = pd.read_json("/kaggle/working/subtaskB_train.jsonl", lines=True)


# In[8]:


dataframe_test = pd.read_json("/kaggle/working/subtaskB_dev.jsonl", lines=True)


# In[9]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[10]:


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
        token_dictionary = self.tokenizer.encode_plus(text,None,add_special_tokens=True,max_length=128,pad_to_max_length=True,return_token_type_ids=True)
        ids = token_dictionary['input_ids']
        attention_mask = token_dictionary['attention_mask']
        token_type_ids = token_dictionary["token_type_ids"]

        return {'ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(attention_mask, dtype=torch.long), 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long), 'labels': torch.tensor(self.labels[index], dtype=torch.float) }


# In[ ]:


seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)


# In[11]:


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


# In[12]:


test_set = CustomDataset(dataframe_test, tokenizer)
test_data = DataLoader(test_set, shuffle=False)


# In[13]:


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 6)
        self.dropout = nn.Dropout(p=0.2,inplace=False)

    def forward(self, ids, attention_mask, token_type_ids):
        out1,pooled_output = self.bert_model(ids, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict=False)
        output = self.fc(self.dropout(pooled_output))
        return output


# In[14]:


def training_data(model, num_epochs, training_loader, learning_rate):
    Loss_train=[]
    accuracy_list=[]
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params= model.parameters(), lr= learning_rate)
    for epoch in range (num_epochs):
        epoch_train_loss = 0.0
        correct_pred = 0
        total_samples = 0
        for i,dataset in enumerate(training_loader):
            ids = dataset['ids'].to(device, dtype = torch.long)
            mask = dataset['attention_mask'].to(device, dtype = torch.long)
            labels = dataset['labels'].to(device, dtype = torch.long)
            token_type_ids = dataset['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
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
        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")


# In[15]:


def testing_data(model,testing_loader):
    total_samples = 0
    correct_pred = 0
    for i,dataset in enumerate(testing_loader):
        ids = dataset['ids'].to(device, dtype = torch.long)
        mask = dataset['attention_mask'].to(device, dtype = torch.long)
        labels = dataset['labels'].to(device, dtype = torch.long)
        token_type_ids = dataset['token_type_ids'].to(device, dtype = torch.long)
        outputs = model(ids, mask, token_type_ids)
        total_samples += labels.size(0)
        preds = torch.argmax (outputs, dim=1)
        correct_pred += (preds == labels).sum().item()
    test_accuracy = 100 * correct_pred / total_samples
    return test_accuracy


# In[16]:


model_1 = BERT()
model_1 = nn.DataParallel(model_1, device_ids=[0, 1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_1 = model_1.to(device)


# In[17]:


num_params_main_model = sum(p.numel() for p in model_1.parameters())
print(f"Number of trainable parameters: {num_params_main_model}")


# In[18]:


training_data(model_1, 5, training_data1, learning_rate= 2e-05)


# In[19]:


with torch.no_grad():
    test_1 = testing_data(model_1,test_data)    


# In[20]:


print(f"Accuracy on validation set for 1 percent of data: {test_1}")


# In[21]:


model_5 = BERT()
model_5 = nn.DataParallel(model_5, device_ids=[0, 1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_5 = model_5.to(device)


# In[22]:


training_data(model_5, 5, training_data5, learning_rate= 2e-05)


# In[23]:


with torch.no_grad():
    test_5 = testing_data(model_5,test_data)    


# In[33]:


print(f"Accuracy on validation set for 5 percent of data: {test_5}")


# In[25]:


model_10 = BERT()
model_10 = nn.DataParallel(model_10, device_ids=[0, 1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_10 = model_10.to(device)


# In[26]:


training_data(model_10, 5, training_data10, learning_rate= 2e-05)


# In[27]:


with torch.no_grad():
    test_10 = testing_data(model_10,test_data)    


# In[28]:


print(f"Accuracy on validation set for 10 percent of data: {test_10}")


# In[29]:


model_50 = BERT()
model_50 = nn.DataParallel(model_50, device_ids=[0, 1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_50 = model_50.to(device)


# In[30]:


training_data(model_50, 5, training_data50, learning_rate= 2e-05)


# In[31]:


with torch.no_grad():
    test_50 = testing_data(model_50,test_data)    


# In[32]:


print(f"Accuracy on validation set for 50 percent of data: {test_50}")


# In[36]:


Accuracies = [22.7, 47.8, 48.7, 50.5]
data_percentage = [0.01, 0.05, 0.1, 0.5]


# In[37]:


plt.plot(data_percentage, Accuracies, marker='o')  
plt.xlabel('Data Percentage')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Data Percentage')
plt.show() 


# In[ ]:




