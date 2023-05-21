#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_excel(r'/Users/praneethkumarpalepu/Downloads/Name_Gender-2.xlsx')


# In[3]:


df.head()


# In[4]:


name = df['name']


# In[5]:


gender = df['gender']


# In[6]:


gender.value_counts()


# In[7]:


df.isnull().sum()


# In[8]:


df.dropna(subset=['name'], inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.shape


# In[11]:


gender = df['gender']


# In[12]:


gender.value_counts()


# In[13]:


df.head()


# In[14]:


df['name'] = df['name'].apply(lambda x:x.lower())


# In[15]:


df['name'] = df['name'].apply(lambda x:re.sub(r'[^a-z]', '', x))


# In[16]:


df['name']


# In[48]:


def get_array_from_names(name_series, chars_to_use=8):
    updated_name = []
    for name in name_series:
        if len(name)>=chars_to_use:
            first_name = name[:int((chars_to_use/2))]
            last_name = name[-int((chars_to_use/2)):]
            new_name = first_name+last_name
            updated_name.append(new_name)
        
        else:
            zeros_to_pad = chars_to_use - len(name)
            padded_value = ''
            for i in range(zeros_to_pad):
                padded_value +='0'
            
            new_name = padded_value + name
            updated_name.append(new_name)
    
    return updated_name
    


# In[51]:


alphabet = '0abcdefghijklmnopqrstuvwxyz'
numbers = [i for i in range(0, 27)]


# In[52]:


alpha_values = dict(zip(alphabet, numbers))


# In[49]:


updated_name = get_array_from_names(df['name'], 8)


# In[53]:


def get_feature_array_from_updated_names(updated_names):
    total_names = []
    for name in updated_names:
        ind_name = []
        for char in name:
            ind_name.append(alpha_values[char])
        
        total_names.append(ind_name)
    
    feature_array = np.array(total_names)
    
    return feature_array


# In[20]:


total_names = []
for name in updated_name:
    ind_name = []
    for char in name:
        ind_name.append(alpha_values[char])
    total_names.append(ind_name)    
    


# In[21]:


total_names = np.array(total_names)


# In[22]:


total_names


# In[23]:


lr = LogisticRegression()


# In[24]:


len(total_names)


# In[25]:


30172*0.2


# In[26]:


xtr, xte, ytr, yte = train_test_split(total_names, gender, test_size=0.2, random_state=0)


# In[27]:


xtr.shape


# In[28]:


len(ytr)


# In[29]:


lr.fit(xtr, ytr)


# In[30]:


lr_pred = lr.predict(xte)


# In[31]:


accuracy_score(yte, lr_pred)


# In[78]:


rfc = RandomForestClassifier(n_estimators=250, random_state=1)


# In[79]:


rfc.fit(xtr, ytr)


# In[80]:


rfc_pred = rfc.predict(xte)


# In[81]:


accuracy_score(yte, rfc_pred)


# In[59]:


test_name = ['radha', 'kiran', 'fathima', 'varun', 'kishore']


# In[60]:


feature_names = get_array_from_names(test_name, 8)


# In[61]:


feature_names


# In[62]:


feature_array = get_feature_array_from_updated_names(feature_names)


# In[63]:


feature_array


# In[64]:


rfc.predict(feature_array)


# In[65]:


lr.predict(feature_array)


# In[ ]:


total_names = []
for name in df['name']:
     ind_name = []
     for char in name:
          ind_name.append(alpha_values[char])
     total_names.append(ind_name)

