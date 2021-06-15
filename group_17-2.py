#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tkinter as tk
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('train.csv')
df.head()


# In[3]:


X = df[['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37']]


# In[4]:


X.shape


# In[5]:


Y = df[['revenue']]


# In[6]:


lenc = LabelEncoder()


# In[7]:


typ = df[['Type']]
city_group = df[['City Group']]


# In[10]:


type_lenc = lenc.fit_transform(typ)
ge_le = lenc.fit_transform(city_group)


# In[12]:


type_lenc = type_lenc.reshape(-1,1)
ge_le = ge_le.reshape(-1,1)


# In[14]:


type_lenc.shape


# In[15]:


type_lenc = pd.DataFrame(type_lenc)
ge_le = pd.DataFrame(ge_le)


# In[21]:


x = pd.concat([ge_le,type_lenc,X],axis=1)


# In[22]:


x.shape


# In[24]:


model = LinearRegression()
model.fit(X,Y)


# In[25]:


model.coef_


# In[26]:


model.intercept_


# In[27]:


predict = model.predict([['0','2','4','5','3','5','2','2','5','4','4','5','4','4','5','0','0','0','0','0','2','1','1','1','1','0','0','0','0','3','3','0','0','0','0','0','0','0','0']])


# In[29]:


ac = (predict/6363241)*100


# In[30]:


ac


# In[31]:


x.corr()


# In[32]:


predict


# In[33]:


df['Type'].hist(bins=20)


# In[34]:


df['City'].hist(bins=20)


# In[35]:


for i in df.columns[5:-1]:
    df[i].hist(bins=20)
    plt.title(i)
    plt.show()


# In[36]:


df['revenue'].hist(bins=20)


# In[37]:


df['City Group'].hist(bins=20)


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25, random_state = 10)


# In[ ]:


xtrain.info()


# In[ ]:


xtest.info()


# In[ ]:


ytrain.shape


# In[ ]:


ytest.shape


# In[ ]:


print('%.1f%%'%(model.score(xtest, ytest) * 100))


# In[ ]:


91.5%


# In[ ]:




