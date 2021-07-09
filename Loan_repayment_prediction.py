#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


loans = pd.read_csv('loan_data.csv')
loans.head()


# In[3]:


loans.info()


# In[4]:


loans.describe()


# In[5]:


sns.set_style('darkgrid')
plt.hist(loans['fico'].loc[loans['credit.policy']==1], bins=30, label='Credit.Policy=1')
plt.hist(loans['fico'].loc[loans['credit.policy']==0], bins=30, label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[6]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=30, alpha=0.5, color='blue', label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30, alpha=0.5, color='green', label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[7]:


plt.figure(figsize=(12,6))
sns.countplot(data=loans, x='purpose', hue='not.fully.paid')


# In[8]:


plt.figure(figsize=(10,6))
sns.jointplot(x='fico', y='int.rate', data=loans)


# In[9]:


sns.lmplot(data=loans, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid', palette='Set2')


# In[10]:


loans.head()


# In[11]:


purpose_c = pd.get_dummies(loans['purpose'], drop_first=True)
loans_f = pd.concat([loans, purpose_c], axis=1).drop('purpose', axis=1)
loans_f.head()


# In[12]:


from sklearn.model_selection import train_test_split
y = loans_f['not.fully.paid'] 
X = loans_f.drop('not.fully.paid', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


log = LinearRegression()


# In[15]:


log.fit(X_train,y_train)


# In[16]:


y_pred = log.predict(X_test)


# In[18]:


log.score(X_test,y_test)*1000


# In[ ]:




