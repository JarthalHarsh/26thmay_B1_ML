#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.head()


# In[3]:


dataset.info()


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset.shape


# In[6]:


x1 = dataset.iloc[:, 0].values
y1 = dataset.iloc[:, 4].values
plt.scatter(x1,y1,color='Orange',s=50)
plt.xlabel('UserID')
plt.ylabel('Purchased')
plt.title('UserID vs Purchased')
plt.show()


# In[7]:


x1 = dataset.iloc[:, 1].values
y1 = dataset.iloc[:, 4].values
plt.scatter(x1,y1,color='pink',s=50)
plt.xlabel('Gender')
plt.ylabel('Purchased')
plt.title('Gender vs Purchased')
plt.show()


# In[8]:


x1 = dataset.iloc[:, 2].values
y1 = dataset.iloc[:, 4].values
plt.scatter(x1,y1,color='purple',s=50)
plt.xlabel('Age')
plt.ylabel('Purchased')
plt.title('Age vs Purchased')
plt.show()


# In[9]:


x1 = dataset.iloc[:, 3].values
y1 = dataset.iloc[:, 4].values
plt.scatter(x1,y1,color='red',s=50)
plt.xlabel('Estimatedsalary')
plt.ylabel('Purchased')
plt.title('Estimatedsalary vs Purchased')
plt.show()


# In[10]:


import seaborn as sns
plt.figure(figsize=(7,4)) #7 is the size of the width and 4 is parts.... 
sns.heatmap(dataset.corr(),annot=True,cmap='cubehelix_r') 


# In[11]:


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


# In[12]:


print(X)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


log = LinearRegression()


# In[19]:


log.fit(X_train,y_train)


# In[20]:


y_pred = log.predict(X_test)


# In[21]:


log.score(X_test,y_test)*100


# In[ ]:




