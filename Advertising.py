#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


ad_data = pd.read_csv('advertising.csv')


# In[30]:


ad_data.head()


# In[31]:


ad_data.info()


# In[32]:


ad_data.describe()


# In[34]:


plt.hist(ad_data['Age'],bins = 30)
plt.xlabel('Age')
plt.ylabel('Values')
plt.show()


# In[35]:


sns.jointplot(ad_data['Age'],ad_data['Area Income'])


# In[36]:


sns.jointplot(ad_data['Age'],ad_data['Daily Time Spent on Site'],kind = 'kde')


# In[37]:


sns.jointplot(ad_data['Daily Time Spent on Site'], ad_data['Daily Internet Usage'])


# In[38]:


sns.pairplot(ad_data,hue_order  = "Clicked on Ad")


# In[39]:


ad_data.head()


# In[40]:


ad_data.drop(['Ad Topic Line','City','Country','Timestamp'],axis = 1,inplace = True)


# In[41]:


X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]


# In[42]:


y = ad_data['Clicked on Ad']


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[48]:


from sklearn.linear_model import LinearRegression


# In[49]:


log = LinearRegression()


# In[50]:


log.fit(X_train,y_train)


# In[51]:


y_pred = log.predict(X_test)


# In[53]:


log.score(X_test,y_test)*100


# In[ ]:




