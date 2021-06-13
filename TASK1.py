#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np


# In[92]:


df=pd.read_excel('Task1.xlsx')
df


# In[93]:


#Costliest Food
max_value=df['PRICE(RS)'].max()
for i in range(0,len(df.index)):
    if df['PRICE(RS)'][i]==max_value:
        print(df["NAMES"][i])


# In[106]:


#Cheapest Sweet
filter=df['TASTE']=='sweet'
min_value=df['PRICE(RS)'].where(filter).min()
print(df.loc[np.where((df['TASTE']=='sweet') & (df['PRICE(RS)']==min_value))]['NAMES'])


# In[107]:


#White Food Found In Jaipur
print(df.loc[np.where((df['COLOR']=='white') & (df['PLACE']=='Jaipur'))]['NAMES'])


# In[123]:


#Which Cousine is most popular
print(df['CUISINE'].value_counts()[df['CUISINE'].value_counts() == df['CUISINE'].value_counts().max()])


# In[125]:


#Counts of different tastes for foods
print(df['TASTE'].value_counts()[:5].sort_values(ascending=False))


# In[127]:


#Food which is available in Indore and taste is sweet
print(df.loc[np.where((df['TASTE']=='sweet') & (df['PLACE']=='Indore'))]['NAMES'])


# In[ ]:




