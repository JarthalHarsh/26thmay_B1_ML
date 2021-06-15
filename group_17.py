#!/usr/bin/env python
# coding: utf-8

# ## Weekend project
# #created by
# #Mohit khandelwal
# #Harsh jarthal
# #Mayank jain
# #Himanshu jangid
# #lekhraj paliwal
# #Hitesh khilyani

# In[1]:


import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import tkinter as tk
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')


# In[26]:


df_test = pd.read_csv('test.csv')
df_train= pd.read_csv('train.csv')


# In[27]:


df_test


# In[28]:


df_train


# In[29]:


sns.distplot(df_train['revenue'])


# In[30]:


df_train['City'].value_counts()


# In[31]:


df_train['City Group'].value_counts()


# In[32]:


df_test['City Group'].value_counts()


# In[33]:


df_train['Type'].value_counts()


# In[38]:


df_test['Type'].value_counts()


# In[39]:


df_train['Open Date'].value_counts()


# In[40]:


sns.scatterplot(x="revenue", y="P2", hue="Type", data=df_train)


# In[41]:


Y_train = df_train['revenue']
Y_train.head()


# In[43]:


df_feat = df_train.drop(['revenue'], axis=1)
df_feat = df_feat.drop(['Id'], axis=1)
df_feat = df_feat.drop(['Open Date'], axis=1)
df_feat = df_feat.drop(['City'], axis=1)
df_feat = df_feat.drop(['Type'], axis=1)
df_feat.head()


# In[44]:


df_test = df_test.drop(['Id'], axis=1)
df_test = df_test.drop(['Open Date'], axis=1)
df_test = df_test.drop(['City'], axis=1)
df_test = df_test.drop(['Type'], axis=1)
df_test.head()


# In[46]:


total = df_feat.isnull().sum().sort_values(ascending=False)
percent = (df_feat.isnull().sum()/df_feat.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[48]:


df_feat.columns


# In[49]:


df_pcols = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
       'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35',
       'P36', 'P37']
for i, column in enumerate(df_pcols):
    plt.figure()
    sns.distplot(df_feat[column])


# In[50]:


from sklearn.preprocessing import StandardScaler


# In[51]:


df_feat = pd.get_dummies(df_feat, prefix=['CityGroup'])


# In[52]:


df_test = pd.get_dummies(df_test, prefix=['CityGroup'])


# In[53]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features_train = scaler.fit_transform(df_feat)


# In[54]:


scaled_features_test = scaler.fit_transform(df_test)


# In[55]:


scaled_features_df_feat = pd.DataFrame(scaled_features_train, index=df_feat.index, columns=df_feat.columns)


# In[56]:


scaled_features_df_test = pd.DataFrame(scaled_features_test, index=df_test.index, columns=df_test.columns)


# In[57]:


scaled_features_df_feat.head()


# In[58]:


scaled_features_df_test.head()


# In[59]:


from sklearn.ensemble import RandomForestRegressor


# In[60]:


regr = RandomForestRegressor(n_estimators=20)
regr.fit(scaled_features_df_feat,Y_train)


# In[61]:


predictions = regr.predict(scaled_features_df_test)


# In[64]:


from sklearn.linear_model import Lasso


# In[65]:


regr2 = Lasso(alpha=0.1)
regr2.fit(scaled_features_df_feat,Y_train)


# In[66]:


predictions2 = regr2.predict(scaled_features_df_test)


# In[69]:


df_test1 = pd.read_csv('test.csv')
sub = pd.DataFrame()
sub['Id'] = df_test1['Id']
sub['Prediction'] = predictions
sub.to_csv('submissions.csv',index=False)


# In[70]:


df_test1 = pd.read_csv('test.csv')
sub = pd.DataFrame()
sub['Id'] = df_test1['Id']
sub['Prediction'] = predictions2
sub.to_csv('submissions2.csv',index=False)


# In[71]:


from IPython.display import FileLink
FileLink(r'submissions.csv')


# In[12]:


app = tk.Tk()
app.title('restaurant revenue predictor')
app.geometry('500x600')

tk.Label(app, text='Open Date').place(x=25,y=25)
tk.Label(app, text='City').place(x=25,y=100)
tk.Label(app, text='City Group').place(x=25,y=125)
tk.Label(app, text='Type').place(x=25,y=150)
tk.Label(app, text='Revenue').place(x=25,y=175)

rest_var = tk.Variable(app)
tk.Entry(app, textvariable=rest_var, width=28).place(x=150,y=25)

revenue_var = tk.Variable(app)
revenue_var.set('Type restaurant')
tk.Label(app, textvariable=rest_var).place(x=150,y=100)
r_var = tk.Variable(app)
r_var.set('Type restaurant')
tk.Label(app, textvariable=r_var).place(x=150,y=125)

def find_revenue():
    pass

tk.Button(app, text='Find Revenue', command=find_revenue).place(x=125,y=60)


app.mainloop()


# In[4]:





# In[ ]:




