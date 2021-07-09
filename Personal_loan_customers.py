#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings as war


# In[3]:


train=pd.read_csv('Loan_Modelling.csv')
train.head(5)


# In[4]:


train.lt(0).any()


# In[5]:


train.loc[train['Experience']<0,'Experience']=0
train.lt(0).any()


# In[6]:


train.isnull().sum()


# In[7]:


train.head()


# In[10]:


s=train['Age'].value_counts().head(25)
ax=s.plot.bar(width=0.9,color="Green") 
plt.xlabel("Age")
plt.ylabel("Count")
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Age + 1.5, v.Age, color='green',rotation=90)


# In[11]:


s=train['Experience'].value_counts().head(25)
ax=s.plot.bar(width=.9,color="Purple") 
plt.xlabel("Experience")
plt.ylabel("Count")
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Experience + 1.5, v.Experience, color='Purple',rotation=90)


# In[13]:


s=train['Income'].value_counts().head(25)
ax=s.plot.bar(width=.9,color="Indigo") 
plt.xlabel("Income in Dollar")
plt.ylabel("Count")
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Income + 1.5, v.Income, color='Indigo',rotation=90)


# In[17]:


sb.scatterplot(x="Personal_Loan",y="Income",data=train,hue="Personal_Loan")


# In[18]:


maxii=train.loc[(train['Personal_Loan']==1),'Income'].max()
minii=train.loc[(train['Personal_Loan']==1),'Income'].min()
print(maxii)
print(minii)


# In[19]:


sb.lineplot(x="Age",y="Experience",data=train,marker='.',markersize=12,ci=None,color='red')


# In[20]:


sb.lineplot(x="Experience",y="Income",data=train,color='black',ci=None,marker='*',markersize=12)


# In[21]:


sb.lineplot(x="Age",y="Income",data=train,color='green',ci=None,marker='o',markersize=10)


# In[22]:


sb.scatterplot(x="Income",y="Mortgage",hue="Mortgage",data=train)


# In[25]:


sb.stripplot(x="Personal_Loan",y="Mortgage",data=train)


# In[26]:


sb.stripplot(x="Education",y="Mortgage",data=train,hue="Education")


# In[27]:


sb.scatterplot(x="Family",y="Mortgage",data=train,hue="Family")


# In[28]:


s=train.Family[train['Personal_Loan']==1].value_counts().sort_index()
#ax=plot(kind='bar',alpha=0.5,color="Orange")
col=['Red','Green','Indigo','Orange']
ax=s.plot.bar(width=.9,color=col) 
plt.xlabel("Family")
plt.ylabel("Count of taken Personal Loan")
for i, v in s.reset_index().iterrows():
    ax.text(i, v.Family + 1.5, v.Family, color='Indigo')


# In[29]:


train['Active']=0


# In[31]:


train.loc[((train['Securities_Account']==1)|(train['CD_Account']==1)|(train['Online']==1)|(train['CreditCard']==1)),'Active']=1


# In[32]:


train.head()


# In[33]:


sb.stripplot(y="Income",x="Active",data=train)


# In[34]:


train['Response']=0


# In[35]:


train.loc[((train['Family']==3)|(train['Family']==4)),'Response']=1
train.head(10)


# In[36]:


test=pd.DataFrame(train.loc[(((train['Family']==1)|(train['Family']==2))&(train['Mortgage']!=0)),'Mortgage'])


# In[37]:


test.head()


# In[38]:


test.shape


# In[39]:


sb.boxplot(x=test['Mortgage'])


# In[45]:


Q1 = test.quantile(0.25)
Q3 = test.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[46]:


test = test[~((test < (Q1-1.5 * IQR)) |(test > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[47]:


test.shape


# In[48]:


first=Q1-1.5 * IQR
second=Q3 + 1.5 * IQR


# In[49]:


print(first)
print(second)


# In[50]:


sb.boxplot(x=test['Mortgage'])


# In[51]:


train.loc[((train['Mortgage']==0)&(train['Active']==1)),'Response']=1
train.head(10)


# In[52]:


train.loc[(((train['Family']==1)&(train['Family']==2))&((train['Mortgage']>=second[0])&(train['Mortgage']<=first[0]))),'Response']=1


# In[53]:


train.head(10)


# In[54]:


train['Mon_Income']=(train['Income']/12.0)


# In[55]:


train.head(10)


# In[56]:


train['Ultimate']=(train['Mon_Income']-train['CCAvg'])


# In[57]:


train.head(10)


# In[58]:


sb.boxplot(x=train['Ultimate'],color='orange')


# In[59]:


train['Ultimate'].shape


# In[60]:


(train['Response']>0).sum()


# In[61]:


tesst=pd.DataFrame(train['Ultimate'])


# In[62]:


Q1 = tesst.quantile(0.25)
Q3 = tesst.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[63]:


tesst = tesst[~((tesst < (Q1-1.5 * IQR)) |(tesst > (Q3 + 1.5 * IQR))).any(axis=1)]
tesst.head(10)


# In[64]:


fifi=tesst['Ultimate'].max()
sisi=tesst['Ultimate'].min()


# In[65]:


sb.boxplot(x=tesst['Ultimate'],color='indigo')


# In[66]:


tesst.shape


# In[79]:


train.loc[((train['Ultimate']>=fifi)&(train['Ultimate']<=sisi)),'Response']=1


# In[83]:


train.head(10)


# In[84]:


(train['Response']>0).sum()

