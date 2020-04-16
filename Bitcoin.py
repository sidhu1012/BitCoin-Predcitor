#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bit_df=pd.read_csv('D:\\coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
bit_df.head()


# In[3]:


bit_df.size


# In[4]:


bit_df


# In[5]:


bit_df=bit_df.dropna()
bit_df


# In[6]:


df=bit_df[['Open','Close','Volume_(BTC)','Volume_(Currency)','Weighted_Price']]


# In[7]:


X=df[['Open','Close','Volume_(BTC)','Volume_(Currency)']].values
X


# In[8]:


Y=df[['Weighted_Price']].values


# In[13]:


from sklearn.preprocessing import MinMaxScaler
X=MinMaxScaler().fit_transform(X)
X


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=5)


# In[15]:


from sklearn import linear_model
regr=linear_model.LinearRegression().fit(x_train,y_train)


# In[16]:


y_hat=regr.predict(x_test)


# In[18]:


mse=np.mean((y_hat-y_test)**2)
print(mse)


# In[20]:


vs=regr.score(X,Y)
print(vs)


# In[ ]:




