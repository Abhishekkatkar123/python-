#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14,8))


# In[2]:


df=pd.read_csv("USA_Housing.csv")
df.head()


# In[3]:


# Checking for Null Values
df.info()


# In[4]:


# summary of Data
df.describe().T


# In[5]:


# Dropping Address Column
df.drop(['Address'],axis=1,inplace=True)
df.head()


# In[6]:


sns.pairplot(df)


# In[7]:


sns.heatmap(df.corr(),annot=True)


# In[8]:


X = df[df.columns[~df.columns.isin(['Price'])]].values
y = df[df.columns[df.columns.isin(['Price'])]].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)


# In[9]:


from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()

X = df[df.columns[~df.columns.isin(['Price'])]]
X = pd.DataFrame(pre_process.fit_transform(X))

y = df[df.columns[df.columns.isin(['Price'])]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)
model.score(X_test,y_test)


# In[10]:


y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

from math import sqrt

rms = sqrt(mse)
rms


# In[11]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
print('root mean sqared error:',rms)


# In[ ]:




