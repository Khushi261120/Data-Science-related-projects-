#!/usr/bin/env python
# coding: utf-8

# ## Problem statement:To create a model which can predict the amount.

# In[1]:


### Import the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("TaxiFare.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


sns.countplot(x="no_of_passenger",data=df)


# In[8]:


df = df[df['no_of_passenger'] == 1]
df = df.drop(['unique_id', 'no_of_passenger'], axis=1)
df.head()


# In[9]:


df.shape


# In[10]:


corr_matrix = df.corr()
corr_matrix['amount'].sort_values(ascending=False)


# In[11]:


import datetime
from math import sqrt

for i, row in df.iterrows():
    dt = datetime.datetime.strptime(row['date_time_of_pickup'], '%Y-%m-%d %H:%M:%S UTC')
    df.at[i, 'day_of_week'] = dt.weekday()
    df.at[i, 'pickup_time'] = dt.hour
    x = (row['longitude_of_dropoff'] - row['longitude_of_pickup']) * 54.6 # 1 degree == 54.6 miles
    y = (row['latitude_of_dropoff'] - row['latitude_of_pickup']) * 69.0   # 1 degree == 69 miles
    distance = sqrt(x**2 + y**2)
    df.at[i,'distance'] = distance
    
df.head()


# In[12]:


df= df.drop(['date_time_of_pickup'], axis=1)


# In[13]:


df.head()


# In[14]:


df.describe()


# In[15]:


df = df[(df['distance'] > 1.0) & (df['distance'] < 10.0)]
df = df[(df['amount'] > 0.0) & (df['amount'] < 50.0)]
df.shape


# In[16]:


plt.figure(figsize=(10,10))
corr = df.corr()
sns.heatmap(corr, annot= True)


# In[17]:


#33 lets build our linear model
# independent variables 
X =df.drop(['amount'],axis=1)
#THE dpendent variable
y =df[['amount']]


# In[18]:


#split X and y into training and test set in 70:30 ratio
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.30,random_state=0)


# In[19]:


model_1= LinearRegression()
model_1.fit(X_train,y_train)


# In[20]:


model_1.score(X_train,y_train)


# In[21]:


## out of sample score (R^2)

model_1.score(X_test,y_test)


# In[37]:


from sklearn.ensemble import RandomForestRegressor
rfcl = RandomForestRegressor(n_estimators = 150, random_state=0,max_features=3)
rfcl = rfcl.fit(X_train, y_train)
y_predict = rfcl.predict(X_test)
print(rfcl.score(X_test, y_test))



# In[24]:


from sklearn.ensemble import GradientBoostingRegressor
gbcl = GradientBoostingRegressor(n_estimators = 200,random_state=0)
gbcl = gbcl.fit(X_train, y_train)
y_predict = gbcl.predict(X_test)
print(gbcl.score(X_test, y_test))
print(gbcl.score(X_train, y_train))


# In[ ]:




