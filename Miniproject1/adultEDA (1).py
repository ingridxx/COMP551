#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt

header = ['age','workclass','fnlwgt','education','education_num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
train_df = pd.read_csv('Downloads/adult.data', names = header)
test_df = pd.read_csv('Downloads/adult.test', comment = '|', names = header) #Comment = '|' to ignore first line


# In[2]:


train_df.head()


# In[3]:


train_df.shape


# In[4]:


test_df.head()


# In[5]:


test_df.shape


# In[2]:


adult = pd.concat([test_df, train_df])


# In[7]:


adult.info()


# In[8]:


adult.describe()


# In[3]:


#Check if there is missing values 
adult.isnull().sum() #no NaN 

for i,j in zip(adult.columns,(adult.values.astype(str) == ' ?').sum(axis = 0)):
    if j > 0:
        print(str(i) + ': ' + str(j) + ' missing')    


# In[4]:


all_data = [train_df, test_df]


# In[5]:


#Dropping the missing values
for data in all_data:
    for i in data.columns:
        data[i].replace(' ?', np.nan, inplace=True)
    data.dropna(inplace=True)


# ## Exploratory Analysis
# Compute basic statistics on the data to understand it better. E.g., what are the distributions of the positive vs.
# negative classes, what are the distributions of some of the numerical features? what are the correlations between
# the features? how does the scatter plots of pair-wise features look-like for some subset of features?

# In[6]:


# Reformat income column
adult['income']=adult['income'].map({' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1}) 


# In[7]:


# Count of >50K & <=50K
sns.countplot(adult['income'],label="Count")


# ## Data Preparation

# In[13]:


train_df['income'] = train_df['income'].map({' <=50K': 0, ' >50K': 1})
test_df['income'] = test_df['income'].map({' <=50K.': 0, ' >50K.': 1}) 


# One-hot encode categorical variables (columns 1,3,5,6,7,8,9,13).

# In[17]:


columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
onehotencoder = OneHotEncoder()
X_train_adult = columnTransformer.fit_transform(train_df).toarray()
X_test_adult = columnTransformer.fit_transform(test_df).toarray()


# In[14]:


#X_test_adult = np.array(columnTransformer.fit_transform(test_df), dtype = np.str)


# In[16]:


X_train_adult.shape 


# In[23]:


X_train_adult[0]


# In[18]:


X_test_adult.shape


# In[19]:


y_train_adult = train_df.iloc[:,-1]
y_test_adult = test_df.iloc[:,-1]


# In[20]:


y_train_adult.shape


# In[21]:


y_test_adult.shape


# In[24]:


y_train_adult=y_train_adult.array
y_test_adult=y_test_adult.array


# In[ ]:




