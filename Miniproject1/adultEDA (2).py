#!/usr/bin/env python
# coding: utf-8

# In[130]:


import numpy as np
import pandas as pd
from random import randrange
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt

header = ['age','workclass','fnlwgt','education','education_num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
train_df = pd.read_csv('Downloads/adult.data', names = header)
test_df = pd.read_csv('Downloads/adult.test', comment = '|', names = header) #Comment = '|' to ignore first line


# In[3]:


train_df.head()


# In[4]:


train_df.shape


# In[5]:


test_df.head()


# In[6]:


test_df.shape


# In[7]:


adult = pd.concat([test_df, train_df])


# In[8]:


adult.info()


# In[9]:


adult.describe()


# In[10]:


#Check if there is missing values 
adult.isnull().sum() #no NaN 

for i,j in zip(adult.columns,(adult.values.astype(str) == ' ?').sum(axis = 0)):
    if j > 0:
        print(str(i) + ': ' + str(j) + ' missing')    


# In[11]:


all_data = [train_df, test_df]


# In[12]:


#Dropping the missing values
for data in all_data:
    for i in data.columns:
        data[i].replace(' ?', np.nan, inplace=True)
    data.dropna(inplace=True)


# ## Exploratory Analysis
# Compute basic statistics on the data to understand it better. E.g., what are the distributions of the positive vs.
# negative classes, what are the distributions of some of the numerical features? what are the correlations between
# the features? how does the scatter plots of pair-wise features look-like for some subset of features?

# In[13]:


# Reformat income column
adult['income']=adult['income'].map({' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1}) 


# In[14]:


# Count of >50K & <=50K
sns.countplot(adult['income'],label="Count")


# ## Data Preparation

# In[68]:


train_df['income'] = train_df['income'].map({' <=50K': 0, ' >50K': 1})
test_df['income'] = test_df['income'].map({' <=50K.': 0, ' >50K.': 1}) 


# One-hot encode categorical variables (columns 1,3,5,6,7,8,9,13).

# In[70]:


columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
onehotencoder = OneHotEncoder()
# X_train_adult = columnTransformer.fit_transform(train_df).toarray()
# X_test_adult = columnTransformer.fit_transform(test_df).toarray()
train_adult = columnTransformer.fit_transform(train_df).toarray()
test_adult = columnTransformer.fit_transform(test_df).toarray()


# In[21]:


# y_train_adult = train_df.iloc[:,-1]
# y_test_adult = test_df.iloc[:,-1]


# In[24]:


# y_train_adult=y_train_adult.array
# y_test_adult=y_test_adult.array


# In[101]:


def cv_split(dataset, folds): #use 5 folds
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return array(dataset_split)


# In[131]:


# for each fold, take the fold as test set and the remaining folds as training set
dataset = train_adult
folds = 5
acc = []

for i in range(folds):
    test = cv_split(dataset, folds)[i] #one fold (1x6512x109)
    train = np.concatenate([cv_split(dataset, folds)[:i], cv_split(dataset, folds)[i+1:]], axis=0) #(4x6512x109)
    #-----------------------------------------------------------------------------------------------
    # feeding into LR
    model = LogisticRegression(train, percentage=0.6)
    pred = model.predict(train, theta)
    acc[i] = model.get_test_acc(test, test[:,-1], thetas) 
return np.mean(acc)


# In[127]:


class LogisticRegression:

    def __init__(self, data, percentage=0.5):  # percentage = train/cv+test split
        self.data = data
        self.percentage = percentage

        self.train_X, self.train_y, self.sub_X, self.sub_y = self.split_data(self.data, self.percentage)

        self.test_X, self.test_y, self.cv_X, self.cv_y = self.split_data(pd.concat([self.sub_X, self.sub_y], axis=1),
                                                                         0.5)

        self.thetas = self.gradient_descent(self.train_X.values, self.train_y.values, self.cv_X.values,
                                            self.cv_y.values)

        self.testing_accuracy = self.get_test_acc(self.test_X.values, self.test_y.values, self.thetas)

#     def split_data(self, data, percentage=0.5):
#         val = np.random.rand(len(data)) < percentage  # splits data and sorts into x, y values
#         train = data[val]
#         test = data[~val]

#         train_X = train.iloc[:, :-1]
#         train_y = train.iloc[:, -1]

#         test_X = test.iloc[:, :-1]
#         test_y = test.iloc[:, -1]
#         return train_X, train_y, test_X, test_y

    def predict_proba(self, X, theta):
        return self.sigmoid(np.dot(X, theta))

    def predict(self, X, theta):
        prediction = self.predict_proba(X, theta)
        predict_arr = []
        for i in prediction:
            if i >= 0.5:
                predict_arr.append(1)
            else:
                predict_arr.append(0)

        return predict_arr

    def accuracy(self, predict_arr, y):
        correct = 0
        for i, j in zip(predict_arr, y):
            if i == j[0]:
                correct += 1
        return correct / len(y)  # accuracy = # tp+tn / total

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, X, y, theta, lambdaa):  # lambdaa is regularization term
        N, D = len(X[0]), len(X[0])
        yh = self.sigmoid(np.dot(X, theta))
        grad = np.dot(X.T, yh - y) / N
        grad[1:] += lambdaa * theta[1:]
        return grad

    def gradient_descent(self, X, y, cv_X, cv_y, learning_rate=0.01, max_iter=50000, beta=0.99,
                         reg_term=0):  # attempted termination condition - lack of improvement in cross validation set
        N, D = len(X[0]), len(X[0])
        theta = np.zeros((len(X[0]), 1))
        y = np.reshape(y, (-1, 1))  # creates two-dimensional array
        cv_y = np.reshape(cv_y, (-1, 1))
        iterate, cv_acc, prev_cv_acc, d_theta = 0, 0, 0, 0
        max_cv_acc = 0  # maximum cross validation accuracy - records thetas at highest cv_acc
        best_theta = theta
        g = np.inf
        eps = 1e-2
        while (
                cv_acc >= prev_cv_acc - 0.02):  # can add in 'or np.linalg.norm(g) > eps' to stop when gradient becomes too small, 0.03 gives buffer
            g = self.gradient(X, y, theta, reg_term)
            d_theta = (1 - beta) * g + beta * d_theta  # momentum
            theta = theta - learning_rate * d_theta
            cv_pred = self.predict(cv_X, theta)
            prev_cv_acc = cv_acc
            cv_acc = self.accuracy(cv_pred, cv_y)
            if cv_acc > max_cv_acc:  # checks if maximum accuracy thus far
                max_cv_acc = cv_acc
                best_theta = theta
            iterate += 1
            if iterate > max_iter:  # since it may not always converge, place a hard ceiling on number of iterations
                break
        print(max_cv_acc)
        print(cv_acc)
        return best_theta

    def get_test_acc(self, test_X, test_y, thetas):
        test_y = np.reshape(test_y, (-1, 1))

        return 1 - accuracy(predict(test_X, thetas), test_y)


# In[ ]:




