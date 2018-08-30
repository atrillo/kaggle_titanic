# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 21:22:05 2018

@author: Antonio
"""

# Importing stuff =================

# data analysis and wrangling
#import pandas as pd
#import numpy as np
#import random as rnd
#
## visualization
#import seaborn as sns
#import matplotlib.pyplot as plt
##%matplotlib inline
#
## machine learning
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC, LinearSVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import SGDClassifier
#from sklearn.tree import DecisionTreeClassifier

# Loading data from Kaggle ds ==================

#train_df = pd.read_csv('../data/train.csv')
#test_df = pd.read_csv('../data/test.csv')
#combine = [train_df, test_df]

print(train_df.columns.values)
# Print 10 first observations
print (train_df.head(10))

train_df.info()
print('_'*40)
test_df.info()

train_df.describe()
list(train_df.PassengerId)
train_df.describe(include=['O'])

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Pclass', ascending=True)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='SibSp', ascending=True)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=True)

print (train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
