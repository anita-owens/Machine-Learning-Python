#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:32:53 2020

@author: anitaowens
"""



# =============================================================================
# Dataset can be found at: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
# 
# Data Set Information:
# 
# This research aimed at the case of customers default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods
#. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default will be more valuable than the binary result of 
#classification - credible or not credible clients. Because the real probability of default is unknown, this study presented the novel Sorting Smoothing Method
# to estimate the real probability of default. With the real probability of default as the response variable (Y), and the predictive probability of default as the 
# independent variable (X), the simple linear regression result (Y = A + BX) shows that the forecasting model produced by artificial neural network has the highest
# coefficient of determination; its regression intercept (A) is close to zero, and regression coefficient (B) to one. Therefore, among the six data mining techniques,
# artificial neural network is the only one that can accurately estimate the real probability of default.
# 
# Attribute Information:
# 
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
# X2: Gender (1 = male; 2 = female).
# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# X4: Marital status (1 = married; 2 = single; 3 = others).
# X5: Age (year).
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
# X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
# 
# =============================================================================


# Setup workspace

from os import chdir, getcwd
wd=getcwd()
chdir(wd)

wd # check working directory

import os
abspath = os.path.abspath("/Users/anitaowens/Documents/GitHub/Machine-Learning-Python") ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory

wd # check working directory


#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt 
import seaborn as sns


#Import data - CSV file
df = pd.read_csv('datasets/default-of-credit-card-clients.csv')

#Check results
print(df.head())


#Check structure of dataset & see if anything is missing
print(df.info())


#Calculate summary statistics for numeric variables
print(df.describe())


#get column names
list(df.columns)



# =============================================================================
#  Initial Exploratory Analysis
# =============================================================================

#check for null values
count_nan = df.isnull().sum()
print (count_nan[count_nan>0])

#Check for duplicates
dups = df.duplicated()

#count dups
dups.value_counts()




figure, ax = plt.subplots(4,2, figsize=(12,24))

#See the distribution of the data
sns.distplot(df['LIMIT_BAL'],ax= ax[0,0])
sns.countplot(df['MARRIAGE'],ax= ax[0,1])

sns.countplot(df['SEX'],ax=ax[1,0])
sns.countplot(df['EDUCATION'],ax= ax[1,1])


sns.distplot(df['AGE'],ax= ax[2,0])
sns.countplot(df['default payment next month'],ax= ax[2,1])






