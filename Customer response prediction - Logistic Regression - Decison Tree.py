#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 08:35:53 2020

@author: anitaowens
"""

############ Marketing customer response prediction using decision trees ################


# =============================================================================
# The classification goal is to predict if the client will subscribe (yes/no) a
# term deposit (variable y).
# 
# In this article, we will see how we could harness the power of machine
# learning to target the campaigns towards the right set of customers, thereby
# increasing conversion propensity. We will be using past data available on the
# demography, bank details, and transaction patterns of customers who have
# responded and not responded to a campaign, as training data to
# predict the probability if a customer will respond to the campaign.
# 
# 
# =============================================================================

# Go to System Preferences > Keyboard > check "use f1, f2 keys as standard fn keys"
# Spyder IDE shortcuts - run selection or current line F9


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

#Import data - CSV file
df = pd.read_csv('datasets/bank-full.csv', delimiter=";")

#Check results
print(df.head())


#Check structure of dataset
print(df.info())


#Calculate summary statistics for numeric variables
print(df.describe())



#what was the response rate? of our target variable - the y column
print(df.y.value_counts()) #just the counts
print(df.y.value_counts(normalize=True)) #with percentage


target_tab  = print(pd.crosstab(index = df["y"],
                              columns="count"))



