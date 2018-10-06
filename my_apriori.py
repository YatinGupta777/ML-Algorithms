#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:46:00 2018

@author: yatingupta
"""

#Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
'''Header = None means no column headings but first row is also data'''
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)
'''data needs to be a list of list not a dataframe for input'''

transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#Traning apriori on dataset
from apyori import apriori 
rules = apriori(transactions,min_support = 0.003,min_confidence = 0.2,min_lift = 3,min_length = 2)

#Visualizing the results
results = list(rules)
