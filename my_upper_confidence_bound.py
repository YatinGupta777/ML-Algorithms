#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:27:11 2018

@author: yatingupta
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB from scratch
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
'''vector of size with only 0s'''
sums_of_rewards = [0]* d
total_reward = 0 
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if(numbers_of_selections[i] > 0):
            average_reward  =sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad]+reward
    total_reward  = total_reward + reward
    
#Visualizing the results
plt.hist(ads_selected)
plt.title("Ad Selections")
plt.xlabel("Ads")
plt.ylabel("Number of times selected")
plt.show()    
    