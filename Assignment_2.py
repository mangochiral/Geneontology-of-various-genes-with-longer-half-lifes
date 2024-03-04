# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:40:55 2024

@author: chand
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math


# To compute Decay constant for each gene
def compute_slope(y):
    '''This function return as slope value for decay constant. No training is done \
    for the dataset because each timecourse gene intensity is independent of \
   other timecourses'''   
    l_reg = LinearRegression()
    x_var = timecourse1['YORF'].values.reshape((-1,1))
    l_reg.fit(x_var, y)
    m = l_reg.coef_[0]
    return m

# To compute Decay constant for each gene at each time course
def t_half(slope):
    base = 2.71
    thalf= math.log(2,base)/slope
    return thalf



#File loaded and transpose rows to columns
df = pd.read_csv('DecayTimecourse.txt', delimiter= '\t')
df = df.transpose()

# Reset columns names the row th zero of the transposed matrix
column_name = df.iloc[0]
df.columns = column_name

# Remove row zero from final data for data clean and imputation
df = df.iloc[1:]

# Replace the index labels as integers
#df.index = df.index.str.replace('Unnamed:', ' ')
df.index = [i for i in range(27)]

# Drop all columns where at zero time there is nan value in either of 
# the timecourse
df = df.dropna(axis=1, subset=[0,9,18])

# Data was separated in 3 timecourse set
timecourse1 = df.iloc[0:9,:]
timecourse2 = df.iloc[9:18,:]
timecourse3 = df.iloc[18:,:]

# Nan type values were in imputed by placing mean values
timecourse1 = timecourse1.fillna(timecourse1.mean())
timecourse2 = timecourse2.fillna(timecourse1.mean())
timecourse3 = timecourse3.fillna(timecourse1.mean())

print(compute_slope.__doc__)

# Slopes for each time course
slopes1 = timecourse1.iloc[:,1:].apply(lambda col: compute_slope(col), axis = 0)
slopes2 = timecourse2.iloc[:,1:].apply(lambda col: compute_slope(col), axis = 0)
slopes3 = timecourse3.iloc[:,1:].apply(lambda col: compute_slope(col), axis = 0)

#Slopes are aggregated to one pandas Dataframe
slopes_pd = pd.DataFrame({'slope_1': slopes1, 'slope_2': slopes2, 'slope_3': slopes3})

# All the genes with decay constant of zero was replaced with nan type
slope_test = slopes_pd.replace(0,np.nan)

# Genes with nan(0)type value in any time course were dropped to remove false positives
slope_test.dropna(axis = 0, inplace = True)

# Half life were calculated for each gene at each time course 
half_life = slope_test.apply(lambda t1_2: t_half(t1_2), axis = 1)

# Average of the half life was calculated for all genes 
mean_half = half_life.mean(axis = 1)

# Find the value at the 90th percentile
threshold_top = mean_half.quantile(0.9)

# Filter the DataFrame to get the top 10 percent
top_10_percent = mean_half[mean_half >= threshold_top]
pd.DataFrame(top_10_percent).to_csv('top_10.csv')

# Find the value at the 10th percentile
threshold_below = mean_half.quantile(0.1)

# Filter the DataFrame to get the bottom 10 percent
bottom_10_percent = mean_half[mean_half <= threshold_below]
pd.DataFrame(bottom_10_percent).to_csv('bottom_10.csv')