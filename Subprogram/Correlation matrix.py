# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 03:17:23 2018

@author: Mathieu
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib as plt

from Pipeline_creator import data_preprocessing
from Pipeline_creator import f1_scorer
from Pipeline_creator import ds_cs
from Pipeline_creator import ds_cs_b
from Pipeline_creator import ds_2CAT_b
from Pipeline_creator import d_only_b

from sklearn.model_selection import train_test_split

import seaborn as sns

def correlation_matrix_sns(matrix, title='Default'):
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.pyplot.subplots(figsize=(20, 20))
    ax.set_title(title)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

drop=['ID','SCOREBDI']
target = 'ds'
name='test'
data_choice = 'cv'
norm =False
clean = False
th=19
factor = 10
dynamic_drop=False
ban_thresh=10
batch_size = 10

df_do = data_preprocessing(name=name, s_filter=d_only_b, 
                            drop=drop, data_choice = 'cv', 
                            normalization=norm ,clean= clean, th=th,
                            factor = factor, dynamic_drop=dynamic_drop,
                            ban_thresh = ban_thresh)


df_cv = data_preprocessing(name=name, s_filter=ds_2CAT_b, 
                            drop=drop, data_choice = 'cv', 
                            normalization=norm ,clean= clean, th=th,
                            factor = factor, dynamic_drop=dynamic_drop,
                            ban_thresh = ban_thresh)

df_ct = data_preprocessing(name=name, s_filter=ds_2CAT_b, 
                            drop=drop, data_choice = 'ct', 
                            normalization=norm ,clean= clean, th=th,
                            factor = factor, dynamic_drop=dynamic_drop,
                            ban_thresh = ban_thresh)

df_cv.loc[(df_cv['Age'] > 55) & (df_cv['Age'] <= 60) & (df_cv['Gender'] == 1)]

df_do.loc[(df_do['Age'] <= 65) & (df_do['Age'] >= 60) & (df_do['Gender'] == 1)]
df_cv.loc[(df_cv['Age'] <= 65) & (df_cv['Age'] >= 60) & (df_cv['Gender'] == 1) & (df_cv['ds'] == 0)]

df_do.loc[(df_do['Age'] <= 70) & (df_do['Age'] >= 65) & (df_do['Gender'] == 1)]

#nd_df   = df.loc[df['ds']== 0]
#d_df    = df.loc[df['ds']== 1]
#
#difference = nd_df.corr().sub(d_df.corr())
#correlation_matrix_sns(difference, 'Correlation matrix by healthy patients \n with Correlation matrix by depressive patiens')
#
#correlation_matrix_sns(df.corr(), 'Correlation matrix on complete population')
#
#correlation_matrix_sns(d_df.corr(), 'Correlation matrix by depressive patients')
#correlation_matrix_sns(nd_df.corr(), 'Correlation matrix by healthy patients')
 
