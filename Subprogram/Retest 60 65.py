# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:22:09 2018

@author: Mathieu
"""


import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib as plt

from Pipeline_creator import Pipeline_eval
from Pipeline_creator import data_preprocessing
from Pipeline_creator import f1_scorer
from Pipeline_creator import ds_cs
from Pipeline_creator import ds_cs_b
from Pipeline_creator import ds_2CAT
from Pipeline_creator import d_only_b


from Pipeline_creator import kappa

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from numpy.random import seed
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.cluster import FeatureAgglomeration
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_validate
from sklearn.base import clone

seed(23)

name='RE Undersample All 55 60'
classification=True
scorer=kappa
greater_is_better=True
gender = None  
s_filter=ds_2CAT
drop=['ID','SCOREBDI']
target='ds'
data_choice = 'both'
config_dict=None
pop = 250
gen = 20
th = 19
save=True
save_d=True
c_display=0
norm=False
clean=False
dynamic_drop = False
factor = 3
ban_thresh = 10
                   
age = [55,60]          

df = data_preprocessing(name=name, s_filter=s_filter, 
                            drop=drop, data_choice = 'cv', 
                            normalization=norm ,clean= clean, th=th,
                            factor = factor, dynamic_drop=dynamic_drop,
                            ban_thresh = ban_thresh, gender=gender, age=age)

model = GradientBoostingClassifier(learning_rate=1.0, max_depth=8, max_features=0.7000000000000001, min_samples_leaf=4, min_samples_split=2, n_estimators=100, subsample=0.35000000000000003)

#model for 55-60 population : 
#model = make_pipeline(
#    make_union(
#        FunctionTransformer(copy),
#        Normalizer(norm="l1")
#    ),
#    DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=5, min_samples_split=7)
#)

#model for 65-70 population
#model = make_pipeline(
#    make_union(
#        FunctionTransformer(copy),
#        make_pipeline(
#            FeatureAgglomeration(affinity="cosine", linkage="average"),
#            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
#        )
#    ),
#    GaussianNB()
#)

#test = make_pipeline(
#    make_union(
#        FunctionTransformer(copy),
#        Normalizer(norm="l1")
#    ))

x,y = Pipeline_eval(name='Test.jpg', model=clone(model), target = target, df=df, 
              save=save, classification=classification,verbosity = c_display,scorer=scorer,return_algorythms=True, cv=10)

z= cross_validate( model, df.drop(['ds'],axis=1), y=df['ds'])

#exported_pipeline.fit(training_features, training_target)
#results = exported_pipeline.predict(testing_features)

