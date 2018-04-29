# -*- coding: utf-8 -*-

"""
Created on Sat Apr 28 23:07:02 2018

@author: Mathieu
"""


from Pipeline_creator import Run_experiment
from Pipeline_creator import ds_cs

x, model_x = Run_experiment(name='Experiment CV', s_filter=ds_cs,
                   drop=['ID','SCOREBDI', 'Gender','ds'], data_choice = 'cv', 
                   pop = 500, gen = 20, th = 19,save=True)

y, model_y = Run_experiment(name='Experiment CT', s_filter=ds_cs,  
                   drop=['ID','SCOREBDI', 'Gender','ds'], data_choice = 'ct', 
                   pop = 500, gen = 20, th = 19,save=True)

z, model_z = Run_experiment(name='Experiment BOTH', s_filter=ds_cs,  
                   drop=['ID','SCOREBDI', 'Gender','ds'], data_choice = 'both', 
                   pop = 500, gen = 20, th = 19,save=True)
