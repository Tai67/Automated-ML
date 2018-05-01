# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:34:04 2018

@author: Mathieu
"""

# Pipeline-creation set
import tpot
import os
import datetime
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from tpot import TPOTClassifier


from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Bunch of filter functions to diagnose between patients

def ds_2CAT (x, threshhold=19):
    if x < threshhold :
        return 'Not depressive'
    else :
        return "Depressive"

def ds_4CAT (x, threshhold = [10, 19, 29]):
    
    if x < threshhold[0] :
        return 'minimal'
    
    elif threshhold[0] <=x and x < threshhold[1]:
        return "mild"
    
    elif threshhold[1] <= x< threshhold[2] :
        return "moderate"
    
    else :
        return "severe"


def ds_cs ( x, threshhold = 19 ):
    #Gets minimal and mild patients off the batch
    #Lets undersample
    if x == 0:
        return 'Not depressive'
    elif x > threshhold :
        return 'Depressive'
    else :
        return np.nan

#----------------------------------

def categorized(df, s_filter = ds_2CAT, th =19):
    # Returns the final processed Dataframe
    
    result = df.copy()
    result['ds'] = df.SCOREBDI.apply(s_filter, threshhold=th)
    
    if s_filter == ds_cs :
        return result.dropna()
    
    return result

#----------------------------------        
# Bunch of evaluating functions
    
def sens_spec_score(Y_test, Y_pred):
    
    #returns mean value of sens / Spec
    #intended for 2 classes
    Y_test=Y_test.tolist()
    Y_pred=Y_pred.tolist()
    zipped=zip(Y_test,Y_pred)
    n_dep,  pp_diag,n_ndep, np_diag = (0,0,0,0)

    for x in list (zipped):

        if x[0] == 'Depressive':
            n_dep+=1
        else :
            n_ndep+=1

        if x[0]==x[1] and x[1]=='Depressive':
            pp_diag +=1
        elif x[0]==x[1] and x[1]=='Not depressive':
            np_diag +=1

    if n_dep!=0 :
        Sensitivity = np.float64(pp_diag/n_dep)
    else :
        Sensitivity = 0

    if n_ndep!= 0 :    
        Specificity = np.float64(np_diag/n_ndep)
    else :
        Specificity = 0
    score = np.float64((Sensitivity + Specificity)/2)

    return score

def f1_scorer(Y_test, Y_pred, mean_type='macro'):
    #Return f1_score, without balancing between classes
    return f1_score(Y_test,Y_pred,average = mean_type )

def kappa(Y_test, Y_pred, mean_type='macro'):
    #Returns cohen kappa score between classes
    Y_test=[1 if x=='Depressive' else 0 for x in Y_test]
    Y_pred=[1 if x=='Depressive' else 0 for x in Y_pred]
    return cohen_kappa_score(Y_test,Y_pred)

#def s_kappa(Y_test, Y_pred, mean_type='macro',weights='quadratic'):
#    #Returns cohen kappa score between classes
#    return cohen_kappa_score(Y_test,Y_pred,weights=weights )




#----------------------------------

# Returns cv, ct or both combined in one csv
    
# data_choice = 'cv', 'ct' or 'both'
    
def load_data(data_choice = 'cv' ):

    data_file_1 = "1000BRAINS_BDI_Score_CT.csv"
    data_file_2 = "1000BRAINS_BDI_Score_Vol.csv"
    df= None
    
    if data_choice == 'cv':
        df = pd.read_csv(data_file_1,sep=',').dropna()
    elif data_choice == 'ct':
        df = pd.read_csv(data_file_2,sep=';').dropna()
        
    elif data_choice == 'both' :
        df1 = pd.read_csv(data_file_1,sep=',').dropna()
        df2 = pd.read_csv(data_file_2,sep=';').dropna()
        df = pd.merge(df1,df2)
    else :
        raise

    return df


#----------------------------------

#TPOT pipeline generating & training function
# The df & training + scoring parameter are saved in a pickle dictionnary
    
def train_tpot(df, gen, pop, name, scorer, drop, save = True,
               proc=1):
        
    features = df.drop(drop,axis=1)
    target = df.ds
        
    X_train, X_test, y_train, y_test = train_test_split(
            features,target,
            train_size=0.75, test_size=0.25)
        
        
    t_pot = TPOTClassifier(generations=gen, population_size=pop, 
                               verbosity=2,n_jobs=proc, scoring=scorer 
                               )

    t_pot.fit(X_train, y_train)
    #t_pot.score(X_test, y_test)
    
    x=t_pot.fitted_pipeline_
    if save :
        
        features.to_csv(name + ' features.csv', index = False)
        target.to_csv(name + ' targets.csv', index = False)
        
                
        t_pot.export((name+'.py'))
        
        with open((name+'.cfg'), 'wb') as pickle_file:
            pickle.dump({'df': df, 'gen' : gen, 'pop': pop, 'name':name, 
                         'function' : scorer, 'drop' : drop,'X_test':X_test,
                         'y_test':y_test, 'model':x }, pickle_file)

    return x

    
#----------------------------------

def score(Y_test, Y_pred, verbosity=0):
    #mean_absolute_error(Y_test,Y_pred)
    Y_test=Y_test.tolist()
    Y_pred=Y_pred.tolist()
    zipped=zip(Y_test,Y_pred)
    Sensitivity, Specificity = (0,0)
    n_dep,  pp_diag,n_ndep, np_diag = (0,0,0,0)
    
    for x in list (zipped):
        #print(x[0],x[1])
        if x[0] == 'Depressive':
            n_dep+=1
        else :
            n_ndep+=1

        if x[0]==x[1] and x[1]=='Depressive':
            pp_diag +=1
        elif x[0]==x[1] and x[1]=='Not depressive':
            np_diag +=1
    
    if n_dep!=0 :
        Sensitivity = np.float64(pp_diag/n_dep)
    else :
        Sensitivity = np.nan
        
    if n_ndep!= 0 :    
        Specificity = np.float64(np_diag/n_ndep)
    else :
        Specificity = np.nan
        
    if verbosity ==1 :
        print ("Patients dépressifs:", n_dep,"Patients non dépressifs:",n_ndep)
        print("Diagnostics positifs corrects :",pp_diag)
        print("Diagnostics négatifs corrects :",np_diag)
        print("Sensitivity = ",Sensitivity,"Specificity =",Specificity,"\n\n")
        


    return {'sens':Sensitivity, 'spec':Specificity}



def score_model(data,model, drop=['ID','SCOREBDI', 'Gender','ds'],
                score=score, verbosity=1, scorer=f1_scorer):
    
    X = data.drop(drop,axis=1)
    Y = data.ds
    
    
    kf = KFold(n_splits=2,shuffle=True)
    Results = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        m = model
        m.fit(X_train,Y_train)
        Y_pred = m.predict(X_test)
        
        out = score(Y_test, Y_pred,verbosity = verbosity)
        
        c= scorer(Y_test, Y_pred)
        Results.append({'sens': out['sens'], 'spec':out['spec'], 'score':c})
    return Results



#----------------------------------

#Creates the evaluation function, shapes the datas and creates + trains the model


def Pipeline_maker(name='test', s_filter=ds_2CAT, scorer=f1_scorer, 
                   drop=['ID','SCOREBDI', 'Gender','ds'], data_choice = 'cv', 
                   pop = 500, gen = 10, th = 19,save=True):
    
    score_pipe=make_scorer(scorer)
    df = categorized(load_data(data_choice),s_filter=s_filter, th=th) 
    #print(df.shape)
    model = train_tpot(df, gen, pop, name, score_pipe, drop, save)
    
    return model, df

        
#------------------------------



def Pipeline_eval (name='test', scorer=f1_scorer, 
                   drop=['ID','SCOREBDI', 'Gender','ds'],model=None, df= None,
                   save=True, verbosity=1):
    
    if save :
        with open((name+'.cfg'), 'rb') as pickle_file:
            dic=pickle.load(pickle_file)    
        if verbosity >0:
            print("Real test : ")
        Pred = model.predict(dic['X_test'])
        r1 = score(dic['y_test'],Pred, verbosity =verbosity )
        r1['score']=scorer(dic['y_test'],Pred)
    else :
        r1 = None
    
    if verbosity >0:
        print(" Reroll : ")
    r2 = score_model(df, model,verbosity=verbosity)
    

    return {'r1': r1,'r2':r2}

#------------------------------
# Matplotlib graph generating functions

def pipeline_results_display(results, name, filename = None, save_d= True, 
                             scorer=f1_scorer):
    

    plt.axes([0,0,1,1])
    plt.bar(list(results.keys()),list(results.values()), color='b')
    plt.xlabel('Type of score')
    plt.ylabel('Score')
    plt.legend
    plt.title(name+' True score')

    if save_d:
        assert filename!=None
        plt.savefig((filename+' tr.png'))
    plt.show()



def pipeline_retest_results_display(results, name, filename = None, 
                                    save_d= True,scorer=f1_scorer):
    
    df=pd.DataFrame.from_dict(results)

    #plt.axes([0,0,15,1])
    
    plt.bar([str(x+1)+' '+scorer.__name__ for x in range(len(df.iloc[:, 0]))],df.iloc[:, 0], 
             color='b')
    plt.bar([str(x+1)+' Sensitivity' for x in range(len(df.iloc[:, 0]))],df.iloc[:, 1], 
             color='r')
    plt.bar([str(x+1)+' Specificity' 
             for x in range(len(df.iloc[:, 0]))],df.iloc[:, 2], color='c')
    
    
    plt.xlabel('Type of score')
    plt.ylabel('Score')
    plt.legend
    plt.title(name+' retest')

    if save_d:
        assert filename!=None
        plt.savefig((filename+'retest .png'))
    plt.xticks(rotation=-45)
    plt.show()



#------------------------------

def Run_experiment (name='test', s_filter=ds_2CAT, scorer=f1_scorer, 
                   drop=['ID','SCOREBDI', 'Gender','ds'], data_choice = 'cv', 
                   pop = 500, gen = 10, th = 19,save=True, display=True, 
                   c_display=1,
                   save_d= True):
    
    if save :
    
        file_name = './'+ name+'/'+name    
        if not os.path.exists(name):
            os.makedirs(name)
            
        else :
            file_id = str(datetime.datetime.now())
            file_id = file_id.replace('.','_')
            file_id = file_id.replace(' ','-')
            file_id = file_id.replace(':','_')
            file_id = file_id[:-7]

            os.makedirs((name+' '+file_id))
            file_name = './'+ name+' '+file_id+'/'+name
    
    model, df = Pipeline_maker (name=file_name, s_filter=s_filter, 
                                scorer=scorer, 
                                drop=drop, data_choice = data_choice, 
                                pop = pop, gen = gen, th = th,save=save)
    
    results = Pipeline_eval(name=file_name, model=model, df=df, save=save, 
                            verbosity = c_display,scorer=scorer)
    
    if display :
        pipeline_results_display(results['r1'], name, filename=file_name, 
                                 save_d=save_d, scorer=scorer)
        
        pipeline_retest_results_display(results['r2'], name, filename=file_name, 
                                 save_d=save_d, scorer=scorer)

    
    return results



