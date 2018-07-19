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
import seaborn as sns

from tpot import TPOTClassifier
from tpot import TPOTRegressor
from pandas import ExcelWriter


from pandas.util.testing import assert_frame_equal
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score

from sklearn.base import clone

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Bunch of filter functions

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

#Filter functions for binary classification

def ds_2CAT_b (x, threshhold=19):
    if x < threshhold :
        return 0
    else :
        return 1

def ds_cs_b ( x, threshhold = 19 ):
    #Gets minimal and mild patients off the batch
    #Lets undersample
    if x == 0:
        return 0
    elif x > threshhold :
        return 1
    else :
        return np.nan
    
def d_only_b ( x, threshhold = 19 ):
    
    if x>= 19   :  return 1
    else        :  return np.nan


#----------------------------------
      
#/!\ BETA !
# Gets some absurd values out of the batch, corrects them with normalized 
# values              

def repartition (df, marks,name='test') :
    # Used to analyze which brain's region are affected by modifications
    x,i = pd.DataFrame(columns=df.keys()), 0
    #print(marks)
    for val_id_list in marks :
        row = pd.Series(name=str(i))
        row.rename(str(row))
        for key in df.keys():
            row[key]= np.nan
            if str(key) in val_id_list :
                row[key]=True
        #print(row)
        x.loc[i] = row.copy()
        i+=1    
    # Exports the analyzes to csv 
    x.to_csv(path_or_buf=(name + " repartition.csv"))
    x.sum().to_frame().to_csv(path_or_buf=(name + " sommes.csv"))
    
    
    return x.sum()


def mark_and_normalize(df, drop_a,
                       verbosity = 1, 
                       name=None, factor=3): 
    
    datas= df.drop(drop_a,axis=1).copy()
    marks= []
    av_size=[]

    #Median patients value  
    av_size = datas.median(axis=0)
         
    u_a_l_born= {} # Dictionnaire de valeurs relatives aux keys
    for key in datas :
        median = datas[key].median()
        std= datas[key].std()
        u_a_l_born[key] = {'down':(float(median-(factor*std))), 
                               'up':float(median+(factor*std)), 
                               'median': median}
    
    u_a_l_born=pd.DataFrame.from_dict(u_a_l_born)
    median = av_size.median()
    correction, nr_w_datas = 0, 0
    
    # Scans the datas for extreme values
    # These Values are, for each patient, appended to "marks" (list of lists)
    # Marks is sent to "Repartition" for analyses
    # A new value is calculated for the abnormal value ( if clean = TRUE)
    # After the median value of the data and the median value of the patient's
    # data (~)
    # This values are then put into df
    # Original and Modified values are saved in a csv File
    
    for x in datas.iterrows() : 
        
        i=0
        marked, bol = [], False
        
        
        for keys in x[1].keys() :
            
            if (not(u_a_l_born[keys]['down']<  #if value is not in an interval
                   x[1][keys]< 
                   u_a_l_born[keys]['up'])) :   # Or simply way too low 
                
                if verbosity >= 2 :
                    print(u_a_l_born[keys]['down'],' < ' ,
                          x[1][keys],' < ',u_a_l_born[keys]['up'])
                    
                marked.append(keys)             
                                    # Mark Patient, inc amount omp
                if bol==False :
                    nr_w_datas +=1
                    bol = True
                
                ratio = av_size[i]/median     #Normalize the value
                datas[keys][x[0]] = int(ratio*u_a_l_born[keys]['median'])
                
                if verbosity>=2:
                    print('Now : ',
                          int(ratio*u_a_l_born[keys]['median']) )
                correction+=1
            i+=1
        marks.append(marked)
        
    mask = repartition(datas, marks, name) 
    # Produces some Data analyses - no effect 
    # Number of abnormal value per region    
            
    if verbosity>0 :
        
        print("Total values :", datas.shape[0]*datas.shape[1])
        print("Corrected values : ", correction, " - ", \
              correction*100/(datas.shape[0]*datas.shape[1]),"%")
        print("Marked Patients : ", nr_w_datas)
     
    #Keeps track
    df.to_csv(path_or_buf=(name +" original.csv"), index=False)
    for keys in datas.keys() :
        df[keys] = datas[keys].copy()
    datas.to_csv(path_or_buf=(name + " correct.csv"), index=False)

    # Calculates and displays how much area's are having abnorm values
    s = []
    for i in marks:
        for k in i :
            if k not in s:
                s.append(k)
    print("Features with abnormal values : ", len(s))
    
    return df, pd.Series(marks).values, mask # Returns altered, normalized values
    #And a list of every marked / modified value


def analyse_and_clean(df,drop_a=['ID','SCOREBDI', 'Gender','ds','Age'],
                      name = None, clean=False, verbosity = 1, 
                      factor = 3, dynamic_drop=False, ban_thresh = 10):    
    #if clean = true : Gets questionnable patients data out + analyzis
    # Clean = False : Returns original DB with analysis
    
    copy = df.copy()
    results=mark_and_normalize(df,drop_a=drop_a, verbosity=verbosity,
                               name=name, factor = factor)
    datas = results[0].copy()
    marks = results[1].copy()
    mask = results[2].copy()
    datas['marks']=marks
    
    if name!=None :
        
        datas.to_csv(path_or_buf=(name+" abnormal patient datas.csv"), 
                     index=False)
    if clean :
        # Signalize a modification of the Dataset ( experimental )
        print("Imputated")
        copy = datas.drop('marks', axis=1).copy()
    
    if dynamic_drop :
        new_drop = []
        print(mask)
        for key in mask.keys():
            if mask[key] >= ban_thresh :
                new_drop.append(key)
        copy = copy.drop(new_drop,axis=1).copy()
        # Proceeded to anaylze - no modification
    return copy 
    

#----------------------------------

def categorized(df, s_filter = ds_2CAT, th =19):
    # Returns a preprocessed Dataframe with target score
    
    result = df.copy()
    result['ds'] = df.SCOREBDI.apply(s_filter, threshhold=th)
    
    return result.dropna()


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
        
    return np.float64(((Sensitivity * Specificity)*2)/
                       (Sensitivity + Specificity))

def roc_auc_scorer(Y_test, Y_pred, average = 'macro'):
    return roc_auc_score(Y_test, Y_pred,average = 'macro')


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

    data_file_1 = "1000BRAINS_BDI_Score_Vol.csv"
    data_file_2 = "1000BRAINS_BDI_Score_CT.csv"
#    data_file_3 = "
    df= None
    
    if data_choice == 'cv':
        df = pd.read_csv(data_file_1,sep=';').dropna()
    elif data_choice == 'ct':
#        df = pd.read_csv(data_file_2,sep=';').applymap(\
#                lambda x: np.nan if x=='#N/D' else x )
#                
        df = pd.read_csv(data_file_2,sep=',').dropna()
        
    elif data_choice == 'both' :
        df1 = pd.read_csv(data_file_1,sep=';').dropna()
        df2 = pd.read_csv(data_file_2,sep=',').dropna()
        df = pd.merge(df1,df2)
    else :
        raise
        
    return df


#----------------------------------

#TPOT pipeline generating & training function
# The df & training + scoring parameter are saved in a pickle dictionnary
    
def train_tpot(df, gen, pop, name, scorer, save = True,
               proc=1,target='ds', classification=True,
               config_dict=None, rand=23):
        
    features = df.drop(target,axis=1)
    target = df[target]
    
    if save :    
        features.to_csv(path_or_buf=(name + ' features.csv'), index = False)
        target.to_csv(path=(name + ' targets.csv'), index = False)
        
#    X_train, X_test, y_train, y_test = train_test_split(
#            features,target,
#            train_size=0.75, test_size=0.25)
        
    if classification :    
        t_pot = TPOTClassifier(generations=gen, population_size=pop, 
                                   verbosity=2,n_jobs=proc, scoring=scorer, 
                                   config_dict=config_dict, random_state=rand)
    else :
        t_pot = TPOTRegressor(generations=gen, population_size=pop, 
                                   verbosity=2,n_jobs=proc, scoring=scorer,
                                   config_dict=config_dict, random_state=rand)

    t_pot.fit(features, target)
    #t_pot.score(X_test, y_test)
    
    x=t_pot.fitted_pipeline_
    if save :
        
                
        t_pot.export((name+'.py'))
        
        with open((name+'.cfg'), 'wb') as pickle_file:
            pickle.dump({'df': df, 'gen' : gen, 'pop': pop, 'name':name, 
                         'function' : scorer,'X_test':features,
                         'y_test':target, 'model':x }, 
                         pickle_file)

    return x

#----------------------------------
def classifier (x):
    if x==0:
        return 'Not depressive'
    else:
        return 'Depressive'
#----------------------------------
# Score can only be used to quantify Sensitivity and Specificity for DS Scoring

def score(Y_test, Y_pred, verbosity=0):
    #mean_absolute_error(Y_test,Y_pred)
    Y_pred = pd.Series(Y_pred) # pred = pd.Series for character support

    if Y_test.iloc[0]==0 or Y_test.iloc[0]==1:
        Y_test= Y_test.apply(classifier)
        Y_pred= Y_pred.apply(classifier)
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

# Score output doesn't depend on 'score' - can be used for other scores
# Sens and spec dependent on 'score', should only be used for ds 

def score_model(data,model, target = 'ds',
                score=score, verbosity=1, scorer=f1_scorer, 
                return_algorythms=False, cv=3):
    
    X = data.drop(target,axis=1)
    Y = data[target]
    
    if return_algorythms: produced_algorithms=[]
    
    results = pd.DataFrame(columns=['sens', 'spec', 'score']) 
    
    for i in range(10) :
        
        if return_algorythms: produced_algorithms.append([])
        
        kf = KFold(n_splits=cv,shuffle=True)
        exp = pd.DataFrame(columns=['sens', 'spec', 'score'])
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            m = model
            m.fit(X_train,Y_train)
            Y_pred = m.predict(X_test)
        
            out = score(Y_test, Y_pred,verbosity = verbosity)
        
            c= scorer(Y_test, Y_pred)

            exp = exp.append({'sens': out['sens'], 'spec':out['spec'], 
                'score':c}, ignore_index=True)
                
            if return_algorythms: produced_algorithms[i].append(m)

        results = results.append({'sens': exp['sens'].mean(), 
                                  'spec':exp['spec'].mean(), 
                                  'score':exp['score'].mean()},
                                ignore_index=True)
    print(results)   
    ret=pd.DataFrame({'sens': results['sens'].mean(), 
                                'spec':results['spec'].mean(), 
                                'score':results['score'].mean()},index=[0])
    print(ret)
    
    if not(return_algorythms):
        return ret
    else :
        return ret, produced_algorithms
#----------------------------------
    
# gathers all steps of the data alteration

def data_preprocessing(name='test', s_filter=ds_2CAT, 
            drop=['ID','SCOREBDI'], data_choice = 'cv', 
            normalization=False ,clean= False, th=19,
            factor = 3,dynamic_drop=False,ban_thresh = 10, target = 'ds',
            gender = None, age = None, undersample = None, rand = 23):
    
    df = categorized(load_data(data_choice),s_filter=s_filter, th=th).copy()
    
    if target!='Gender':
        df['Gender']= df.Gender.apply(lambda x: 1 if x=='Female' else 0)
    
    #For matching purposes
    if gender != None :
        df = df.loc[(df['Gender'] == gender)]
    if age != None :
        df = df.loc[(df['Age'] <= age[1]) & (df['Age'] >= age[0])]
    if undersample !=None :
        dfd = df.loc[(df['ds'] == ('Depressive')) | (df['ds'] == 1)]
        dfnd= df.loc[(df['ds'] == ('Not depressive')) | (df['ds'] == 0)]
        df = df.sample(frac=1).reset_index(drop=True)

    #Analyzes for absurd values. Imputation possible. Lots of calculations.
    copy = df.copy()
    
    if normalization:
        df = analyse_and_clean (df,name=name, clean = clean, \
            factor = factor, dynamic_drop=dynamic_drop, ban_thresh = 10).copy()
        
        if clean==False and dynamic_drop == False:   
            assert_frame_equal(copy,df,
                        check_exact= True,check_dtype = False, 
                        check_categorical = False, check_names = False
                        )
            
    assert copy.keys().all()==df.keys().all(), \
        "Label added or substracted in spite of Arg "
    

    
    print(df.shape[0],' patients will be used.')
    return df.drop(drop,axis=1)


#----------------------------------

#Creates the evaluation function, shapes the datas and creates + trains the model


def Pipeline_maker(df, name='test', scorer=f1_scorer, classification=True,
                   pop = 500, gen = 10, th = 19,save=True, target='ds',
                   greater_is_better=True, config_dict=None, rand = 23):
    
    score_pipe=make_scorer(scorer, greater_is_better=greater_is_better)
    
    model = train_tpot(df, gen, pop, name, score_pipe, save, target=target,
                       classification=classification,config_dict=config_dict, rand=rand)
    
    return model

        
#------------------------------



def Pipeline_eval (name='test', scorer=f1_scorer,
                   model=None, df= None,
                   save=True, verbosity=1, target='ds', 
                   classification = True, return_algorythms=False, cv=3):
    if classification :
#        if save :
#            with open((name+'.cfg'), 'rb') as pickle_file:
#                dic=pickle.load(pickle_file)    
#                if verbosity >0:
#                    print("Real test : ")
#                Pred = model.predict(dic['X_test'])
#                r1 = score(dic['y_test'],Pred, verbosity =verbosity )
#                r1['score']=scorer(dic['y_test'],Pred)
#        else :
#            r1 = None
#    
        if verbosity >0:
            print(" Reroll : ")
        r2 = score_model(df, model, cv=cv, target=target, verbosity=verbosity, scorer=scorer,return_algorythms=return_algorythms)
    
        if not(return_algorythms): return {'r2':r2}
        else : return {'r2':r2[0]},r2[1] 

#------------------------------
# Matplotlib graph generating functions

def pipeline_results_display(results, name, filename = None, save_d= True, 
                             scorer=f1_scorer):
    
    sns.set()
    plt.axes([0,0,1,1])
    plt.bar(list(results.keys()),list(results.values()), color='b')
    plt.xlabel('Type of score', fontsize=24)
    plt.ylabel('Score', fontsize=24)
    plt.legend
    plt.title(name+' True score')

    if save_d:
        assert filename!=None
        plt.savefig((filename+' tr.png'))
    plt.show()



def pipeline_retest_results_display(results, name, filename = None, 
                                    save_d= True,scorer=f1_scorer):
    
    sns.set()
    df=pd.DataFrame.from_dict(results)

    #plt.axes([0,0,15,1])
    
    plt.bar([str(x+1)+' '+scorer.__name__ for x in range(len(df.iloc[:, 0]))],df.iloc[:, 0], 
             color='b')
    plt.bar([str(x+1)+' Sensitivity' for x in range(len(df.iloc[:, 0]))],df.iloc[:, 1], 
             color='r')
    plt.bar([str(x+1)+' Specificity' 
             for x in range(len(df.iloc[:, 0]))],df.iloc[:, 2], color='c')
    
    
    plt.xlabel('Type of score', fontsize=24)
    plt.ylabel('Score', fontsize=24)
    plt.legend
    plt.title(name+' retest')

    if save_d:
        assert filename!=None
        plt.savefig((filename+'retest .png'))
    plt.xticks(rotation=-45)
    plt.show()

#------------------------------

def assign_filename (name, save):
    file_name=name
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
    return file_name
    

#------------------------------

def Run_experiment (name='test', s_filter=ds_2CAT, scorer=f1_scorer, 
                   drop=['ID','SCOREBDI'], data_choice = 'cv', 
                   pop = 500, gen = 10, th = 19,save=True, display=True, 
                   classification=True, greater_is_better=True,config_dict=None,
                   c_display=1, gender = None, age = None,
                   save_d= True,
                   target='ds',
                   norm= False,
                   clean=False,
                   factor = 3, dynamic_drop=False, ban_thresh = 10, rand = 23, cv = 3):
    z = locals()
    file_name = assign_filename (name, save)
    
    df = data_preprocessing(name=file_name, s_filter=s_filter, 
                            drop=drop, data_choice = data_choice, 
                            normalization=norm ,clean= clean, th=th,
                            factor = factor, dynamic_drop=dynamic_drop,
                            ban_thresh = ban_thresh, gender = gender, age =age )
    
    #Contains the final data_set, ultimately provided to the pipeline making 
    #process
    
    df.to_csv(path_or_buf=(file_name + ' post preprocessing.csv'), index = False)
    
    model = Pipeline_maker (df, name=file_name, scorer=scorer, 
                            pop = pop, gen = gen, th = th,save=save, 
                            classification=classification,target = target,
                            greater_is_better=greater_is_better, config_dict=config_dict, rand=rand)
    
    results = Pipeline_eval(name=file_name, model=clone(model), target = target, 
                            df=df, save=save, classification=classification,
                            verbosity = c_display,scorer=scorer,cv=cv)
    
    if (display and classification) :
#        pipeline_results_display(results['r1'], name, filename=file_name, 
#                                 save_d=save_d, scorer=scorer)
        
        pipeline_retest_results_display(results['r2'], name, 
                                filename=file_name, 
                                save_d=save_d, scorer=scorer)

    
    return results,model


#---------------------------
#    
#df = categorized(load_data('cv'),s_filter=ds_cs_b, th=19) 
#x=normalize(df)
##
##
##
#
#    
#    
#    
    