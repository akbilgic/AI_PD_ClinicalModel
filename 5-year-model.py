#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_recall_curve
from datetime import datetime, date, time, timedelta
from statsmodels.stats import weightstats as stests

from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from scipy.stats.mstats import gmean

import math
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from scipy.stats.mstats import gmean


# In[2]:


year_cut = 1000

features = pd.read_excel('HAAS HRV 2017 dataset_brief_oz.xlsx')
features = features.replace(to_replace=' ', value=np.nan, regex=True)
features


# In[3]:


features2 = features[features['Ex5_year'].notna()]
features2


# In[4]:


features2['PD_ILB_control'].value_counts()


# In[5]:


interested = features2[features2['Ex5_year'] <= features2['CensorYear']]
interested


# In[6]:


interested['PD_ILB_control'].value_counts()


# In[7]:


(interested['CensorYear']  - interested['Ex5_year']).value_counts()


# In[8]:


cases= interested[interested['PD_ILB_control']==2]
cases


# In[9]:


ex2 = cases[(cases['CensorYear']  - cases['Ex5_year']) >year_cut].index
len(ex2)


# In[10]:


(cases['CensorYear']  - cases['Ex5_year']).value_counts()


# In[11]:


dataset_newx  = interested.drop(ex2)
dataset_newx


# In[12]:


dataset_newx['PD_ILB_control'].value_counts()


# In[13]:


allcontrols = dataset_newx[dataset_newx['PD_ILB_control']!=2]
allcontrols


# In[14]:


allcontrols['Autopsy'].value_counts()


# In[15]:


autops_no = allcontrols[allcontrols['Autopsy']==0]
autops_no


# In[16]:


autops_yes = allcontrols[allcontrols['Autopsy']==1]
autops_yes


# In[17]:


exlude_no_aut = autops_no.index
exlude_no_aut


# In[18]:


dataset_newxx  = dataset_newx.drop(exlude_no_aut)
dataset_newxx


# In[19]:


dataset_newxx['PD_ILB_control'].value_counts()


# In[20]:


dataset_new = dataset_newxx[dataset_newxx['PD_ILB_control']!=1]
dataset_new


# In[21]:


dataset_new['Autopsy'].value_counts()


# In[22]:


dataset_new['PD_ILB_control'].value_counts()


# In[23]:


dataset_new['PD_ILB_controlx'] = dataset_new['PD_ILB_control'].replace(1,0)


autop_lwy = dataset_new[['Autopsy' , 'LEWYBDY']]
labels = list((dataset_new['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0


print (features2.shape)

print (dataset_new.shape)


print (np.sum(labels))


# In[24]:


dataset_new['PD_ILB_control'].value_counts()


# In[25]:


dataset_new['PD_ILB_controlx'].value_counts()


# In[26]:


#######################################################################################################

def concat_pd(df, vr1, vr2):
    return df[vr1].fillna(0) + df[vr2] * (1 * pd.isnull(df[vr1]))

#######################################################################################################

dataset_new['BM_frequency_Ex4'] = dataset_new['BM_frequency_Ex4'].replace([1, 2, 3, 4, 5, 7], [2, 3, 4, 1, 0, 0])
# dataset_new['BM_frequency_Ex4'].value_counts()

dataset_new['BM_frequency_Ex3'] = dataset_new['BM_frequency_Ex3'].replace([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 3, 4])
# dataset_new['BM_frequency_Ex3'].value_counts()

dataset_new['new_BM_frequency'] = concat_pd(dataset_new, 'BM_frequency_Ex4', 'BM_frequency_Ex3')
print (dataset_new['new_BM_frequency'].value_counts())

# ((dataset_new['SMKX4']==2) * 1 ).value_counts()

dataset_new['new_smoke']  = (((dataset_new['SMKX4']==2) * 1) + dataset_new['Smoke_ever'])
print (dataset_new['new_smoke'] .value_counts())

#######################################################################################################

predictors_names = ['Sleepy_complete','Simple_reaction_time_Ex5', 'new_BM_frequency',
                            'Choice_reaction_time_Ex5','CASIX4','Olfaction_complete_Ex5_Ex4', 'ECGage','new_smoke',
                              'TBI_LOC',
                        'BMIX4','HYPBP1', 'Coffee_oz_Ex1' ]

predictors = dataset_new[predictors_names]

predictors_columns = [x for x in predictors.columns]


# In[27]:


(dataset_new['CensorYear']  - dataset_new['Ex5_year']).value_counts()


# In[28]:


from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from scipy.stats.mstats import gmean

################################

def xgb_evaluate(num_leaves,min_data_in_leaf, learning_rate,max_depth,feature_fraction,bagging_freq, bagging_fraction,
                lambda_l1, lambda_l2, max_bin):
    
    
    skf_spl = 5
    folds_spl = 5

    param = {'task': 'train',
             'boosting': 'gbdt',
             'objective':'binary',
             'metric': 'auc',
             'num_leaves': int(num_leaves),
             'min_data_in_leaf': int(min_data_in_leaf),
             'learning_rate': learning_rate,
             'max_depth': int(max_depth),
             'feature_fraction': feature_fraction,
             'bagging_freq': int(bagging_freq),
             'bagging_fraction': bagging_fraction,
             'use_missing': True,
             'nthread': 4,
             'lambda_l1': lambda_l1,
             'lambda_l2': lambda_l2,
             'max_bin': int(max_bin)
            }

    print (param)

    skf = StratifiedKFold(n_splits=skf_spl, shuffle=True, random_state=256)


    prediction_validation = []          #np.zeros(len(predictors)*skf_spl)
    true_validation=[]          #np.zeros(len(predictors)*skf_spl)

    prediction_hidden=[]          #np.zeros(len(predictors))
    true_hidden=[]          #np.zeros(len(predictors))
    
    
    oof_hidden = np.zeros(len(predictors))
    oof_val = np.zeros(len(predictors))
    
    for fold_cv_, (cv_idx, hid_idx) in enumerate(skf.split(predictors, labels)):
        strLog = "fold_cv_ {}".format(fold_cv_)
        #print(strLog)

        df_tr = predictors.iloc[cv_idx]
        target= pd.DataFrame(labels).iloc[cv_idx]


        x_hidden = predictors.iloc[hid_idx]
        hidden_test = pd.DataFrame(labels).iloc[hid_idx]

        folds = StratifiedKFold(n_splits=folds_spl, shuffle=True, random_state=256)

        hid_pred = []
        oof_val_i = np.zeros(len(df_tr))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr,target.values)):
            strLog = "fold {}".format(fold_)


            trn_data = lgb.Dataset(df_tr.iloc[trn_idx], label=target.iloc[trn_idx])
            val_data = lgb.Dataset(df_tr.iloc[val_idx], label=target.iloc[val_idx],reference=trn_data)

            num_round = 20000
            clf = lgb.train(param,trn_data,num_round,valid_sets=val_data,early_stopping_rounds=300,verbose_eval=False
                            #categorical_feature=categorical_index
                           )


            prediction_hidden.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            true_hidden.append(hidden_test[0].values)

            hidauc = roc_auc_score(pd.DataFrame(hidden_test), clf.predict(x_hidden, num_iteration=clf.best_iteration))
            hid_pred.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            #print(strLog)
            #print (str(fold_cv_) + str(fold_) + ' hidden set auc is '+str(hidauc))
            


            prediction_validation.append( clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            true_validation.append( target.iloc[val_idx][0].values)


            a=roc_auc_score(target.iloc[val_idx],clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            #print (str(fold_cv_) + str(fold_) + ' valid auc is ' + str(a))
            oof_val_i[val_idx] = clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration)
        oof_hidden[hid_idx] = np.mean(hid_pred, axis = 0)
        oof_val[cv_idx] += oof_val_i

    oof_val = oof_val / skf_spl
################################
    strAUC = roc_auc_score(labels, oof_hidden)
    print('strAUC hidden is ' + str(strAUC) )

    strAUC_val = roc_auc_score(labels, oof_val)
    print('strAUC val  is ' + str(strAUC_val) )
################################


    print ('done')
    fpr, tpr, thresholds = roc_curve(labels, oof_val)
    
    auc_ss = np.full(fpr.shape, strAUC_val)
    
    gmeans = gmean([tpr ,  auc_ss])
    

    
    
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    
    
    return gmeans[ix] #strAUC_val # 




gp_params = {"alpha": 1e-4}




xgb_bo = BayesianOptimization(xgb_evaluate, {'num_leaves': (5, 30), 
                                             'min_data_in_leaf': (1, 5.),
                                             'learning_rate':(0.005, 0.1),
                                             'max_depth': (5, 200),
                                             'feature_fraction': (0.3, 1.),
                                             'bagging_freq': (1, 4.),
                                             'bagging_fraction':(0.3, 1.),
                                             'lambda_l1': (0, 1),
                                             'lambda_l2': (0, 1),
                                             'max_bin': (10,300)
                                             
                                            })
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=75, n_iter=25, acq='ei',  **gp_params)


# In[28]:


autops_no


# In[29]:


autops_no['PD_ILB_control'].value_counts()


# In[30]:


dataset_ILB = dataset_newxx[dataset_newxx['PD_ILB_control']==1]
dataset_ILB


# In[31]:


autops_no['BM_frequency_Ex4'] = autops_no['BM_frequency_Ex4'].replace([1, 2, 3, 4, 5, 7], [2, 3, 4, 1, 0, 0])
# autops_no['BM_frequency_Ex4'].value_counts()

autops_no['BM_frequency_Ex3'] = autops_no['BM_frequency_Ex3'].replace([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 3, 4])
# autops_no['BM_frequency_Ex3'].value_counts()

autops_no['new_BM_frequency'] = concat_pd(autops_no, 'BM_frequency_Ex4', 'BM_frequency_Ex3')
print (autops_no['new_BM_frequency'].value_counts())

# ((autops_no['SMKX4']==2) * 1 ).value_counts()

autops_no['new_smoke']  = (((autops_no['SMKX4']==2) * 1) + autops_no['Smoke_ever'])


print (autops_no['new_smoke'] .value_counts())

#######################################################################################################

autops_no_predictors = autops_no[predictors_names]


# In[32]:


dataset_ILB['BM_frequency_Ex4'] = dataset_ILB['BM_frequency_Ex4'].replace([1, 2, 3, 4, 5, 7], [2, 3, 4, 1, 0, 0])
# dataset_ILB['BM_frequency_Ex4'].value_counts()

dataset_ILB['BM_frequency_Ex3'] = dataset_ILB['BM_frequency_Ex3'].replace([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 3, 4])
# dataset_ILB['BM_frequency_Ex3'].value_counts()

dataset_ILB['new_BM_frequency'] = concat_pd(dataset_ILB, 'BM_frequency_Ex4', 'BM_frequency_Ex3')
print (dataset_ILB['new_BM_frequency'].value_counts())

# ((dataset_ILB['SMKX4']==2) * 1 ).value_counts()

dataset_ILB['new_smoke']  = (((dataset_ILB['SMKX4']==2) * 1) + dataset_ILB['Smoke_ever'])


print (dataset_ILB['new_smoke'] .value_counts())

#######################################################################################################

dataset_ILB_predictors = dataset_ILB[predictors_names]


# In[33]:


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr - (1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    #print (roc)
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    #print (roc_t)
    return list(roc_t['threshold']) 



def auc_conf_int(n1, n2, AUC, z=1.959964):
    q0 = AUC*(1-AUC)
    q1 = (AUC/(2-AUC))-AUC**2
    q2 = (2*(AUC**2)/(1+AUC))-AUC**2
    se = math.sqrt((q0+(n1-1)*q1+(n2-1)*q2)/(n1*n2))
    return (AUC-se*z, AUC+se*z)




def modeling(predictors, labels,  param, predictors_columns):
    
    skf_spl = 5
    folds_spl = 5



    skf = StratifiedKFold(n_splits=skf_spl, shuffle=True, random_state=256)


    prediction_validation = []          #np.zeros(len(predictors)*skf_spl)
    true_validation=[]          #np.zeros(len(predictors)*skf_spl)

    prediction_hidden=[]          #np.zeros(len(predictors))
    true_hidden=[]          #np.zeros(len(predictors))
    sens_hid=[]
    feat = []


    oof_hidden = np.zeros(len(predictors))
    oof_val = np.zeros(len(predictors))


    oof_dataset_ILB_predictors = []
    oof_autops_no_predictors = []

    for fold_cv_, (cv_idx, hid_idx) in enumerate(skf.split(predictors, labels)):
        strLog = "fold_cv_ {}".format(fold_cv_)
        print(strLog)

        df_tr = predictors.iloc[cv_idx]
        target= pd.DataFrame(labels).iloc[cv_idx]
        df_tr_autop_lwy = autop_lwy.iloc[cv_idx]


        x_hidden = predictors.iloc[hid_idx]
        hidden_test = pd.DataFrame(labels).iloc[hid_idx]
        x_hidden_autop_lwy = autop_lwy.iloc[hid_idx]

        folds = StratifiedKFold(n_splits=folds_spl, shuffle=True, random_state=256)

        hid_pred = []

        oof_val_i = np.zeros(len(df_tr))


        for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr,target.values)):
            strLog = "fold {}".format(fold_)


            trn_data = lgb.Dataset(df_tr.iloc[trn_idx], label=target.iloc[trn_idx])
            val_data = lgb.Dataset(df_tr.iloc[val_idx], label=target.iloc[val_idx],reference=trn_data)

            num_round = 20000
            clf = lgb.train(param,trn_data,num_round,valid_sets=val_data,early_stopping_rounds=300,verbose_eval=200
                            #categorical_feature=categorical_index
                           )


            hid_pred.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            # prediction_hidden.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            # true_hidden.append(hidden_test[0].values)
            sens_hid.append(x_hidden_autop_lwy)
            feat.append(clf.feature_importance(importance_type='gain'))

            hidauc = roc_auc_score(pd.DataFrame(hidden_test), clf.predict(x_hidden, num_iteration=clf.best_iteration))

            print(strLog)
            print (str(fold_cv_) + str(fold_) + ' hidden set auc is '+str(hidauc))


            oof_val_i[val_idx] = clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration)

            prediction_validation.append( clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            true_validation.append( target.iloc[val_idx][0].values)


            a=roc_auc_score(target.iloc[val_idx],clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            print (str(fold_cv_) + str(fold_) + ' valid auc is ' + str(a))
            
            oof_dataset_ILB_predictors.append(clf.predict(dataset_ILB_predictors, num_iteration=clf.best_iteration))
            oof_autops_no_predictors.append(clf.predict(autops_no_predictors, num_iteration=clf.best_iteration))
            
        oof_hidden[hid_idx] = np.mean(hid_pred, axis = 0)
        oof_val[cv_idx] += oof_val_i

    
    
    res_oof_dataset_ILB_predictors = np.mean(oof_dataset_ILB_predictors, axis = 0)
    res_oof_autops_no_predictors = np.mean(oof_autops_no_predictors, axis = 0)
    
    oof_val = oof_val / skf_spl

    strAUC = roc_auc_score(labels, oof_hidden)
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    ci1, ci2 = auc_conf_int(n1, n2, strAUC)
    print('strAUC hidden is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    

    strAUC = roc_auc_score(labels, oof_val)
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    ci1, ci2 = auc_conf_int(n1, n2, strAUC)    
    
    print('strAUC val is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    # Add prediction probability to dataframe

    # Find optimal probability threshold
    threshold = Find_Optimal_Cutoff(labels, oof_hidden)
    print (threshold)
    thr =threshold[0]
    print ('Threshold  is ' + str(thr))
    
    binary_hidden = np.array([1 if i > thr else 0 for i in oof_hidden])
    print ('Hidden accuracy_score : ', accuracy_score(labels, binary_hidden)) 
    print (classification_report(labels, binary_hidden))

    CM = confusion_matrix(labels, binary_hidden)
    print ('CM : ',  CM)
    TN, FP, FN, TP = (confusion_matrix(labels, binary_hidden).ravel())
    
    
    print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )
    feature_importance = np.mean(feat, axis=0)
    feature_importance


    # Example data
    feature_names_arg = tuple(np.asarray(predictors_columns)[np.argsort(-feature_importance)])

    y_pos = np.arange(len(feature_names_arg))
    performance = -np.sort(-feature_importance)
    
    print ('here, Threshold  is ' + str(thr))
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    
    ci1, ci2 = auc_conf_int(n1, n2, ACC)    
    
    print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    ci1, ci2 = auc_conf_int(n1, n2, TPR)    
    
    print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    ci1, ci2 = auc_conf_int(n1, n2, TNR)    
    
    print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    f1 = f1_score(labels, binary_hidden)
    
    ci1, ci2 = auc_conf_int(n1, n2, f1)  
    print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    ci1, ci2 = auc_conf_int(n1, n2, PPV)    
    
    print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')

    return oof_hidden, oof_val, feature_names_arg, y_pos, performance, thr, res_oof_dataset_ILB_predictors, res_oof_autops_no_predictors

#######################################################################################################

param ={'task': 'train', 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'num_leaves': 12, 'min_data_in_leaf': 2, 'learning_rate': 0.02684974744087402, 'max_depth': 114, 'feature_fraction': 0.33675033067694526, 'bagging_freq': 1, 'bagging_fraction': 0.9561829614963597, 'use_missing': True, 'nthread': 4, 'lambda_l1': 0.6625827702282564, 'lambda_l2': 0.3532048637416306, 'max_bin': 147}








oof_hidden, oof_val, feature_names_arg, y_pos, performance, thr, ILB_pred, no_autop_pred = modeling(predictors, labels,  param, predictors_columns)

#######################################################################################################


# In[36]:


binary_ILB = np.array([1 if i > thr else 0 for i in ILB_pred])
binary_ILB


# In[37]:


binary_no_aut = np.array([1 if i > thr else 0 for i in no_autop_pred])
binary_no_aut


# In[38]:


no_autop_pred


# In[39]:


oof_hidden


# In[40]:


Base_Predicted_Probs_3_year = pd.Series(oof_hidden, index =dataset_new.index )
Base_Predicted_Probs_3_year


# In[41]:


Base_hidden_prediction_pd = pd.DataFrame({'HHP':dataset_new[ 'HHP' ],
                                     'Predicted_Probs_3_year': Base_Predicted_Probs_3_year,
                                     'PD_ILB_control': dataset_new[ 'PD_ILB_control' ],
                                     'PD_ILB_controlx': dataset_new[ 'PD_ILB_controlx' ],
                                    'DENLV4':dataset_new[ 'DENLV4' ] , 'EKG_to_Censor': dataset_new['EKG_to_Censor'],
                                   'Incidence_years_PD': dataset_new['Incidence_years_PD'],'Ex5_year' : dataset_new[ 'Ex5_year'], 
                                   'CensorYear': dataset_new[ 'CensorYear' ], 'YRDX':dataset_new[ 'YRDX' ],
                                     'ECGX4YY': dataset_new['ECGX4YY'],
                                     'Autopsy':dataset_new['Autopsy'],
                                     'LEWYBDY': dataset_new['LEWYBDY']
                                   }).set_index(dataset_new.index)



# In[42]:


Base_hidden_prediction_pd.shape


# In[43]:



ILB_Predicted_Probs_3_year = pd.Series(ILB_pred, index =dataset_ILB.index )
ILB_Predicted_Probs_3_year


# In[44]:


dataset_ILB[ 'PD_ILB_control' ].value_counts()


# In[45]:


dataset_ILB[ 'PD_ILB_controlx' ] = dataset_ILB[ 'PD_ILB_control' ]


# In[46]:


ILB_hidden_prediction_pd = pd.DataFrame({'HHP':dataset_ILB[ 'HHP' ],
                                     'Predicted_Probs_3_year': ILB_Predicted_Probs_3_year,
                                     'PD_ILB_control': dataset_ILB[ 'PD_ILB_control' ],
                                     'PD_ILB_controlx': dataset_ILB[ 'PD_ILB_controlx' ],
                                    'DENLV4':dataset_ILB[ 'DENLV4' ] , 'EKG_to_Censor': dataset_ILB['EKG_to_Censor'],
                                   'Incidence_years_PD': dataset_ILB['Incidence_years_PD'],'Ex5_year' : dataset_ILB[ 'Ex5_year'], 
                                   'CensorYear': dataset_ILB[ 'CensorYear' ], 'YRDX':dataset_ILB[ 'YRDX' ],
                                     'ECGX4YY': dataset_ILB['ECGX4YY'],
                                     'Autopsy':dataset_ILB['Autopsy'],
                                     'LEWYBDY': dataset_ILB['LEWYBDY']
                                   }).set_index(dataset_ILB.index)


# In[47]:



autops_no_Predicted_Probs_3_year = pd.Series(no_autop_pred, index =autops_no.index )
autops_no_Predicted_Probs_3_year


# In[48]:


autops_no[ 'PD_ILB_control' ].value_counts()


# In[49]:


autops_no[ 'PD_ILB_controlx' ] = autops_no['PD_ILB_control']


# In[50]:


autops_no_hidden_prediction_pd = pd.DataFrame({'HHP':autops_no[ 'HHP' ],
                                     'Predicted_Probs_3_year': autops_no_Predicted_Probs_3_year,
                                     'PD_ILB_control': autops_no[ 'PD_ILB_control' ],
                                     'PD_ILB_controlx': autops_no[ 'PD_ILB_controlx' ],
                                    'DENLV4':autops_no[ 'DENLV4' ] , 'EKG_to_Censor': autops_no['EKG_to_Censor'],
                                   'Incidence_years_PD': autops_no['Incidence_years_PD'],'Ex5_year' : autops_no[ 'Ex5_year'], 
                                   'CensorYear': autops_no[ 'CensorYear' ], 'YRDX':autops_no[ 'YRDX' ],
                                     'ECGX4YY': autops_no['ECGX4YY'],
                                     'Autopsy':autops_no['Autopsy'],
                                     'LEWYBDY': autops_no['LEWYBDY']
                                   }).set_index(autops_no.index)


# In[51]:


Last_hidden_prediction_pd = pd.concat([Base_hidden_prediction_pd, ILB_hidden_prediction_pd, autops_no_hidden_prediction_pd])
Last_hidden_prediction_pd


# In[52]:


Last_hidden_prediction_pd['PD_ILB_control'].value_counts()


# In[53]:


Base_LB_Unknownbig = Last_hidden_prediction_pd[Last_hidden_prediction_pd['PD_ILB_control']== 0]
Base_LB_No = Base_LB_Unknownbig[(Base_LB_Unknownbig['Autopsy'] == 1)]
Base_LB_Unknown = Base_LB_Unknownbig.drop(Base_LB_No.index)
Base_LB_Yes = Last_hidden_prediction_pd[(Last_hidden_prediction_pd['PD_ILB_controlx']==1)]

Cases_1 = Last_hidden_prediction_pd[Last_hidden_prediction_pd['PD_ILB_control']==2]


# In[54]:


Base_LB_Unknownbig.shape, Base_LB_Unknown.shape, Base_LB_No.shape, Base_LB_Yes.shape, Cases_1.shape


# In[55]:



Base_LB_Unknown_pd = pd.DataFrame({'HHP_c':Base_LB_Unknown[ 'HHP' ],
                                    'label':['Base_LB_Unknown' for i in range(Base_LB_Unknown.shape[0])]
                                    }).set_index(Base_LB_Unknown.index)

Base_LB_No_pd = pd.DataFrame({'HHP_c':Base_LB_No[ 'HHP' ],
                                    'label':['Base_LB_No' for i in range(Base_LB_No.shape[0])]
                                    }).set_index(Base_LB_No.index)
Base_LB_Yes_pd = pd.DataFrame({'HHP_c':Base_LB_Yes[ 'HHP' ],
                                    'label':['Base_LB_Yes' for i in range(Base_LB_Yes.shape[0])]
                                    }).set_index(Base_LB_Yes.index)



Cases_1_pd = pd.DataFrame({'HHP_c':Cases_1[ 'HHP' ],
                                    'label':['Cases_1' for i in range(Cases_1.shape[0])]
                                    }).set_index(Cases_1.index)


# In[56]:


labelize = pd.concat([Base_LB_Unknown_pd, Base_LB_No_pd, Base_LB_Yes_pd, Cases_1_pd]).sort_index()
labelize


# In[57]:


Last_hidden_prediction_pd['labelize'] = labelize['label']
Last_hidden_prediction_pd


# In[58]:


Last_hidden_prediction_pd.to_excel("Model_4_Last_hidden_prediction_pd.xlsx")


# In[59]:


Last_hidden_prediction_pd['PD_ILB_controlx'].value_counts()


# In[60]:


Last_hidden_prediction_pd['labelize'].value_counts()


# In[61]:


Base_STAT_LB_No = Last_hidden_prediction_pd[(Last_hidden_prediction_pd['labelize'] == 'Cases_1') |  (Last_hidden_prediction_pd['labelize'] == 'Base_LB_No') ]
Base_STAT_LB_No


# In[62]:


labels_new = list((Base_STAT_LB_No['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = Base_STAT_LB_No['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for Base_STAT_LB_No is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[63]:


Base_STAT_LB_Unknown = Last_hidden_prediction_pd[(Last_hidden_prediction_pd['labelize'] == 'Cases_1') |  (Last_hidden_prediction_pd['labelize'] == 'Base_LB_Unknown') ]
Base_STAT_LB_Unknown


# In[64]:


labels_new = list((Base_STAT_LB_Unknown['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = Base_STAT_LB_Unknown['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_Unknown is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[65]:


Base_STAT_NLB = Last_hidden_prediction_pd[(Last_hidden_prediction_pd['labelize'] == 'Cases_1') |  (Last_hidden_prediction_pd['labelize'] == 'Base_LB_Unknown')  |  (Last_hidden_prediction_pd['labelize'] == 'Base_LB_No') ]
Base_STAT_NLB


# In[66]:


labels_new = list((Base_STAT_NLB['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = Base_STAT_NLB['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_NLB is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[67]:


Last_hidden_prediction_pd['PD_ILB_controlx'].value_counts()


# In[68]:


Last_hidden_prediction_pd['PD_ILB_controlx'].value_counts()


# In[69]:


Last_hidden_prediction_pd['PD_ILB_controlx']= Last_hidden_prediction_pd['PD_ILB_control'].replace(1,0)
Last_hidden_prediction_pd['PD_ILB_controlx'].value_counts()


# In[70]:


labels_new = list((Last_hidden_prediction_pd['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = Last_hidden_prediction_pd['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for Last_hidden_prediction_pd is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# # second model

# In[71]:


dataset_ILBx =dataset_ILB.copy()


# In[72]:


dataset_ILBx['pred_binary'] = binary_ILB
dataset_ILBx


# In[73]:


dataset_ILB_case = dataset_ILBx[dataset_ILBx['pred_binary']==1]
dataset_ILB_control = dataset_ILBx[dataset_ILBx['pred_binary']==0]


# In[74]:


dataset_ILB_case


# In[75]:


dataset_ILB_case['PD_ILB_controlx'] = dataset_ILB_case['PD_ILB_control'].replace(1,2)
dataset_ILB_control['PD_ILB_controlx'] = dataset_ILB_control['PD_ILB_control'].replace(1,0)


# In[76]:


dataset_new['PD_ILB_control'].value_counts()


# In[77]:


dataset_new['PD_ILB_controlx'].value_counts()


# In[78]:


dataset_new.shape


# In[79]:


dataset_ILB_case


# In[80]:


dataset_ILB_control


# In[81]:


dataset_ILB_case = dataset_ILB_case.drop(['pred_binary'], axis=1)
dataset_ILB_control = dataset_ILB_control.drop(['pred_binary'], axis=1)


# In[82]:


dataset_ILB_case


# In[83]:


autops_no['PD_ILB_controlx'] = autops_no['PD_ILB_control']


# In[84]:


autops_no['PD_ILB_controlx'].value_counts()


# In[85]:


dataset_new['PD_ILB_controlx'].value_counts()


# In[86]:


dataset_ILB_control['PD_ILB_controlx'].value_counts()


# In[87]:


dataset_ILB_case['PD_ILB_controlx'].value_counts()


# In[88]:


dataset_new_with_ILB = pd.concat([dataset_new, dataset_ILB_control, dataset_ILB_case, autops_no])
dataset_new_with_ILB


# In[89]:


list(dataset_new_with_ILB['PD_ILB_controlx'])


# In[90]:


dataset_new_with_ILB['PD_ILB_controlx'].value_counts()


# In[91]:


dataset_new_with_ILB['PD_ILB_control'].value_counts()


# In[92]:


year_cut = 5

ex2 = cases[(cases['CensorYear']  - cases['Ex5_year']) >year_cut].index
len(ex2)

dataset_new_with_ILB  = dataset_new_with_ILB.drop(ex2)


# In[93]:


dataset_new_with_ILB['PD_ILB_control'].value_counts()


# In[94]:


dataset_new_with_ILB['PD_ILB_controlx'].value_counts()


# In[95]:


autop_lwy = dataset_new_with_ILB[['Autopsy' , 'LEWYBDY']]
labels = list((dataset_new_with_ILB['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0



print (dataset_new_with_ILB.shape)


print (np.sum(labels))


# In[96]:



predictors = dataset_new_with_ILB[predictors_names]

predictors_columns = [x for x in predictors.columns]


# In[95]:


from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from scipy.stats.mstats import gmean

################################

def xgb_evaluate(num_leaves,min_data_in_leaf, learning_rate,max_depth,feature_fraction,bagging_freq, bagging_fraction,
                lambda_l1, lambda_l2, max_bin):
    
    
    skf_spl = 5
    folds_spl = 5

    param = {'task': 'train',
             'boosting': 'gbdt',
             'objective':'binary',
             'metric': 'auc',
             'num_leaves': int(num_leaves),
             'min_data_in_leaf': int(min_data_in_leaf),
             'learning_rate': learning_rate,
             'max_depth': int(max_depth),
             'feature_fraction': feature_fraction,
             'bagging_freq': int(bagging_freq),
             'bagging_fraction': bagging_fraction,
             'use_missing': True,
             'nthread': 4,
             'lambda_l1': lambda_l1,
             'lambda_l2': lambda_l2,
             'max_bin': int(max_bin)
            }

    print (param)

    skf = StratifiedKFold(n_splits=skf_spl, shuffle=True, random_state=256)


    prediction_validation = []          #np.zeros(len(predictors)*skf_spl)
    true_validation=[]          #np.zeros(len(predictors)*skf_spl)

    prediction_hidden=[]          #np.zeros(len(predictors))
    true_hidden=[]          #np.zeros(len(predictors))
    
    
    oof_hidden = np.zeros(len(predictors))
    oof_val = np.zeros(len(predictors))
    
    for fold_cv_, (cv_idx, hid_idx) in enumerate(skf.split(predictors, labels)):
        strLog = "fold_cv_ {}".format(fold_cv_)
        #print(strLog)

        df_tr = predictors.iloc[cv_idx]
        target= pd.DataFrame(labels).iloc[cv_idx]


        x_hidden = predictors.iloc[hid_idx]
        hidden_test = pd.DataFrame(labels).iloc[hid_idx]

        folds = StratifiedKFold(n_splits=folds_spl, shuffle=True, random_state=256)

        hid_pred = []
        oof_val_i = np.zeros(len(df_tr))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr,target.values)):
            strLog = "fold {}".format(fold_)


            trn_data = lgb.Dataset(df_tr.iloc[trn_idx], label=target.iloc[trn_idx])
            val_data = lgb.Dataset(df_tr.iloc[val_idx], label=target.iloc[val_idx],reference=trn_data)

            num_round = 20000
            clf = lgb.train(param,trn_data,num_round,valid_sets=val_data,early_stopping_rounds=300,verbose_eval=False
                            #categorical_feature=categorical_index
                           )


            prediction_hidden.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            true_hidden.append(hidden_test[0].values)

            hidauc = roc_auc_score(pd.DataFrame(hidden_test), clf.predict(x_hidden, num_iteration=clf.best_iteration))
            hid_pred.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            #print(strLog)
            #print (str(fold_cv_) + str(fold_) + ' hidden set auc is '+str(hidauc))
            


            prediction_validation.append( clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            true_validation.append( target.iloc[val_idx][0].values)


            a=roc_auc_score(target.iloc[val_idx],clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            #print (str(fold_cv_) + str(fold_) + ' valid auc is ' + str(a))
            oof_val_i[val_idx] = clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration)
        oof_hidden[hid_idx] = np.mean(hid_pred, axis = 0)
        oof_val[cv_idx] += oof_val_i

    oof_val = oof_val / skf_spl
################################
    strAUC = roc_auc_score(labels, oof_hidden)
    print('strAUC hidden is ' + str(strAUC) )

    strAUC_val = roc_auc_score(labels, oof_val)
    print('strAUC val  is ' + str(strAUC_val) )
################################


    print ('done')
    fpr, tpr, thresholds = roc_curve(labels, oof_val)
    
    auc_ss = np.full(fpr.shape, strAUC_val)
    
    gmeans = gmean([tpr ,  auc_ss])
    

    
    
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    
    
    return gmeans[ix] #strAUC_val # 




gp_params = {"alpha": 1e-4}




xgb_bo = BayesianOptimization(xgb_evaluate, {'num_leaves': (5, 30), 
                                             'min_data_in_leaf': (1, 5.),
                                             'learning_rate':(0.005, 0.1),
                                             'max_depth': (5, 200),
                                             'feature_fraction': (0.3, 1.),
                                             'bagging_freq': (1, 4.),
                                             'bagging_fraction':(0.3, 1.),
                                             'lambda_l1': (0, 1),
                                             'lambda_l2': (0, 1),
                                             'max_bin': (10,300)
                                             
                                            })
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=75, n_iter=25, acq='ei',  **gp_params)


# In[97]:


dataset_new_with_ILB['PD_ILB_controlx'].value_counts()


# In[98]:


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr - (1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    #print (roc)
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    #print (roc_t)
    return list(roc_t['threshold']) 



def auc_conf_int(n1, n2, AUC, z=1.959964):
    q0 = AUC*(1-AUC)
    q1 = (AUC/(2-AUC))-AUC**2
    q2 = (2*(AUC**2)/(1+AUC))-AUC**2
    se = math.sqrt((q0+(n1-1)*q1+(n2-1)*q2)/(n1*n2))
    return (AUC-se*z, AUC+se*z)




def modeling(predictors, labels,  param, predictors_columns):
    
    skf_spl = 5
    folds_spl = 5



    skf = StratifiedKFold(n_splits=skf_spl, shuffle=True, random_state=256)


    prediction_validation = []          #np.zeros(len(predictors)*skf_spl)
    true_validation=[]          #np.zeros(len(predictors)*skf_spl)

    prediction_hidden=[]          #np.zeros(len(predictors))
    true_hidden=[]          #np.zeros(len(predictors))
    sens_hid=[]
    feat = []


    oof_hidden = np.zeros(len(predictors))
    oof_val = np.zeros(len(predictors))


    for fold_cv_, (cv_idx, hid_idx) in enumerate(skf.split(predictors, labels)):
        strLog = "fold_cv_ {}".format(fold_cv_)
        print(strLog)

        df_tr = predictors.iloc[cv_idx]
        target= pd.DataFrame(labels).iloc[cv_idx]
        df_tr_autop_lwy = autop_lwy.iloc[cv_idx]


        x_hidden = predictors.iloc[hid_idx]
        hidden_test = pd.DataFrame(labels).iloc[hid_idx]
        x_hidden_autop_lwy = autop_lwy.iloc[hid_idx]

        folds = StratifiedKFold(n_splits=folds_spl, shuffle=True, random_state=256)

        hid_pred = []

        oof_val_i = np.zeros(len(df_tr))


        for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr,target.values)):
            strLog = "fold {}".format(fold_)


            trn_data = lgb.Dataset(df_tr.iloc[trn_idx], label=target.iloc[trn_idx])
            val_data = lgb.Dataset(df_tr.iloc[val_idx], label=target.iloc[val_idx],reference=trn_data)

            num_round = 20000
            clf = lgb.train(param,trn_data,num_round,valid_sets=val_data,early_stopping_rounds=300,verbose_eval=200
                            #categorical_feature=categorical_index
                           )


            hid_pred.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            # prediction_hidden.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            # true_hidden.append(hidden_test[0].values)
            sens_hid.append(x_hidden_autop_lwy)
            feat.append(clf.feature_importance(importance_type='gain'))

            hidauc = roc_auc_score(pd.DataFrame(hidden_test), clf.predict(x_hidden, num_iteration=clf.best_iteration))

            print(strLog)
            print (str(fold_cv_) + str(fold_) + ' hidden set auc is '+str(hidauc))


            oof_val_i[val_idx] = clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration)

            prediction_validation.append( clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            true_validation.append( target.iloc[val_idx][0].values)


            a=roc_auc_score(target.iloc[val_idx],clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            print (str(fold_cv_) + str(fold_) + ' valid auc is ' + str(a))

        oof_hidden[hid_idx] = np.mean(hid_pred, axis = 0)
        oof_val[cv_idx] += oof_val_i

    oof_val = oof_val / skf_spl

    strAUC = roc_auc_score(labels, oof_hidden)
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    ci1, ci2 = auc_conf_int(n1, n2, strAUC)
    print('strAUC hidden is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    

    strAUC = roc_auc_score(labels, oof_val)
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    ci1, ci2 = auc_conf_int(n1, n2, strAUC)    
    
    print('strAUC val is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    # Add prediction probability to dataframe

    # Find optimal probability threshold
    threshold = Find_Optimal_Cutoff(labels, oof_val)
    print (threshold)
    thr =threshold[0]
    print ('Threshold  is ' + str(thr))
    
    binary_hidden = np.array([1 if i > thr else 0 for i in oof_hidden])
    print ('Hidden accuracy_score : ', accuracy_score(labels, binary_hidden)) 
    print (classification_report(labels, binary_hidden))

    CM = confusion_matrix(labels, binary_hidden)
    print ('CM : ',  CM)
    TN, FP, FN, TP = (confusion_matrix(labels, binary_hidden).ravel())
    
    
    print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )
    feature_importance = np.mean(feat, axis=0)
    feature_importance


    # Example data
    feature_names_arg = tuple(np.asarray(predictors_columns)[np.argsort(-feature_importance)])

    y_pos = np.arange(len(feature_names_arg))
    performance = -np.sort(-feature_importance)
    
    print ('here, Threshold  is ' + str(thr))
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    
    ci1, ci2 = auc_conf_int(n1, n2, ACC)    
    
    print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    ci1, ci2 = auc_conf_int(n1, n2, TPR)    
    
    print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    ci1, ci2 = auc_conf_int(n1, n2, TNR)    
    
    print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    f1 = f1_score(labels, binary_hidden)
    
    ci1, ci2 = auc_conf_int(n1, n2, f1)  
    print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    ci1, ci2 = auc_conf_int(n1, n2, PPV)    
    
    print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')

    return oof_hidden, oof_val, feature_names_arg, y_pos, performance, thr

#######################################################################################################

param = {'task': 'train', 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'num_leaves': 25, 'min_data_in_leaf': 4, 'learning_rate': 0.09848250957744414, 'max_depth': 39, 'feature_fraction': 0.4536801607066062, 'bagging_freq': 1, 'bagging_fraction': 0.8279271246615183, 'use_missing': True, 'nthread': 4, 'lambda_l1': 0.6339386213183115, 'lambda_l2': 0.03643748852246498, 'max_bin': 122}







oof_hidden, oof_val, feature_names_arg, y_pos, performance, thr = modeling(predictors, labels,  param, predictors_columns)

#######################################################################################################


# In[99]:


Predicted_Probs_3_year = pd.Series(oof_hidden, index =dataset_new_with_ILB.index )
Predicted_Probs_3_year


# In[100]:


hidden_prediction_pd = pd.DataFrame({'HHP':dataset_new_with_ILB[ 'HHP' ],
                                     'Predicted_Probs_3_year': Predicted_Probs_3_year,
                                     'PD_ILB_control': dataset_new_with_ILB[ 'PD_ILB_control' ],
                                     'PD_ILB_controlx': dataset_new_with_ILB[ 'PD_ILB_controlx' ],
                                     
                                     
                                     
                                    'DENMD1':dataset_new_with_ILB[ 'DENMD1' ] ,
                                     'DENMV2':dataset_new_with_ILB[ 'DENMV2' ] ,
                                     'DENLD3':dataset_new_with_ILB[ 'DENLD3' ] ,
                                     'DENLV4':dataset_new_with_ILB[ 'DENLV4' ] ,
                                     'DENTOT':dataset_new_with_ILB[ 'DENTOT' ] ,
                                     
                                     
                                     
                                     
                                     'EKG_to_Censor': dataset_new_with_ILB['EKG_to_Censor'],
                                   'Incidence_years_PD': dataset_new_with_ILB['Incidence_years_PD'],'Ex5_year' : dataset_new_with_ILB[ 'Ex5_year'], 
                                   'CensorYear': dataset_new_with_ILB[ 'CensorYear' ], 'YRDX':dataset_new_with_ILB[ 'YRDX' ],
                                     'ECGX4YY': dataset_new_with_ILB['ECGX4YY'],
                                     'Autopsy':dataset_new_with_ILB['Autopsy'],
                                     'LEWYBDY': dataset_new_with_ILB['LEWYBDY']
                                   }).set_index(dataset_new_with_ILB.index)


hidden_prediction_pd


# In[101]:


(hidden_prediction_pd['CensorYear']  - hidden_prediction_pd['Ex5_year']).value_counts()


# In[102]:


hidden_prediction_pd['PD_ILB_control'].value_counts()


# In[103]:


hidden_prediction_pd['PD_ILB_controlx'].value_counts()


# In[104]:


LB_Unknownbig = hidden_prediction_pd[dataset_new_with_ILB['PD_ILB_control']== 0]
LB_No = LB_Unknownbig[(LB_Unknownbig['Autopsy'] == 1)]
LB_Unknown = LB_Unknownbig.drop(LB_No.index)
LB_Yes_cases = hidden_prediction_pd[(hidden_prediction_pd['LEWYBDY']==1)&(hidden_prediction_pd['PD_ILB_control']==1) &(hidden_prediction_pd['PD_ILB_controlx']==2)]
LB_Yes_control = hidden_prediction_pd[(hidden_prediction_pd['LEWYBDY']==1)&(hidden_prediction_pd['PD_ILB_control']==1) &(hidden_prediction_pd['PD_ILB_controlx']==0)]

Cases_1 = hidden_prediction_pd[hidden_prediction_pd['PD_ILB_control']==2]


# In[105]:


LB_Unknownbig.shape, LB_Unknown.shape, LB_No.shape, LB_Yes_cases.shape, LB_Yes_control.shape, Cases_1.shape


# In[106]:



LB_Unknown_pd = pd.DataFrame({'HHP_c':LB_Unknown[ 'HHP' ],
                                    'label':['LB_Unknown' for i in range(LB_Unknown.shape[0])]
                                    }).set_index(LB_Unknown.index)

LB_No_pd = pd.DataFrame({'HHP_c':LB_No[ 'HHP' ],
                                    'label':['LB_No' for i in range(LB_No.shape[0])]
                                    }).set_index(LB_No.index)
LB_Yes_cases_pd = pd.DataFrame({'HHP_c':LB_Yes_cases[ 'HHP' ],
                                    'label':['LB_Yes_cases' for i in range(LB_Yes_cases.shape[0])]
                                    }).set_index(LB_Yes_cases.index)


LB_Yes_control_pd = pd.DataFrame({'HHP_c':LB_Yes_control[ 'HHP' ],
                                    'label':['LB_Yes_control' for i in range(LB_Yes_control.shape[0])]
                                    }).set_index(LB_Yes_control.index)


Cases_1_pd = pd.DataFrame({'HHP_c':Cases_1[ 'HHP' ],
                                    'label':['Cases_2' for i in range(Cases_1.shape[0])]
                                    }).set_index(Cases_1.index)


# In[107]:


labelize = pd.concat([LB_Unknown_pd, LB_No_pd, LB_Yes_cases_pd, LB_Yes_control_pd, Cases_1_pd]).sort_index()
labelize


# In[108]:


hidden_prediction_pd['labelize'] = labelize['label']
hidden_prediction_pd


# In[109]:


hidden_prediction_pd.to_excel("Model_5_5_year_hidden_prediction.xlsx")


# In[110]:


STAT_LB_No = hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') |  (hidden_prediction_pd['labelize'] == 'LB_No') ]
STAT_LB_No


# In[111]:


labels_new = list((STAT_LB_No['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_No['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_No is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[112]:


STAT_LB_Yes = hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') |  (hidden_prediction_pd['labelize'] == 'LB_Yes_control') ]
STAT_LB_Yes


# In[113]:


labels_new = list((STAT_LB_Yes['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_Yes['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_Yes is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[114]:


STAT_LB_Unknown = hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') |  (hidden_prediction_pd['labelize'] == 'LB_Unknown') ]
STAT_LB_Unknown


# In[115]:


labels_new = list((STAT_LB_Unknown['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_Unknown['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_Unknown is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[116]:


STAT_NLB = hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') |  (hidden_prediction_pd['labelize'] == 'LB_Unknown')  |  (hidden_prediction_pd['labelize'] == 'LB_No') ]
STAT_NLB


# In[117]:


labels_new = list((STAT_NLB['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_NLB['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_NLB is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[118]:


labels_new = list((hidden_prediction_pd['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = hidden_prediction_pd['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for hidden_prediction_pd is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[119]:


hidden_prediction_pd['PD_ILB_control'].value_counts()


# In[120]:


hidden_prediction_pd['PD_ILB_control'].replace(1,0).value_counts()


# In[121]:


labels_new = list(((hidden_prediction_pd['PD_ILB_control'].replace(1,0))/2).astype(int))  # pd =1, controls=0
oof_hidden_new = hidden_prediction_pd['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for control hidden_prediction_pd is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[122]:


labels_new = list(((hidden_prediction_pd['PD_ILB_control'].replace(1,2))/2).astype(int))  # pd =1, controls=0
oof_hidden_new = hidden_prediction_pd['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for control hidden_prediction_pd is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[123]:



predictors_columns_type = ['categorical', 
                          'continuous',
                          'categorical', 
                          'continuous',
                          'continuous',
                          'categorical', 
                          'continuous',
                         'categorical', 
                          'categorical', 
                          'continuous',
                          'categorical',
                          'continuous']
#######################################################################################################
predictorsx = predictors.copy()


prod_all = []

cols_sim = []


for i in range(len(predictors_columns)):
    
    print ("Simulation for " +predictors_columns[i]+ ' whose data type is ' +  predictors_columns_type[i])
    
    interested = predictors_columns[i]
    
    np_interested_col = np.array(predictorsx[interested])
    #np_interested_col = np_interested_col[~numpy.isnan(np_interested_col)]
    
    simulated_col = np.zeros(predictorsx.shape[0])
    
    prod =[]
    
    np_interested_colx = np_interested_col[~np.isnan(np_interested_col)]
    
    if predictors_columns_type[i]=="categorical":
        
        
        
        
        val = np.max(np_interested_colx)
        
        
        val2 = sorted(set(np_interested_colx))[0]
        
        for i in range(predictorsx.shape[0]):
            
            
            if np_interested_col[i]==val:
                simulated_col[i] = val2
                prod.append(-1)
                
            elif str(np_interested_col[i])=='nan':
                
                simulated_col[i] = np.nan
                prod.append(0)          
                
            else:
                ind = sorted(set(np_interested_colx)).index(np_interested_col[i])
                simulated_col[i] = sorted(set(np_interested_colx))[ind + 1]
                prod.append(1)
        
        #predictorsx[interested] = simulated_col
        cols_sim.append(simulated_col)
        
    elif predictors_columns_type[i]=='continuous':
        
        val = np.std(np_interested_colx)
        
        for i in range(predictorsx.shape[0]):

            if np_interested_col[i]==np.nan:
                simulated_col[i] = np.nan
                prod.append(0) 
            
            else:
                simulated_col[i] = np_interested_col[i] + val
                prod.append(1)                 

        cols_sim.append(simulated_col)
        
    prod_all.append(prod)
    
#######################################################################################################

def modeling_new(predictors, labels,  param, predictors_columns, interested_id):
    
    skf_spl = 5
    folds_spl = 5
    predictors_new = predictors.copy()
    
    predictors_new.iloc[:,interested_id] = cols_sim[interested_id]
    pr = np.array(prod_all[interested_id])

    skf = StratifiedKFold(n_splits=skf_spl, shuffle=True, random_state=256)


    prediction_validation = []          #np.zeros(len(predictors)*skf_spl)
    true_validation=[]          #np.zeros(len(predictors)*skf_spl)

    prediction_hidden=[]          #np.zeros(len(predictors))
    true_hidden=[]          #np.zeros(len(predictors))
    sens_hid=[]
    feat = []


    oof_hidden = np.zeros(len(predictors))
    oof_val = np.zeros(len(predictors))
    
    
    oof_hidden_new = np.zeros(len(predictors))
    pr_sim_hid = np.zeros(len(predictors))

    for fold_cv_, (cv_idx, hid_idx) in enumerate(skf.split(predictors, labels)):
        strLog = "fold_cv_ {}".format(fold_cv_)
        print(strLog)

        df_tr = predictors.iloc[cv_idx]
        df_tr_new = predictors_new.iloc[cv_idx]
        target= pd.DataFrame(labels).iloc[cv_idx]
        df_tr_autop_lwy = autop_lwy.iloc[cv_idx]


        x_hidden = predictors.iloc[hid_idx]
        x_hidden_new = predictors_new.iloc[hid_idx]
        pr_sim  = pr[hid_idx]
        
        hidden_test = pd.DataFrame(labels).iloc[hid_idx]
        x_hidden_autop_lwy = autop_lwy.iloc[hid_idx]
        
        
        

        folds = StratifiedKFold(n_splits=folds_spl, shuffle=True, random_state=256)

        hid_pred = []
        hid_pred_new = []

        oof_val_i = np.zeros(len(df_tr))


        for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr,target.values)):
            strLog = "fold {}".format(fold_)


            trn_data = lgb.Dataset(df_tr.iloc[trn_idx], label=target.iloc[trn_idx])
            val_data = lgb.Dataset(df_tr.iloc[val_idx], label=target.iloc[val_idx],reference=trn_data)

            num_round = 20000
            clf = lgb.train(param,trn_data,num_round,valid_sets=val_data,early_stopping_rounds=300,verbose_eval=200
                            #categorical_feature=categorical_index
                           )


            hid_pred.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            hid_pred_new.append(clf.predict(x_hidden_new, num_iteration=clf.best_iteration))
            # prediction_hidden.append(clf.predict(x_hidden, num_iteration=clf.best_iteration))
            # true_hidden.append(hidden_test[0].values)
            sens_hid.append(x_hidden_autop_lwy)
            feat.append(clf.feature_importance(importance_type='gain'))

            hidauc = roc_auc_score(pd.DataFrame(hidden_test), clf.predict(x_hidden, num_iteration=clf.best_iteration))

            print(strLog)
            print (str(fold_cv_) + str(fold_) + ' hidden set auc is '+str(hidauc))


            oof_val_i[val_idx] = clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration)

            prediction_validation.append( clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            true_validation.append( target.iloc[val_idx][0].values)


            a=roc_auc_score(target.iloc[val_idx],clf.predict(df_tr.iloc[val_idx], num_iteration=clf.best_iteration))
            print (str(fold_cv_) + str(fold_) + ' valid auc is ' + str(a))

        oof_hidden[hid_idx] = np.mean(hid_pred, axis = 0)
        oof_val[cv_idx] += oof_val_i
        
        oof_hidden_new[hid_idx] = np.mean(hid_pred_new, axis = 0)
        
        pr_sim_hid[hid_idx] = pr_sim
    oof_val = oof_val / skf_spl
    
    
    
    strAUC = roc_auc_score(labels, oof_hidden_new)
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    ci1, ci2 = auc_conf_int(n1, n2, strAUC)
    print('strAUC hidden simulated is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    
    
    
    

    strAUC = roc_auc_score(labels, oof_hidden)
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    ci1, ci2 = auc_conf_int(n1, n2, strAUC)
    print('strAUC hidden is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    

    strAUC = roc_auc_score(labels, oof_val)
    
    n1=np.sum(labels)
    n2=len(labels)-np.sum(labels)
    ci1, ci2 = auc_conf_int(n1, n2, strAUC)    
    
    print('strAUC val is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')
    # Add prediction probability to dataframe

    # Find optimal probability threshold
    threshold = Find_Optimal_Cutoff(labels, oof_hidden)
    print (threshold)
    thr =threshold[0]
    print ('Threshold  is ' + str(thr))
    
    binary_hidden = np.array([1 if i > thr else 0 for i in oof_hidden])
    print ('Hidden accuracy_score : ', accuracy_score(labels, binary_hidden)) 
    print (classification_report(labels, binary_hidden))

    CM = confusion_matrix(labels, binary_hidden)
    print ('CM : ',  CM)
    TN, FP, FN, TP = (confusion_matrix(labels, binary_hidden).ravel())
    print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )
    feature_importance = np.mean(feat, axis=0)
    feature_importance


    # Example data
    feature_names_arg = tuple(np.asarray(predictors_columns)[np.argsort(-feature_importance)])

    y_pos = np.arange(len(feature_names_arg))
    performance = -np.sort(-feature_importance)

    return oof_hidden, oof_val, feature_names_arg, y_pos, performance, thr, oof_hidden_new, pr_sim_hid

#######################################################################################################


base=[]
sim=[]
Sim_conf_inter = []




for i in range(len(predictors_columns)):
    
    print ("Simulation for " +predictors_columns[i]+ ' whose data type is ' +  predictors_columns_type[i])
    

    Xoof_hidden, Xoof_val, Xfeature_names_arg, Xy_pos, Xperformance, Xthr, Sim_oof_hidden, pr_x = modeling_new(predictors,
                                                                                                       labels,
                                                                                                       param, predictors_columns,
                                                                                                            i)

    
    
    difference = (Xoof_hidden - Sim_oof_hidden) * pr_x
    
    base.append(Xoof_hidden)
    sim.append(Sim_oof_hidden)
    
    
    print (len(difference))
    
    sci1, sci2 = sms.DescrStatsW(difference).tconfint_mean()
    
    print (np.mean(difference) , sci1, sci2)
    
    Sim_conf_inter.append((np.mean(difference) , sci1, sci2))


# In[124]:


for i in range(len(Sim_conf_inter)):
    print (Sim_conf_inter[i], predictors_columns_type[i],  predictors_columns[i])


# In[125]:


direction_color= [
'red' ,# (-0.007199662936575551, -0.008884261478115222, -0.005515064395035882) categorical Sleepy_complete
'red' ,# (-0.021445234344347096, -0.029584665487497552, -0.01330580320119664) continuous Simple_reaction_time_Ex5
'green' ,# (0.029462686611589767, 0.022843409353934366, 0.03608196386924517) categorical new_BM_frequency
'grey' ,# (-0.031562256489936484, -0.04232812058041369, -0.020796392399459275) continuous Choice_reaction_time_Ex5
'red' ,# (-0.020945717651911687, -0.027427839131334265, -0.014463596172489122) continuous CASIX4
'green' ,# (0.006985169470583623, 0.002073276692284574, 0.011897062248882671) categorical Olfaction_complete_Ex5_Ex4
'green' ,# (0.05627371706364421, 0.045782517507008805, 0.06676491662027963) continuous ECGage
'green' ,# (0.002124560485875781, -0.001324402625490908, 0.00557352359724247) categorical new_smoke
'red' ,# (0.0028116968551180964, 0.001358223773337043, 0.0042651699368991495) categorical TBI_LOC
'red' ,# (-0.013018761676287837, -0.01975631800091666, -0.0062812053516590055) continuous BMIX4
'green' ,# (-0.003413915194304468, -0.0052137837894731184, -0.0016140465991358172) categorical HYPBP1
'green' # (0.0364379542719352, 0.029134184701835555, 0.04374172384203484) continuous Coffee_oz_Ex1
]


# In[126]:


feature_names_arg


# In[127]:


feature_names_arg = (
 'CRT',
 'SRT',
 'BMI',
 'OLF',
 'BMF',
 'CASI',
 'AGE',
 'COF',
 'DTS',
 'SMK',                       
 'TBI',
 'HYP'
 )


# In[129]:


predictors_names = [
 'DTS',
 'SRT',
 'BMF',
 'CRT',
 'CASI',
 'OLF',
 'AGE',
 'SMK',
 'TBI',
 'BMI',
 'HYP',
 'COF']


# In[130]:


myorder = [3,1,9,5,2,4,6,11,0,7,8,10 ]
new_predictors_names = [predictors_names[i] for i in myorder]
new_predictors_names



# In[131]:


direction_color_arg = [direction_color[i] for i in myorder]
direction_color_arg 


# In[132]:




oof_hidden, oof_val, feature_names_arg, y_pos, performance, thr = modeling(predictors, labels,  param, predictors_columns)

feature_names_arg = (
 'CRT',
 'SRT',
 'BMI',
 'OLF',
 'BMF',
 'CASI',
 'AGE',
 'COF',
 'DTS',
 'SMK',                       
 'TBI',
 'HYP'
 )


# In[133]:


dpi_val=600

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data

ax.barh(y_pos, performance, align='center', color = direction_color_arg )
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names_arg)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain Importance')
#ax.set_title('Variable Importance Analysis')
plt.yticks(fontsize=12, weight = 'bold')
plt.savefig("VIA_model5d.png", dpi=dpi_val, bbox_inches='tight')
plt.show()

W_list = (np.array(Sim_conf_inter)[:,0] - np.array(Sim_conf_inter)[:,1])

plt.figure(figsize=(12,6))
plt.errorbar(x=np.array(Sim_conf_inter)[:,0], y=range(12), xerr=W_list, fmt='o', ecolor=direction_color, lw=3)

y = np.linspace(0, 12,100)
x = y*0
plt.plot(x,y, ':b',color='grey', label='Threshold Value')

plt.yticks(y_pos, predictors_names, rotation='horizontal')
plt.yticks(fontsize=12, weight = 'bold')



plt.savefig("model5d_var_direct.png", dpi=dpi_val, bbox_inches='tight')

plt.show()

binary_hidden = np.array([1 if i > thr else 0 for i in oof_hidden])
print ('Hidden accuracy_score : ', accuracy_score(labels, binary_hidden)) 
print (classification_report(labels, binary_hidden))

CM = confusion_matrix(labels, binary_hidden)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels, binary_hidden).ravel())
print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy|
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


ax= plt.subplot()


sns.heatmap(CM, annot=True, ax = ax,  fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'PD']); ax.yaxis.set_ticklabels(['Control', 'PD']);
plt.savefig("CM_model5d.png", dpi=dpi_val, bbox_inches='tight')


# In[134]:


new_predictors_names = [predictors_names[i] for i in myorder]
new_predictors_names


# In[135]:



new_direction_color = [direction_color[i] for i in myorder]
new_direction_color


# In[136]:


W_list = (np.array(Sim_conf_inter)[:,0] - np.array(Sim_conf_inter)[:,1])
new_W_list = [W_list[i] for i in myorder]
new_W_list


# In[137]:


mean_of = np.array(Sim_conf_inter)[:,0]
new_mean_of = [mean_of[i] for i in myorder]
new_mean_of


# In[138]:


plt.figure(figsize=(12,6))
plt.errorbar(x=new_mean_of[::-1], y=range(12), xerr=new_W_list[::-1], fmt='o', ecolor=new_direction_color[::-1], lw=3)

y = np.linspace(0, 12,100)
x = y*0
plt.plot(x,y, ':b',color='grey', label='Threshold Value')

plt.yticks(y_pos, new_predictors_names[::-1], rotation='horizontal')
plt.yticks(fontsize=12, weight = 'bold')



plt.savefig("new_model5d_var_direct.png", dpi=dpi_val, bbox_inches='tight')

plt.show()


# In[190]:


thr = 0.26237751134639836


# In[191]:



fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['DENLV4']
           , color='g', label='LB_No')

ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['DENLV4']
           , color='r', label='LB_Yes')


ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['DENLV4']
           , color='black', label='Cases-2')

#######################################################################################################



ax.set_xlabel('Prediction Value')
ax.set_ylabel('DENLV4')
#ax.set_title('scatter plot')
x = np.linspace(0, 42,100)
y = thr +x*0
plt.plot(y,x, ':b',color='grey', label='Threshold Value')
plt.legend(numpoints=1, loc=1)
plt.savefig('xDENLV4_pred_model5d.png', dpi=dpi_val, bbox_inches='tight')  # saves the current figure
plt.show()


# In[192]:



fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['DENLV4']
           , color='g', label='LB_No')

#ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['Predicted_Probs_3_year'] ,
#           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['DENLV4']
#           , color='r', label='LB_Yes')


ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['DENLV4']
           , color='black', label='Cases-2')

#######################################################################################################



ax.set_xlabel('Prediction Value')
ax.set_ylabel('DENLV4')
#ax.set_title('scatter plot')
x = np.linspace(0, 42,100)
y = thr +x*0
plt.plot(y,x, ':b',color='grey', label='Threshold Value')
plt.legend(numpoints=1, loc=1)
plt.savefig('xDENLV4_pred_model5d_V2.png', dpi=dpi_val, bbox_inches='tight')  # saves the current figure
plt.show()


# In[142]:


DENLV4_pred  = hidden_prediction_pd[pd.notna(hidden_prediction_pd['DENLV4'])]
DENLV4_pred = DENLV4_pred[(DENLV4_pred['labelize']=='LB_No') | (DENLV4_pred['labelize']=='Cases_2')]
DENLV4_pred


# In[143]:


r = np.corrcoef(DENLV4_pred['DENLV4'], DENLV4_pred['Predicted_Probs_3_year'])
r


# In[144]:


thr


# In[145]:


hidden_prediction_pd['labelize'].value_counts()


# In[146]:


thr = 0.21

labels_new = list((STAT_LB_No['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_No['Predicted_Probs_3_year']
binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_No is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[147]:


thr = 0.16

labels_new = list((STAT_LB_No['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_No['Predicted_Probs_3_year']
binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_No is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[148]:


thr = 0.26

labels_new = list((STAT_LB_No['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_No['Predicted_Probs_3_year']
binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_No is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[149]:


labels_new = list((STAT_LB_Unknown['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_Unknown['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_Unknown is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

thr = 0.26



binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])

print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[150]:


STAT_LB_Unknown['labelize'].value_counts()


# In[151]:


labels_new = list((STAT_LB_Unknown['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_Unknown['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_Unknown is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

thr = 0.504



binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])

print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[152]:


labels_new = list((STAT_LB_Unknown['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_LB_Unknown['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_Unknown is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

thr = 0.19



binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])

print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[153]:


labels_new = list((STAT_NLB['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_NLB['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_NLB is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



thr = 0.23



binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])

print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[154]:


labels_new = list((STAT_NLB['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_NLB['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_NLB is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



thr = 0.42



binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])

print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[155]:


labels_new = list((STAT_NLB['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = STAT_NLB['Predicted_Probs_3_year']


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_NLB is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



thr = 0.16



binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])

print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[156]:


thr = 0.23
NC_LB_Yes_cont = hidden_prediction_pd[((hidden_prediction_pd['labelize'] == 'LB_Yes_cases')|(hidden_prediction_pd['labelize'] == 'LB_Yes_control')) & (hidden_prediction_pd['Predicted_Probs_3_year'] <= thr)  ]
NC_LB_Yes_case = hidden_prediction_pd[((hidden_prediction_pd['labelize'] == 'LB_Yes_cases')|(hidden_prediction_pd['labelize'] == 'LB_Yes_control')) & (hidden_prediction_pd['Predicted_Probs_3_year'] > thr)  ]
print(np.mean(NC_LB_Yes_cont['DENLV4']), np.std(NC_LB_Yes_cont['DENLV4']))
print(np.mean(NC_LB_Yes_case['DENLV4']), np.std(NC_LB_Yes_case['DENLV4']))

print(np.mean(NC_LB_Yes_cont['DENTOT']), np.std(NC_LB_Yes_cont['DENTOT']))
print(np.mean(NC_LB_Yes_case['DENTOT']), np.std(NC_LB_Yes_case['DENTOT']))


# In[157]:


hidden_prediction_pd['labelize'].value_counts()


# In[158]:


thr = 0.23

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['DENLV4']
           , color='g', label='LB_No')

ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['DENLV4']
           , color='r', label='LB_Yes')


ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['DENLV4']
           , color='black', label='Cases-2')

#######################################################################################################



ax.set_xlabel('Prediction Value')
ax.set_ylabel('DENLV4')
#ax.set_title('scatter plot')
x = np.linspace(0, 42,100)
y = thr +x*0
plt.plot(y,x, ':b',color='grey', label='Threshold Value')
plt.legend(numpoints=1, loc=1)
plt.savefig('xDENLV4_pred_model5d.png', dpi=dpi_val, bbox_inches='tight')  # saves the current figure
plt.show()



fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['DENLV4']
           , color='g', label='LB_No')

#ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['Predicted_Probs_3_year'] ,
#           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['DENLV4']
#           , color='r', label='LB_Yes')


ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['DENLV4']
           , color='black', label='Cases-2')

#######################################################################################################



ax.set_xlabel('Prediction Value')
ax.set_ylabel('DENLV4')
#ax.set_title('scatter plot')
x = np.linspace(0, 42,100)
y = thr +x*0
plt.plot(y,x, ':b',color='grey', label='Threshold Value')
plt.legend(numpoints=1, loc=1)
plt.savefig('xDENLV4_pred_model5d_V2.png', dpi=dpi_val, bbox_inches='tight')  # saves the current figure
plt.show()


# In[159]:


# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# generate 2 class dataset


oof_hidden_new_LB_No = STAT_LB_No['Predicted_Probs_3_year']
labels_new_LB_No = list((STAT_LB_No['PD_ILB_controlx']/2).astype(int))


Base_oof_hidden_new_LB_No = Base_STAT_LB_No['Predicted_Probs_3_year']
Base_labels_new_LB_No = list((Base_STAT_LB_No['PD_ILB_controlx']/2).astype(int))


oof_hidden_new = STAT_NLB['Predicted_Probs_3_year']
labels_new = list((STAT_NLB['PD_ILB_controlx']/2).astype(int))


Base_oof_hidden_new = Base_STAT_NLB['Predicted_Probs_3_year']
Base_labels_new = list((Base_STAT_NLB['PD_ILB_controlx']/2).astype(int))


# calculate scores
final_auc_no = roc_auc_score(labels_new_LB_No, oof_hidden_new_LB_No)
base_auc_no = roc_auc_score(Base_labels_new_LB_No, Base_oof_hidden_new_LB_No)

final_auc = roc_auc_score(labels_new, oof_hidden_new)
base_auc = roc_auc_score(Base_labels_new, Base_oof_hidden_new)

# summarize scores
print('Final Model: Case_2, LB_No: ROC AUC=%.3f' % (final_auc_no))
print('Baseline Model: Case_2, LB_No: ROC AUC=%.3f' % (base_auc_no))

print('Final Model: Case_2, LB_No, LB_Unknown: ROC AUC=%.3f' % (final_auc))
print('Baseline Model: Case_2, LB_No, LB_Unknown: ROC AUC=%.3f' % (base_auc))

# calculate roc curves
ns_fpr_no, ns_tpr_no, _ = roc_curve(labels_new_LB_No, oof_hidden_new_LB_No)
lr_fpr_no, lr_tpr_no, _ = roc_curve(Base_labels_new_LB_No, Base_oof_hidden_new_LB_No)

ns_fpr, ns_tpr, _ = roc_curve(labels_new, oof_hidden_new)
lr_fpr, lr_tpr, _ = roc_curve(Base_labels_new, Base_oof_hidden_new)


# plot the roc curve for the model
pyplot.plot(ns_fpr_no, ns_tpr_no, marker='.',  label='Final Model: Cases-2, LB_No (AUC=0.803) ')
pyplot.plot(lr_fpr_no, lr_tpr_no, linestyle='--',label='Baseline Model: Cases, LB_No (AUC=0.68) ')

pyplot.plot(ns_fpr, ns_tpr, marker='.', label='Final Model: Cases-2, LB_No, LB_Unknown (AUC=0.691) ')
pyplot.plot(lr_fpr, lr_tpr, linestyle='--', label='Baseline Model: Cases, LB_No, LB_Unknown (AUC=0.635) ')


# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend(fontsize=8)

plt.savefig('AUC_comparison_5_year.png', dpi=dpi_val, bbox_inches='tight')  # saves the current figure

# show the plot
pyplot.show()


# In[160]:


LB_Unknownbig.shape, LB_Unknown.shape, LB_No.shape, LB_Yes_cases.shape, LB_Yes_control.shape, Cases_1.shape


# In[161]:


thr = 0.23

LB_Unknown['binary_hidden'] = np.array([1 if i > thr else 0 for i in LB_Unknown['Predicted_Probs_3_year']])


# In[162]:


inter = LB_Unknown.copy()

fp, tn = inter[(inter['PD_ILB_control']== 0) & (inter['binary_hidden']== 1)].shape[0] , inter[(inter['PD_ILB_control']== 0) & (inter['binary_hidden']== 0)].shape[0]
fp, tn, fp/inter.shape[0], tn/inter.shape[0]


# In[163]:


LB_No['binary_hidden'] = np.array([1 if i > thr else 0 for i in LB_No['Predicted_Probs_3_year']])

inter = LB_No.copy()

fp, tn = inter[(inter['PD_ILB_control']== 0) & (inter['binary_hidden']== 1)].shape[0] , inter[(inter['PD_ILB_control']== 0) & (inter['binary_hidden']== 0)].shape[0]
fp, tn, fp/inter.shape[0], tn/inter.shape[0]


# In[164]:


Cases_1['binary_hidden'] = np.array([1 if i > thr else 0 for i in Cases_1['Predicted_Probs_3_year']])

inter = Cases_1.copy()

tp, fn = inter[(inter['PD_ILB_control']== 2) & (inter['binary_hidden']== 1)].shape[0] , inter[(inter['PD_ILB_control']== 2) & (inter['binary_hidden']== 0)].shape[0]
tp, fn, tp/inter.shape[0], fn/inter.shape[0]


# In[165]:


hidden_prediction_pd['Predicted_Probs_3_year']


# In[166]:



labels_new = list((hidden_prediction_pd['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = hidden_prediction_pd['Predicted_Probs_3_year']

thresh = Find_Optimal_Cutoff(labels_new, oof_hidden_new)
thr = thresh[0]
binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_No is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[167]:


hidden_prediction_pd['binary_hidden_new'] = binary_hidden_new


# In[168]:


hidden_prediction_pd


# In[169]:


LB_Unknownbig = hidden_prediction_pd[dataset_new_with_ILB['PD_ILB_control']== 0]
LB_No = LB_Unknownbig[(LB_Unknownbig['Autopsy'] == 1)]
LB_Unknown = LB_Unknownbig.drop(LB_No.index)
LB_Yes_cases = hidden_prediction_pd[(hidden_prediction_pd['LEWYBDY']==1)&(hidden_prediction_pd['PD_ILB_control']==1) &(hidden_prediction_pd['PD_ILB_controlx']==2)]
LB_Yes_control = hidden_prediction_pd[(hidden_prediction_pd['LEWYBDY']==1)&(hidden_prediction_pd['PD_ILB_control']==1) &(hidden_prediction_pd['PD_ILB_controlx']==0)]

Cases_1 = hidden_prediction_pd[hidden_prediction_pd['PD_ILB_control']==2]


# In[170]:


LB_Unknownbig.shape, LB_Unknown.shape, LB_No.shape, LB_Yes_cases.shape, LB_Yes_control.shape, Cases_1.shape


# In[171]:


thr


# In[172]:


inter = LB_Unknown.copy()

fp, tn = inter[(inter['PD_ILB_control']== 0) & (inter['binary_hidden_new']== 1)].shape[0] , inter[(inter['PD_ILB_control']== 0) & (inter['binary_hidden_new']== 0)].shape[0]
fp, tn, fp/inter.shape[0], tn/inter.shape[0]


# In[173]:



inter = LB_No.copy()

fp, tn = inter[(inter['PD_ILB_control']== 0) & (inter['binary_hidden_new']== 1)].shape[0] , inter[(inter['PD_ILB_control']== 0) & (inter['binary_hidden_new']== 0)].shape[0]
fp, tn, fp/inter.shape[0], tn/inter.shape[0]


# In[174]:



inter = LB_Yes_control.copy()

fp, tn = inter[(inter['PD_ILB_controlx']== 0) & (inter['binary_hidden_new']== 1)].shape[0] , inter[(inter['PD_ILB_controlx']== 0) & (inter['binary_hidden_new']== 0)].shape[0]
fp, tn, fp/inter.shape[0], tn/inter.shape[0]


# In[175]:


inter = Cases_1.copy()

tp, fn = inter[(inter['PD_ILB_control']== 2) & (inter['binary_hidden_new']== 1)].shape[0] , inter[(inter['PD_ILB_control']== 2) & (inter['binary_hidden_new']== 0)].shape[0]
tp, fn, tp/inter.shape[0], fn/inter.shape[0]


# In[176]:


inter = LB_Yes_cases.copy()

tp, fn = inter[(inter['PD_ILB_controlx']== 2) & (inter['binary_hidden_new']== 1)].shape[0] , inter[(inter['PD_ILB_controlx']== 2) & (inter['binary_hidden_new']== 0)].shape[0]
tp, fn, tp/inter.shape[0], fn/inter.shape[0]


# In[179]:


thr = 0.4

labels_new = list((hidden_prediction_pd['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = hidden_prediction_pd['Predicted_Probs_3_year']
binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_No is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[187]:


thr = 0.173

labels_new = list((hidden_prediction_pd['PD_ILB_controlx']/2).astype(int))  # pd =1, controls=0
oof_hidden_new = hidden_prediction_pd['Predicted_Probs_3_year']
binary_hidden_new = np.array([1 if i > thr else 0 for i in oof_hidden_new])


strAUC = roc_auc_score(labels_new, oof_hidden_new)

n1=np.sum(labels_new)
n2=len(labels_new)-np.sum(labels_new)
ci1, ci2 = auc_conf_int(n1, n2, strAUC)
print('strAUC for STAT_LB_No is ' + str(strAUC) + "[" + str(ci1) + ', '+ str(ci2) + ']')



print ('Hidden accuracy_score : ', accuracy_score(labels_new, binary_hidden_new)) 
print (classification_report(labels_new, binary_hidden_new))

CM = confusion_matrix(labels_new, binary_hidden_new)
print ('CM : ',  CM)
TN, FP, FN, TP = (confusion_matrix(labels_new, binary_hidden_new).ravel())


print ('TN, FP, FN, TP : ', TN, FP, FN, TP)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ( 'TPR, TNR, f1_score: ', TPR, TNR, f1_score(labels, binary_hidden) )


print ('here, Threshold  is ' + str(thr))


ci1, ci2 = auc_conf_int(n1, n2, ACC)    

print('ACC val is ' + str(ACC) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TPR)    

print('TPR (Sensitivity) val is ' + str(TPR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, TNR)    

print('TNR (Specificity)  val is ' + str(TNR) + "[" + str(ci1) + ', '+ str(ci2) + ']')

f1 = f1_score(labels, binary_hidden)

ci1, ci2 = auc_conf_int(n1, n2, f1)  
print('F1 score is ' + str(f1) + "[" + str(ci1) + ', '+ str(ci2) + ']')

ci1, ci2 = auc_conf_int(n1, n2, PPV)    

print('PPV val is ' + str(PPV) + "[" + str(ci1) + ', '+ str(ci2) + ']')


# In[188]:


hidden_prediction_pd.to_excel("Model_5_5_year_hidden_predictionv2.xlsx")


# In[197]:


thr = 0.26237751134639836

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['DENLV4']
           , color='g', label='LB-No')


ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') ]['DENLV4']
           , color='grey', label='LB-Yes-Case')

ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_control') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_control') ]['DENLV4']
           , color='lime', label='LB-Yes-Control')



ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['DENLV4']
           , color='black', label='Case-2')

#######################################################################################################



ax.set_xlabel('Prediction Value')
ax.set_ylabel('DENLV4')
#ax.set_title('scatter plot')
x = np.linspace(0, 42,100)
y = thr +x*0
plt.plot(y,x, ':b',color='grey', label='Cutoff')
plt.legend(numpoints=1, loc=1)
plt.savefig('model4_xDENLV4_pred_model5d.png', dpi=dpi_val, bbox_inches='tight')  # saves the current figure
plt.show()



fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_No') ]['DENLV4']
           , color='g', label='LB-No')

#ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['Predicted_Probs_3_year'] ,
#           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'LB_Yes_cases') | (hidden_prediction_pd['labelize'] == 'LB_Yes_control')]['DENLV4']
#           , color='r', label='LB_Yes')


ax.scatter(hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['Predicted_Probs_3_year'] ,
           hidden_prediction_pd[(hidden_prediction_pd['labelize'] == 'Cases_2') ]['DENLV4']
           , color='black', label='Cases-2')

#######################################################################################################



ax.set_xlabel('Prediction Value')
ax.set_ylabel('DENLV4')
#ax.set_title('scatter plot')
x = np.linspace(0, 42,100)
y = thr +x*0
plt.plot(y,x, ':b',color='grey', label='Cutoff')
plt.legend(numpoints=1, loc=1)
plt.savefig('model4_xDENLV4_pred_model5d_V2.png', dpi=dpi_val, bbox_inches='tight')  # saves the current figure
plt.show()


# In[199]:


oof_hidden, oof_val, feature_names_arg, y_pos, performance, thr = modeling(predictors, labels,  param, predictors_columns)


# In[200]:


objects3 =(
  'Choice Reaction Time (CRT)',
  'Simple Reaction Time (SRT)',
 'Body Mass Index (BMI)',
 'Olfaction Score (OLF)',
 'Bowel Movement Frequency (BMF)',
 'The Cognitive Abilities Screening Instrument Score (CASI)',
 'Age (AGE)',
 'Coffee (COF)',
 'Daytime Sleepiness (DTS)',
 'Smoking (SMK)',                       
 'Traumatic Brain Injury (TBI)',
 'Hypertension (HYP)'
 )


# In[201]:


objects4 =(
  'Choice Reaction Time',
  'Simple Reaction Time',
 'Body Mass Index',
 'Olfaction Score',
 'Bowel Movement Frequency',
 'The Cognitive Abilities Screening Instrument Score',
 'Age',
 'Coffee',
 'Daytime Sleepiness',
 'Smoking',                       
 'Traumatic Brain Injury',
 'Hypertension'
 )


# In[202]:


dpi_val=600

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,4))


# Example data

ax.barh(y_pos, performance, align='center', color='grey')
        #, color = direction_color_arg )
ax.set_yticks(y_pos)
ax.set_yticklabels(objects3)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain Importance')
#ax.set_title('Variable Importance Analysis')
plt.yticks(fontsize=12, weight = 'bold')
plt.savefig("VIA_model5d_same_color.png", dpi=dpi_val, bbox_inches='tight')
plt.show()


# In[203]:


dpi_val=600

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,4))


# Example data

ax.barh(y_pos, performance, align='center', color='grey')
        #, color = direction_color_arg )
ax.set_yticks(y_pos)
ax.set_yticklabels(objects3)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain Importance')
#ax.set_title('Variable Importance Analysis')
plt.yticks(fontsize=12, weight = 'bold')
plt.savefig("VIA_model5d_same_color.eps",format='eps',  dpi=dpi_val, bbox_inches='tight')
plt.show()


# In[204]:


dpi_val=600

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,4))


# Example data

ax.barh(y_pos, performance, align='center', color='grey')
        #, color = direction_color_arg )
ax.set_yticks(y_pos)
ax.set_yticklabels(objects4)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain Importance')
#ax.set_title('Variable Importance Analysis')
plt.yticks(fontsize=12, weight = 'bold')
plt.savefig("VIA_model5d_same_color_V2.png", dpi=dpi_val, bbox_inches='tight')
plt.show()


# In[205]:


dpi_val=600

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,4))


# Example data

ax.barh(y_pos, performance, align='center', color='grey')
        #, color = direction_color_arg )
ax.set_yticks(y_pos)
ax.set_yticklabels(objects4)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain Importance')
#ax.set_title('Variable Importance Analysis')
plt.yticks(fontsize=12, weight = 'bold')
plt.savefig("VIA_model5d_same_color_V2.eps",format='eps',  dpi=dpi_val, bbox_inches='tight')
plt.show()


# In[210]:


new_predictors_names = [predictors_names[i] for i in myorder]

new_direction_color = [direction_color[i] for i in myorder]
W_list = (np.array(Sim_conf_inter)[:,0] - np.array(Sim_conf_inter)[:,1])
new_W_list = [W_list[i] for i in myorder]
mean_of = np.array(Sim_conf_inter)[:,0]
new_mean_of = [mean_of[i] for i in myorder]
plt.figure(figsize=(12,6))
plt.errorbar(x=new_mean_of[::-1], y=range(12), xerr=new_W_list[::-1], fmt='o', ecolor='grey', lw=3)

y = np.linspace(0, 12,100)
x = y*0
plt.plot(x,y, ':b',color='grey', label='Threshold Value')

plt.yticks(y_pos, objects3[::-1], rotation='horizontal')
plt.yticks(fontsize=12, weight = 'bold')



plt.savefig("Final_EB.png", dpi=dpi_val, bbox_inches='tight')

plt.show()


# In[211]:


new_predictors_names = [predictors_names[i] for i in myorder]

new_direction_color = [direction_color[i] for i in myorder]
W_list = (np.array(Sim_conf_inter)[:,0] - np.array(Sim_conf_inter)[:,1])
new_W_list = [W_list[i] for i in myorder]
mean_of = np.array(Sim_conf_inter)[:,0]
new_mean_of = [mean_of[i] for i in myorder]
plt.figure(figsize=(12,6))
plt.errorbar(x=new_mean_of[::-1], y=range(12), xerr=new_W_list[::-1], fmt='o', ecolor='grey', lw=3)

y = np.linspace(0, 12,100)
x = y*0
plt.plot(x,y, ':b',color='grey', label='Threshold Value')

plt.yticks(y_pos, objects3[::-1], rotation='horizontal')
plt.yticks(fontsize=12, weight = 'bold')



plt.savefig("Final_EB.eps",format='eps', dpi=dpi_val, bbox_inches='tight')

plt.show()


# In[212]:


new_predictors_names = [predictors_names[i] for i in myorder]

new_direction_color = [direction_color[i] for i in myorder]
W_list = (np.array(Sim_conf_inter)[:,0] - np.array(Sim_conf_inter)[:,1])
new_W_list = [W_list[i] for i in myorder]
mean_of = np.array(Sim_conf_inter)[:,0]
new_mean_of = [mean_of[i] for i in myorder]
plt.figure(figsize=(12,6))
plt.errorbar(x=new_mean_of[::-1], y=range(12), xerr=new_W_list[::-1], fmt='o', ecolor='grey', lw=3)

y = np.linspace(0, 12,100)
x = y*0
plt.plot(x,y, ':b',color='grey', label='Threshold Value')

plt.yticks(y_pos, objects4[::-1], rotation='horizontal')
plt.yticks(fontsize=12, weight = 'bold')



plt.savefig("Final_EB_v2.png", dpi=dpi_val, bbox_inches='tight')

plt.show()


# In[213]:


new_predictors_names = [predictors_names[i] for i in myorder]

new_direction_color = [direction_color[i] for i in myorder]
W_list = (np.array(Sim_conf_inter)[:,0] - np.array(Sim_conf_inter)[:,1])
new_W_list = [W_list[i] for i in myorder]
mean_of = np.array(Sim_conf_inter)[:,0]
new_mean_of = [mean_of[i] for i in myorder]
plt.figure(figsize=(12,6))
plt.errorbar(x=new_mean_of[::-1], y=range(12), xerr=new_W_list[::-1], fmt='o', ecolor='grey', lw=3)

y = np.linspace(0, 12,100)
x = y*0
plt.plot(x,y, ':b',color='grey', label='Threshold Value')

plt.yticks(y_pos, objects4[::-1], rotation='horizontal')
plt.yticks(fontsize=12, weight = 'bold')



plt.savefig("Final_EB_v2.eps",format='eps', dpi=dpi_val, bbox_inches='tight')

plt.show()


# In[ ]:




