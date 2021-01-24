import os
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import lightgbm as lgb
from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_recall_curve
from statsmodels.stats import weightstats as stests
from sklearn.metrics import precision_recall_curve

import math
import statsmodels.stats.api as sms


thr = 0.26237751134639836


with open(sys.argv[1], "r") as file:
    labels = eval(file.readline())


predictors = pd.read_csv(sys.argv[2], index_col=0)
predictors

models = glob.glob('Models-5year\*5_year.txt')

#print (models)
pred_prob = []

for i in range(len(models)):
    clf_name = models[i]
    clf = lgb.Booster(model_file=clf_name)
    
    pred_prob.append(clf.predict(predictors, num_iteration=clf.best_iteration))
    
pred_prob_arg = np.mean(pred_prob, axis=0)
binary_pred = np.array([1 if i > thr else 0 for i in pred_prob_arg])


with open("risk_predictions", "w") as file:
    file.write(str(pred_prob_arg))


with open("binary_predictions", "w") as file:
    file.write(str(binary_pred))


#print('AUC is ' + str(roc_auc_score(labels, pred_prob_arg)) )
