#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import seaborn as sns
import gc
pd.options.mode.chained_assignment = None

import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score

import os
root = os.getcwd() + os.sep + 'data' + os.sep
image_fp = os.getcwd() + os.sep + 'image' + os.sep
submission_fp = os.getcwd() + os.sep


def reduce_memory(df):
    
    """
    This function reduce the dataframe memory usage by converting it's type for easier handling.
    
    Parameters: Dataframe
    Return: Dataframe
    """
    
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    
    for col in df.columns:
        if df[col].dtypes in ["int64", "int32", "int16"]:
            
            cmin = df[col].min()
            cmax = df[col].max()
            
            if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            
            elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            
            elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        if df[col].dtypes in ["float64", "float32"]:
            
            cmin = df[col].min()
            cmax = df[col].max()
            
            if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            
            elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    
    print("")
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    
    return df

try:
    model = xgb.Booster()
    model.load_model("model.txt")
    feature_names = pd.read_csv(root+'feature_names.csv').values.flatten()
    
except:
    df = pd.read_pickle(root + 'Finaldata.pkl')
    df = reduce_memory(df)
    
    df['order_diff'] = df.order_number - df.last_ordered_in
    
    label = 'reordered'
    x_cols = df.columns.drop('reordered')
    X = df[x_cols]
    y = df[label]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)
    
    from datetime import datetime
    starttime = datetime.now()
    print('\tstart time:', starttime)
    xgb_params = {
        "objective"        :"reg:logistic",
        "eval_metric"      :"logloss",
        "eta"              :0.1,
        "max_depth"        :6,
        "min_child_weight" :10,
        "gamma"            :0.70,
        "subsample"        :0.76,
        "colsample_bytree" :0.95,
        "alpha"            :2e-05,
        "scale_pos_weight" :10,
        "lambda"           :10
    }
    watchlist= [(D_train, "train")]
    model = xgb.train(params=xgb_params, dtrain=D_train, num_boost_round = 80, evals = watchlist, verbose_eval = 10)
    model.save_model("model.txt")
    pd.DataFrame(model.feature_names).to_csv(root+'feature_names.csv', index=False)
    endtime = datetime.now()
    print('\tend time:', endtime)
    print('\ttime consumption:', endtime-starttime)

    probability = model.predict(D_test)
    predictions = [1 if i > 0.5 else 0 for i in probability]
    print ("\nClassification report:\n",classification_report(y_test, predictions))
    print ("Accuracy Score:\t",accuracy_score(y_test, predictions))
    
    
    #confusion matrix
    conf_matrix = confusion_matrix(y_test,predictions)
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    sns.heatmap(conf_matrix, fmt = "d",annot=True, cmap='Blues')
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.title('Confuion Matrix')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    
    #f1-score
    f1 = f1_score(y_test, predictions)
    print("F1 Score:\t\t", f1)
    
    #roc_auc_score
    model_roc_auc = roc_auc_score(y_test,probability) 
    print ("Area Under Curve:\t", model_roc_auc, "\n")
    fpr,tpr,thresholds = roc_curve(y_test,probability)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    threshold = np.round(thresholds[ix],3)
    
    plt.subplot(122)
    plt.plot(fpr, tpr, color='darkorange', lw=1, label = "Auc : %.3f" %model_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best Threshold:' + str(threshold))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.gcf().tight_layout()
    plt.gcf().savefig(image_fp+'XGBoost Scores.png', dpi=144, transparent=True)
    plt.show()
    
    
    fig, ax = plt.subplots(figsize = (10,15))
    xgb.plot_importance(model, ax = ax)
    ax.set_title('Feature Importance')
    plt.gcf().tight_layout()
    fig.savefig(image_fp+'XGBoost Feature Importance Plot.png', dpi=144, transparent=True)
    
    
    
test_df = pd.read_pickle(root + 'Testdata.pkl')
test_df = reduce_memory(test_df)
test_df['order_diff'] = test_df.order_number - test_df.last_ordered_in

#### Save submission
D_pred = xgb.DMatrix(test_df)
probability = model.predict(D_pred)

i = np.where(probability>0.5)[0]
submission = pd.DataFrame(index=test_df.index[i]).reset_index('product_id').astype(str)
submission = submission.groupby(['order_id'])['product_id'].aggregate(lambda x: ' '.join(x))
submission = pd.DataFrame(index=test_df.index.get_level_values('order_id').unique()).sort_index()\
    .merge(submission, how='left', left_index=True, right_index=True)
submission.fillna('None', inplace=True)
submission.columns = ['products']
submission.to_csv(submission_fp+'submission.csv')




