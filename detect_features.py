#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script investigating the improvement of model accuracy for erythromycin 
prediction

Developed by: Harry Chown

Date: 03/09/21
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.svm import LinearSVC
from numpy import mean, std
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import sys  
import warnings

def removeUniq(inputData):
    """ Remove features which are only present/absent in a single organism"""
    remove_feat = []
    for feat_index in range(0, len(inputData.columns)):
        feat = list(inputData.iloc[:,feat_index])
        uniq_feat_var = set(feat)
        for var in uniq_feat_var:
            if feat.count(var) == 1:
                remove_feat.append(feat_index)

    outData = inputData.drop(inputData.columns[remove_feat], axis=1, inplace=False)  
    return(outData)


def loocvModel(X_input, Y_input, input_model, probability=True):
    # Create method for LOOCV
    loocv = LeaveOneOut()
    # Store all y predictions and scores
    all_y_pred=[]
    all_y_score=[]
    collective_weights={}
    # Split dataset and perform modelling
    for train_index, test_index in loocv.split(X_input):
        X_train, X_test = X_input.iloc[train_index, :], X_input.iloc[test_index, :]
        y_train, y_test = Y_input[train_index], Y_input[test_index]
        # Fit model to training datasets
        input_model.fit(X_train, y_train)
        # Make predictions on the testing data set
        y_pred = input_model.predict(X_test)
        all_y_pred.append(int(y_pred))
        
        if probability==True:
            y_score = input_model.predict_proba(X_test)[:, 1]
        else:
            y_score = input_model.decision_function(X_test)
        all_y_score.append(y_score)
        
        # Retrieve feature weights from ensemble
        feature_weights = [estimator.coef_ for estimator in input_model.estimators_]
        feature_ids = input_model.estimators_features_
        for subspace_index in range(0, len(feature_ids)):
            for feature_index in range(0, len(feature_ids[subspace_index])):
               feature = feature_ids[subspace_index][feature_index]
               weight = feature_weights[subspace_index][0][feature_index]
               # Store the resulting weight in the dictionary
               collective_weights[feature] = collective_weights.get(feature, []) + [weight]
    # Obtain the weights for each feature
    average_feature_weights = []
    for feature, weights in collective_weights.items():
        average_feature_weights.append([feature, mean(weights), mean(weights)**2])
    # Convert the list to a dataframe
    out_weight_df = pd.DataFrame(average_feature_weights)
    out_weight_df.columns = ["feature_index", "weight", "sqrd_weight"]
    # Order weights
    out_weight_df = out_weight_df.sort_values("weight", ascending=False)
    out_weight_index=out_weight_df["feature_index"]
    out_weight_names=X_input.columns[out_weight_index]
    out_weight_df["feature_names"]=out_weight_names
    
   
    # Obtain scores for the model
    y_df=pd.DataFrame(list(zip(all_y_pred, Y_input)),
               columns =['y_pred', 'y_true'])
    fpr, tpr, thresholds = roc_curve(Y_input, all_y_score)
    cfm = confusion_matrix(Y_input, all_y_pred).ravel()
    AUC = auc(fpr, tpr)
    MCC = matthews_corrcoef(Y_input, all_y_pred)
    ACC = accuracy_score(Y_input, all_y_pred)
    loocv_score=[ACC, AUC, MCC]+list(cfm)
    loocv_output=[loocv_score, out_weight_df, y_df]
    return(loocv_output)

def topFeatures(originalData, featureData, colname, ascension = True):
    """Using feature weights, identify features that have the most importance"""
    # Order dataframe to identify the features importance
    df_sorted = featureData.sort_values(colname, ascending=ascension)
    top_features = list(df_sorted.iloc[0:20, 0])
    top_feature_names = [X_data.columns[col_id] for col_id in top_features]
    return([top_feature_names, df_sorted])

def subsetByFeature(originalData, importantData):
    # Subset data based on features with the highest importance
    squared_weight = list(importantData["sqrd_weight"])
    average_weight = mean(squared_weight)
    above_average = importantData.loc[importantData['sqrd_weight'] > average_weight]
    subsetData = originalData.iloc[:,above_average["feature_index"]]
    return(subsetData)

def runBagged(c_val, in_mf, in_ms, X_input, Y_input):
    linear_model = LinearSVC(penalty="l1", 
                                 class_weight='balanced', 
                                 loss='squared_hinge', 
                                 dual = False, 
                                 C = c_val)
    bagged_linear_model = BaggingClassifier(base_estimator=linear_model, 
                                            max_features=in_mf, 
                                            n_estimators = 50,
                                            random_state=1, 
                                            max_samples=in_ms, 
                                            bootstrap=True)
    # Perform modelling
    bagged_results=loocvModel(X_data, Y_data, bagged_linear_model)
    return(bagged_results)
"""
Obtain Data
"""
raw_data = pd.read_csv("/home/harry/Documents/partition_scripts/bacteria_pangenome/ML_indata/binary_matrix.csv", 
                       index_col=0)
raw_phenotype = pd.read_csv("/home/harry/Documents/partition_scripts/bacteria_pangenome/ML_indata/binary_phenotypes_edit.csv", 
                           index_col=0)
scale_bool = False
"""
Set outputs
"""
outdir="/home/harry/Documents/partition_scripts/bacteria_pangenome/ML_output/"
# Iterate through each drug group
# and store results
drug_init=[["drug", "acc", "auc", "mcc", "tn", "fp", "fn", "tp"]]
drug_improve=[["drug", "acc", "auc", "mcc", "tn", "fp", "fn", "tp"]]
drug_c=[["drug", "c_val"]]

#drug="erythromycin"
for drug in raw_phenotype.columns:
#for drug in ["erythromycin"]:
    y_phenotype = raw_phenotype[drug]
    print(drug)
    """
    Subset Data and clean
    """
    Y_data = y_phenotype.dropna()
    # Use this to subset input data, subset twice to remove ambiguities in Y and X
    #rel_data = input_x_data.loc[y_nan_removed.index,:]
    rel_data = raw_data.reindex(Y_data.index) 
    # Remove columns where all rows are the same
    nunique = rel_data.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    unique_row = rel_data.drop(cols_to_drop, axis=1)
    # Remove columns where there is only a difference in one organism
    X_data = removeUniq(unique_row)
    column_names = X_data.columns
    # Normalize data, if required
    if scale_bool == True:
        X_data
        scaler = StandardScaler()
        scaler.fit(X_data)
        X_data = pd.DataFrame(scaler.transform(X_data))
        X_data.columns = column_names
    """
    Perform Modelling
    """
    # Establish model
    mf=int(len(X_data.columns)*0.4)
    ms=int(len(Y_data)*0.8)
    # C optimisation
    c_log_scale = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 ]
    opt_mcc = 0
    opt_c = 0
    init_score=[]
    init_weight=[]
    init_pred=[]
    mcc_scale = []
    c_log_scale=[0.01]
    for c_1 in c_log_scale:
        results=runBagged(c_1, mf, ms, X_data, Y_data)
        mcc=results[0][2]
        mcc_scale.append(mcc)
        print(c_1)
        print(mcc)
        if mcc > opt_mcc:
            opt_mcc = mcc
            opt_c = c_1
            init_score=results[0]
            init_weight=results[1]
            init_pred=results[2]
        elif mcc < opt_mcc and opt_c > 0:
            print("Optimum C1: " + str(opt_c))
            c2_start = opt_c / 2
            print("Optimum MCC: " + str(opt_mcc))
            print("Start C2: " + str(c2_start))
            c2_end = opt_c + c2_start
        
            
            
            
            
            
    
        
    """
    Subset with important features
    """
    # Subset data for "improved" set
    # Extract features which have a sqrd weight greater than 0.001
    sub_df=init_weight.loc[init_weight.sqrd_weight>0.001]
    # Subset the X data to include only these features
    sub_X = X_data.iloc[:,sub_df["feature_index"]]
    """
    Establish a new model
    """
    # Establish model
    mf=int(len(sub_X.columns)*0.4)
    ms=int(len(Y_data)*0.8)
    results_improved=runBagged(opt_c, mf, ms, X_data, Y_data)
    """
    Perform modelling with feature subset
    """
    improve_score=results_improved[0]
    improve_weight=results_improved[1]
    improve_pred=results_improved[2]
    
    # Save/store results
    drug_init.append([drug]+init_score)
    drug_improve.append([drug]+improve_score)
    drug_c.append([drug]+[opt_c])
    
    init_weight.to_csv(outdir+drug+"_weight.csv", index=False)
    init_pred.to_csv(outdir+drug+"_pred.csv", index=False)
    
    improve_weight.to_csv(outdir+drug+"_imp_weight.csv", index=False)
    improve_pred.to_csv(outdir+drug+"_imp_pred.csv", index=False)
    


drug_df=pd.DataFrame(drug_init)
improved_df=pd.DataFrame(drug_improve)
c_df=pd.DataFrame(drug_c)

drug_df.to_csv(outdir+"init_scores.csv", index=False, header=False)
improved_df.to_csv(outdir+"imp_scores.csv", index=False, header=False)
c_df.to_csv(outdir+"opt_c.csv", index=False, header=False)