# -*- coding: utf-8 -*-
"""
Created on Tue Oct 01 09:14:09 2013

@author: madhav.kumar

Objective: Perform k-fold cross-validation using sklearn
            and output predictions on training and test

"""
import numpy as np
import pandas as pd

# Functions to save and load pickle objects
def saveObject(obj, filename):
    out = open(filename, "wb")
    pickle.dump(obj, out)
    out.close()
    
# Function to load pickle objects
def loadObject(filename):
    pkl_file = open(filename, "rb")
    out = pickle.load(pkl_file)
    pkl_file.close()
    return out

# Function to create n cross validation folds
def createFolds(data, nfolds):
    rows = data.shape[0]
    folds = range(0, nfolds)*ceil(rows/nfolds)
    folds = folds[0:rows]
    np.random.shuffle(folds)
    data['fold'] = folds
    return data

# Cross validate model    
def cvModel(train, test, target, feat, idcol, k, model, classify= True,
            model_test= "model.pkl"):
    ''' Train a model using k-fold cross validation
        and return cross-validated predictions on 
        training and test data sets
    '''
    val_pred = pd.DataFrame({idcol: [], 
                             'target': [],
                             'pred': []})
    test_pred = pd.DataFrame(test[idcol])
    
    for i in range(0, k):
        print("Fold", i+1, "of", k)
        tr = train['fold'] != i
        va = train['fold'] == i
        # fit model
        model.fit(train.ix[tr, feat], target[tr])
        # score on validation
        scored = pd.DataFrame({idcol: train.ix[va, idcol], 'target': target[va]})
        if classify:
            scored['pred'] = model.predict_proba(train.ix[va, feat])[:,1]
        else:
            scored['pred'] = model.predict(train.ix[va, feat])        
        val_pred = val_pred.append(scored)
        print("Scored on val")
        
        
    print("Build model on entire training data")
    model.fit(train[feat], target)
    saveObject(model, model_test)
    # score on test
    if classify:
        test_pred['pred'] = model.predict_proba(test[feat])[:,1]
    else:
        test_pred['pred'] = model.predict(test[feat])
    print("Scored on test")
    
    return val_pred, test_pred
    
