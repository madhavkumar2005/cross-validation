# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 06:26:39 2013

@author: madhav.kumar
Modeling routine for Kaggle competitions
"""

import pandas as pd
import pickle

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

def buildModel(train, test, target, feat, idcol= "sort_col", valcol="val", 
               valid= 1, trainid= 0, mode= "both", model= "required",
               model_val = "model_val.pkl", model_test = "model_test.pkl",
               classify= True):
    ''' Train a model using subset of training data and score on validation.
        Refit the model on the entire training data and score on test.
    '''
    if mode in ["val", "both"]:
        tr = train[valcol] == trainid
        va = train[valcol] == valid
        model.fit(train.ix[tr, feat], target[tr])
        val_pred = pd.DataFrame({idcol: train.ix[va, idcol], 'target': target[va]})
        if classify:
            val_pred['pred'] = model.predict_proba(train.ix[va, feat])[:,1]
        else:
            val_pred['pred'] = model.predict(train.ix[va, feat])
        saveObject(model, model_val)
        print "Val model saved to disk"
        
    if mode in ["test", "both"]:
        model.fit(train[feat], target)
        test_pred = pd.DataFrame({idcol: test[idcol]})
        if classify:
            test_pred['pred'] = model.predict_proba(test[feat])[:,1]
        else:
            test_pred['pred'] = model.predict(test[feat])
        saveObject(model, model_test)
        print "Test model saved to disk" 
        
    if mode == "val":
        return val_pred
    elif mode == "test":
        return test_pred
    else:
        return val_pred, test_pred
		
