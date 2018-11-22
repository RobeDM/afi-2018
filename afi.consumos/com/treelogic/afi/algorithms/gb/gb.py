#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
import numpy as np
import pickle

np.random.seed(12)

class GBModel:
    
    def __init__(self, myGB, myNormalizer, myValidation):
        self.classifier = myGB
        self.normalizer = myNormalizer
        self.validation = myValidation


def trainXG(data,labels,max_data=-1):
    
    # RANDOMIZE DATA
    nSamples=data.shape[0]
    randomPermutation = np.random.permutation(nSamples)
    data=data[randomPermutation,:]
    labels=labels[randomPermutation]

    # NORMALIZE MEAN 0 Y VARIANCE 1
    normalizer = Normalizer()
    data=normalizer.fit_transform(data)


    # LIMIT TRAINING DATA
    if (max_data>0):
        data=data[0:max_data,:]
        labels=labels[0:max_data]


    BestDepth,BestAccuracy = _crossValidation(data,labels)
    
    # TRAIN
    print "Training model depth=",BestDepth
    xgbclassifier=xgb.XGBClassifier(max_depth=BestDepth, n_estimators=100, learning_rate=0.05,subsample=0.5)
    xgbclassifier.fit(data, labels)


    return GBModel(xgbclassifier,normalizer, BestAccuracy)


def _crossValidation(data,labels,FOLDS=3):
    depths=np.array([7,11])
    
    CVfolds=StratifiedKFold(labels,FOLDS)
    BestAccuracy=-1.0
    (train,test) = list(CVfolds)[0]
    
    for depth in depths:
        
            Accuracy = 0.0
            print "Training parameters: depth=",depth
            
            Xcvtr=data[train,:]
            Xcvtst=data[test,:]
            Ycvtr=labels[train]
            Ycvtst=labels[test]

            xgbclassifier=xgb.XGBClassifier(max_depth=depth, n_estimators=100, learning_rate=0.05,subsample=0.9)
            xgbclassifier.fit(Xcvtr,Ycvtr)

            predictions=xgbclassifier.predict(Xcvtst)
            Accuracy = Accuracy + np.float((predictions==Ycvtst).sum())/Ycvtst.shape[0]

            if(Accuracy> BestAccuracy):
                BestAccuracy = Accuracy
                BestDepth = depth
                

            print "Right Now Best Accuracy CV:",BestAccuracy, "Depth:", BestDepth

    print "Best Accuracy CV:",BestAccuracy, "Depth:", BestDepth
    return BestDepth,BestAccuracy


def saveGBModel(model, file):
    pickle.dump(model, open( file, "wb" ) )


def loadGBModel(file):
    return pickle.load(open(file, "rb"))


def predictGB(data,model):
    data = model.normalizer.transform(data)
    return model.predict(data)


def predictGB_proba(data,model):
    data = model.normalizer.transform(data)
    return model.predict_proba(data)

def testXG(data,labels,model):
    data = model.normalizer.transform(data)
    predictions = np.array(model.classifier.predict(data))
    accuracy = np.float((predictions==labels).sum())/labels.shape[0]
    return predictions,accuracy