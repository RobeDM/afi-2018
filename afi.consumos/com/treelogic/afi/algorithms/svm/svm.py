#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import numpy as np

np.random.seed(12)
#selection = np.array([2,9,0,6,4,10,1,7,3])

class SVMModel:
    def __init__(self, mySVC, myNormalizer, myValidation):
        self.classifier = mySVC
        self.normalizer = myNormalizer
        self.validation = myValidation


def trainSVM(data,labels,max_data=-1):

    # RANDOMIZE DATA
    (nSamples,nFeatures)=data.shape
    randomPermutation = np.random.permutation(nSamples)
    data=data[randomPermutation,:]
    labels=labels[randomPermutation]

    # NORMALIZE  MEAN 0 Y VARIANCE 1
    normalizador = Normalizer()
    data=normalizador.fit_transform(data)


    # LIMIT DATA
    if (max_data>0):
        data=data[0:max_data,:]
        labels=labels[0:max_data]


    
    BestC,BestG,BestAccuracy = _crossValidation(data,labels)
    
    # TRAINING
    print "Training model c=",BestC," y gamma=", BestG
    svmclassifier = SVC(C=BestC, gamma=BestG, kernel='rbf',cache_size=4000)
    svmclassifier.fit(data,labels)

    return SVMModel(svmclassifier,normalizador, BestAccuracy)


def _crossValidation(data,labels):
    
    # HYPERPARAMETERS
    Cs=[100,10]
    gammas=[10,100]
    
    CVfolds=StratifiedKFold(labels,4)
    (train,test)=list(CVfolds)[0]
    BestAccuracy=-1.0
    BestC = -1
    BestG = -1


    for c in Cs:
        for gamma in gammas:
            Accuracy = 0.0
            print("Training parameters: C=",c," y gamma=",gamma)

            Xcvtr=data[train,:]
            Xcvtst=data[test,:]
            Ycvtr=labels[train]
            Ycvtst=labels[test]

            svmclassifier = SVC(C=c, gamma=gamma, kernel='rbf',cache_size=4000)
            svmclassifier.fit(Xcvtr,Ycvtr)
            predictions=svmclassifier.predict(Xcvtst)
            Accuracy = np.float((predictions==Ycvtst).sum())/Ycvtst.shape[0]

            if(Accuracy> BestAccuracy):
                BestAccuracy = Accuracy
                BestC = c
                BestG = gamma

            print("Right Now Best Accuracy CV:",BestAccuracy, "C:", BestC, "G:",BestG)

    print("Best Accuracy CV:",BestAccuracy, "C:", BestC, "G:",BestG)
    return BestC,BestG,BestAccuracy


def saveSVMModel(model, file):
    pickle.dump( model, open( file, "wb" ) )


def loadSVMModel(file):
    return pickle.load( open( file, "rb" ) )


def predict(data,model):
    data = model.normalizer.transform(data)
    return model.classifier.predict(data)


def testSVM(data,labels,model):
    data = model.normalizer.transform(data)
    predictions = model.classifier.predict(data)
    accuracy = np.float(np.float((predictions==labels).sum()) / np.float(labels.shape[0]))
    return predictions,accuracy
