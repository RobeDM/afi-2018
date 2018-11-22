#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import sys

from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedKFold

import numpy as np


sys.setrecursionlimit(40000)

np.random.seed(12)

input_dim = 37
hidden = 300
output = 6  
    
class NNmodel:
    
    def __init__(self, myModel, myNormalizer):
        self.normalizer = myNormalizer
        self.classifier = myModel

def trainNN(data,labels,max_data=-1):
    
    # RANDOMIZE DATA
    nSamples=data.shape[0]
    randomPermutation = np.random.permutation(nSamples)
    data=data[randomPermutation,:]
    labels=labels[randomPermutation]
    numlabels = len(np.unique(labels))
    
    # NORMALIZE DATA - MEAN 0 Y VARIANCE 1
    normalizer = MinMaxScaler(feature_range=(0, 0.1))
    data=normalizer.fit_transform(data)
    
    # LIMIT NUMBER OF EXAMPLES
    if (max_data>0):
        data=data[0:max_data,:]
        labels=labels[0:max_data]
    
    BestLRate,BestAccuracy = _crossValidation(data,labels)
    
        
    labels = np.eye(numlabels)[np.int_(labels)]
    n_epoch = 20
    batch_size = 16
    
    #CREATING THE NEURAL NETWORK
    model = Sequential()
    model.add(Dense(hidden,input_dim=input_dim,init='lecun_uniform',activation='relu'))
    model.add(Dense(output_dim=6,init='lecun_uniform'))
    model.add(Activation('softmax'))
    decay = BestLRate/n_epoch
    nadam = Nadam(lr=BestLRate)
            
    model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
    model.fit(data, labels, nb_epoch=n_epoch, shuffle=True, batch_size=batch_size)
            
    return NNmodel(model,normalizer)

def _crossValidation(data,labels,FOLDS = 3):

    CVfolds=StratifiedKFold(labels,FOLDS)
    BestAccuracy=-1.0
    (train,test) = list(CVfolds)[0]
    numlabels = len(np.unique(labels))
    
    lrates = [0.001,0.01,0.1]

    Xcvtr=data[train,:]
    Xcvtst=data[test,:]
    Ycvtr=labels[train]
    Ycvtst=labels[test]
    
    Ycvtr = np.eye(numlabels)[np.int_(Ycvtr)]
    Ycvtst = np.eye(numlabels)[np.int_(Ycvtst)]
    

    for l in lrates:
        
            print "Training parameters: lrate=",l
            n_epoch = 10
            batch_size = 16
            #CREATING THE NEURAL NETWORK
            model = Sequential()
            model.add(Dense(hidden,input_dim=input_dim,init='lecun_uniform',activation='relu'))
            model.add(Dense(output_dim=6,init='lecun_uniform'))
            model.add(Activation('softmax'))
            decay = l/n_epoch

            nadam = Nadam(lr=l)
            model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
            model.fit(Xcvtr, Ycvtr, nb_epoch=n_epoch, shuffle=True, batch_size=batch_size)
            
            predictions = model.predict(Xcvtst)
            predictions =np.argmax(predictions,axis=1)
            labelsTest = np.argmax(Ycvtst,axis= 1)
            Accuracy = np.float((predictions==labelsTest).sum())/Ycvtst.shape[0]
            
            
            if(Accuracy> BestAccuracy):
                BestAccuracy = Accuracy
                BestLRate = l

                

            print "Right Now Best Accuracy CV:",BestAccuracy, "LRate:", BestLRate
    print "Best Accuracy CV:",BestAccuracy, "LRate:", BestLRate
    return BestLRate,BestAccuracy
    
def predictNN(data,model):
    data = model.normalizer.transform(data)
    result = model.classifier.predict(data)
    result =np.argmax(result,axis=1)
    return np.array(result)


def testNN(data,labels,model):
    data = model.normalizer.transform(data)
    predictions = model.classifier.predict(data)
    predictions =np.argmax(predictions,axis=1)
    accuracy = np.float((predictions==labels).sum())/labels.shape[0]
    return predictions,accuracy

def saveNNModel(modelo, file):
    pickle.dump( modelo, open(file, "wb" ) )


def loadNNModel(fichero):
    return pickle.load( open(file, "rb" ) )

