import numpy as np
from sklearn import preprocessing

np.random.seed(12)

def ReadEvents(file):
    data = np.loadtxt(file,skiprows=1,delimiter=';',usecols=range(0,37))
    labels = np.loadtxt(file,skiprows=1,delimiter=';',usecols=(37,),dtype='str')
    (nSamples,nFeatures)=data.shape
    randomPermutation = np.random.permutation(nSamples)
    data=data[randomPermutation,:]
    labels=labels[randomPermutation]
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))
    labels = le.transform(labels)
    return data,labels
    