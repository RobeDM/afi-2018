from read_events import ReadEvents 
from com.treelogic.afi.algorithms.gb.gb import trainXG, testXG


data,labels = ReadEvents('../dataset/dataset_eventos.csv')

nSamples = data.shape[0]

dataTrain = data[0:(int)(nSamples*0.7),:]
labelsTrain = labels[0:(int)(nSamples*0.7)]

dataTest = data[(int)(nSamples*0.7):,:]
labelsTest = labels[(int)(nSamples*0.7):]

modelXG = trainXG(dataTrain,labelsTrain)
(predictions,accuracy) = testXG(dataTest,labelsTest,modelXG)
print "Test accuracy: %s"%accuracy