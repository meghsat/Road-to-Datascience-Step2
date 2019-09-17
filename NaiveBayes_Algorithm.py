#Handling Data:
import csv
import random
import math
import pandas as pd      # importing pandas as “pd” to use pandas directly
def loadCsv():
         lines=csv.reader(open(r'C:\Users\dmsss\Downloads\dataminingproject\diabetesnew.csv')) #opening the csv file that we wanted to analyze
         dataset=list(lines)   #we are converting “csv.reader” type values into “list of values” that contains each row of dataset into a list (see after code part pics 1&2). 
         for i in range(len(dataset)):  #here the length of dataset==768 values, so range is (0,768)
                  dataset[i]=[float(x) for x in dataset[i]]    # we are converting everything into float datatype and refilling the list
        return dataset
#Splitting the data for training and testing part
def splitDataset(dataset, splitRatio):   
         trainSize = int(len(dataset) * splitRatio)    #splitRatio defines the ratio in which we need to divide the dataset for training and testing.
         trainSet = []
         copy = list(dataset)
        while len(trainSet) < trainSize:  
                  index = random.randrange(len(copy))  # random.randrange(100) generates a  random number between 0to100.
                  trainSet.append(copy.pop(index))  #appending elements into the trainset popping elements at random indexes
        return [trainSet, copy]
#Data Summarization
def separateByClass(dataset):    #here we are taking TRAINING SET as our dataset
                   separated = {}  #creating a python dictionary
                  for i in range(len(dataset)):
                                 vector = dataset[i]  #loading the dataset values into vector
                                if (vector[-1] not in separated): #here we are separating the training dataset by class value so that we can calculate statistics for each class
                                                        separated[vector[-1]] = []
#last attribute (-1) is the class value(the attribute we need to predict). The function returns a map of class values to lists of data instances.
                                                         separated[vector[-1]].append(vector)  
                                   return separated  # separated dictionary consists of all the rows segregated by their class value 
 
#We need to calculate the mean of each attribute for a class value.
def mean(numbers):
                      return sum(numbers)/float(len(numbers))

#We also need to calculate the standard deviation of each attribute for a class value. The standard deviation describes the variation of spread of the data, and we will use it to characterize the expected spread of each attribute in our Gaussian distribution when calculating probabilities.
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
return math.sqrt(variance)

# For a given list of instances (for a class value) we can calculate the mean and the standard deviation for each attribute.The zip function groups the values for each attribute across our data instances into their own lists so that we can compute the mean and standard deviation values for the attribute.
def summarize(dataset):
summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
del summaries[-1]  #clearing the class attribute summaries
return summaries

def summarizeByClass(dataset):
separated = separateByClass(dataset)
summaries = {}
for classValue, instances in separated.items():
summaries[classValue] = summarize(instances)
return summaries
4)	Make  PREDICTION
def calculateProbability(x, mean, stdev):
#calculate the exponent first, then calculate the main division. 
exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
probabilities = {}
#probability of an attribute belonging to a class, we can combine the probabilities of all of the attribute values for a data instance and come up with a probability of the entire data instance belonging to the class.
for classValue, classSummaries in summaries.items():
probabilities[classValue] = 1
for i in range(len(classSummaries)):
mean, stdev = classSummaries[i]
x = inputVector[i]
probabilities[classValue] *= calculateProbability(x, mean, stdev)
return probabilities

def predict(summaries, inputVector):
probabilities = calculateClassProbabilities(summaries, inputVector)
bestLabel, bestProb = None, -1
for classValue, probability in probabilities.items(): #calculate the probability of a data instance belonging to each class value, we can look for the largest probability and return the associated class.
if bestLabel is None or probability > bestProb:
bestProb = probability
bestLabel = classValue
return bestLabel

def getPredictions(summaries, testSet):
predictions = []
for i in range(len(testSet)):
result = predict(summaries, testSet[i])	#will do this and return a list of predictions for each test instance
predictions.append(result)  #appending the results to predictions
return predictions

def getAccuracy(testSet, predictions): #we will be comparing predictions to class values in dataset.
correct = 0
for i in range(len(testSet)):
if testSet[i][-1] == predictions[i]:
correct += 1
return (correct/float(len(testSet))) * 100.0 #accuracy is calculated as an accuracy ratio between 0 and 100%

def main():
filename = 'diabetesnew.csv'    #Taking the dataset
splitRatio = 0.67  #defining the split ratio in which we need to divide the dataset for training and testing.
dataset = loadCsv()   #calling the loadCsv method
trainingSet, testSet = splitDataset(dataset, splitRatio)   #splitting the dataset into training and testing
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
#for training we are using 514 and for testing we use 254 rows
# prepare model
summaries = summarizeByClass(trainingSet)
# test model
predictions = getPredictions(summaries, testSet)  #calling the predictions function to get predictions
accuracy = getAccuracy(testSet, predictions) #calculating accuracy
print('Accuracy: {0}%'.format(accuracy))

main()
