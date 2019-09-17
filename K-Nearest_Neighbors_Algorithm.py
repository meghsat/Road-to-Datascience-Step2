from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn
from pprint import pprint
from sklearn.svm import SVC

diab = np.genfromtxt('./DATA/pima_indians_diabetes.txt', delimiter=",")  #here I am taking .txt type dataset and converting it into csv file using delimiter.
X,y = diab[:,:-1], diab[:,-1:].squeeze() #X-contains first 7 columns(attributes used for associating and finding rules)  and y-- contains outcome class values(things we need to predict)
print (X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y) # here we are splitting the dataset containing 768 rows into two categories one for training(576 rows) and testing(192 rows)
print (X_train.shape, X_test.shape)
diab_knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train) #here we are using 3 neighbors and fitting the model using X_train for training and y_train as target values
y_pred = diab_knn.predict(X_test)  # predicting the class labels that we segregated for testing 
y_train_pred = diab_knn.predict(X_train) #predicting the class labels that we segregated for training, so here we have to get a very high accuracy since we are trying to predict the already available values
print ("Results with 3 Neighbors")
print (metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])   #using the “metrics.classification_report”on testing data we can generate the report containing precision, recall,f1-score and support, which I explained below
print (metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])) #using the “metrics.classification_report”on testing data we can generate the report containing precision, recall,f1-score and support
print ('\nConfussion matrix:\n',confusion_matrix(y_test,y_pred)) #generates confusion matrix containing total cases used for calculating above mentioned recall, precision and accuracy.
