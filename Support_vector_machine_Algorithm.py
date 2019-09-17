import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('file_k-nn_headings.csv')  # selecting the dataset
print(dataset.describe())
X = dataset.iloc[:, 0:8]  #splitting the dataset 0-8 are attributes
y = dataset.iloc[:, 8] # and 8th one is the label attribute that we need to predict
sns.heatmap(X.corr(), annot = True)  # plotting the heat map to get correlation
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']  # setting the dataset to clear the missing values
for column in zero_not_accepted:
    X[column] = X[column].replace(0, np.NaN) # first replacing the 0’s with NAN
    median = int(X[column].median(skipna=True))  # calculating median just as we did previously by skipping NAN values
    X[column] = X[column].replace(np.NaN, median) # Now replacing the NAN values with the calculated median.
## Var[X] = p(1-p)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))  # used this to remove all low variance values from the dataset, Features with a training-set variance lower than this threshold will be removed.
X_filtered = sel.fit_transform(X) # fitting the threshold to data and transforming it
print(X.head(1))
print(X_filtered[0])
#DiabetesPedigreeFunction was dropped since this didn’t met the threshold parameters look in X_filtered this function value wont be there in the array so we dropped it.
X = X.drop('DiabetesPedigreeFunction', axis=1)
top_4_features = SelectKBest(score_func=chi2, k=4)# selecing features corresponding to highest scores, where k= Number of top features to select, this will take array X and y and returns scores. Ch2== chi-squared stats of non-negative features for classification.
X_top_4_features = top_4_features.fit_transform(X, y) # we need fit the above features to our dataset
print(X.head())
print(X_top_4_features)
X = X.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20) # splitting dataset into training and testing
sc_X = StandardScaler()  #this will remove mean and scales to unit variance==(value-mean)/standard deviation 
X_train = sc_X.fit_transform(X_train) # fitting the scaling to dataset
X_test = sc_X.transform(X_test)
classifier = SVC(random_state=0, kernel='rbf') # implementing the support vector machine using Linear kernel
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)  #predicting the results 
cm = confusion_matrix(y_test, y_pred) #generates confusion matrix containing total cases used for calculating above mentioned recall, precision and accuracy.
print (cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
