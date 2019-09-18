import pandas as pd
import numpy as np
import random
import sys
from impyute.imputation.cs import fast_knn
df = pd.read_csv(r'./DATA/diabetes.csv')
#before adding nan
print(df.head(10))

nan_percent = {'Pregnancies':0.10, 'Glucose':0.15, 'BloodPressure':0.10,'SkinThickness':0.12,'Insulin':0.10,'BMI':0.13,'DiabetesPedigreeFunction':0.11,'Age':0.11,'Outcome':0.12}

for col in df:
    for i, row_value in df[col].iteritems():
        if random.random() <= nan_percent[col]:
            df[col][i] = np.nan
#after adding nan            
print(df.head(10))
df.to_csv(r'NaNdiabetes3.csv')
diab = np.genfromtxt('NaNdiabetes3.csv', delimiter=",")
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS

# start the KNN training
imputed_training=fast_knn(diab, k=3)
print(imputed_training)
pd.DataFrame(imputed_training).to_csv("file.csv")
