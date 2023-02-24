#imports
import numpy as np
import csv
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Warnings to make sure I keep getting error
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


#Read in dataset using pandas
filename='heart_2020_cleaned.csv'
numberOfRows = 25000
dSet = pd.read_csv(filename, nrows=numberOfRows)

#print(dSet.to_string())

#Indexing Strings
#No = 0, Yes = 1
#Female = 0, Male = 1
#Age category has 0 being the oldest, 12 being the youngest
#Race is (from 0 to 5) White, Black, Asian, Native, Hispanic, Other
#Diabetes is no/yes with no, borderline being 2 and yes during preganancy being 3
dSet['HeartDisease']=dSet['HeartDisease'].map({'No':0, 'Yes':1})
dSet['Smoking']=dSet['Smoking'].map({'No':0 ,'Yes':1})
dSet['Stroke']=dSet['Stroke'].map({'No':0 ,'Yes':1})
dSet['Sex']=dSet['Sex'].map({'Female':0 ,'Male':1})
dSet['AgeCategory']=dSet['AgeCategory'].map({'80 or older':0 ,'75-79':1, '70-74':2, '65-69':3, '60-64':4, '55-59':5, '50-54':6, '45-49':7, '40-44':8, '35-39':9, '30-34':10, '25-29':11, '18-24':12})
dSet['Race']=dSet['Race'].map({'White':0 ,'Black':1,'Asian':2,'American Indian/Alaskan Native':3,'Hispanic':4,'Other':5})
dSet['Diabetic']=dSet['Diabetic'].map({'No':0 ,'Yes':1,'No, borderline diabetes':2,'Yes (during pregnancy)':3})
dSet['PhysicalActivity']=dSet['PhysicalActivity'].map({'No':0 ,'Yes':1})
dSet['Asthma']=dSet['Asthma'].map({'No':0 ,'Yes':1})
    
#Printing
#print("Printing Data Set after indexing:")
#print(dSet.to_string())

#Splitting dataset by target and feature
dSet_target = dSet[["HeartDisease"]]
#print(dSet_target.to_string())
dSet_features = dSet[["BMI", "Smoking", "Stroke", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "SleepTime", "Asthma"]]
#print(dSet_features.to_string())

#Getting data ready for testing
X_train, X_test, y_train, y_test = train_test_split(dSet_features, dSet_target, test_size=0.3)

#normalize data (standardizing values)
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler()

#KNN Algorithm!
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=30)
classifier.fit(X_train, np.ravel(y_train,order='C'))
KNeighborsClassifier(n_neighbors=30)
y_pred = classifier.predict(X_test)

#Printing results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#print accuracy score
accuracy =  accuracy_score(y_test,y_pred)*100
print(accuracy)
#Usually 90.5 - 90.7

#Results:
#Best Sample Size is around 25k, Best k value is about 30
#????
#7000 people - we say no heart attack, they actually dont have heart attack (correct)
#691 people - we say heart attack, they actually dont have heart attack (wrong) (we are just being safe wrong)
#19 people - we say no heart attack, they actually did (wrong) (this is bad wrong)
#22 people - we say heart attack, they actually did have heart attack (correct)


#Still To Do (not in order):

#Understand confusion matrix and classificaion report
#Fix warnings

#Remove outliers and stuff, removing empty values (clean data)

#Plot k-optimization and find best k-value (on Potter's big example)

#Make graphs for different sample sizes, test-train ratio
#Cross validation

#Weight different columns!!

#Random graphs for different variables

#ORDER
# - Understand confusion matrix and classification report
# - Fix warnings for KNN thing saying that 0.0 is being put in
# Clean data (remove outliers, remove empty values, weird false data)
# Graphs for our data
# Weight different columns differently
# K optimization, sample size optimization, test-train ratio optimization with graphs
# Cross validation to improve performance (?)

#Write paper

