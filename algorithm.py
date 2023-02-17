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


#Read in dataset using pandas
filename='heart_2020_cleaned.csv'
numberOfRows = 100
dSet = pd.read_csv(filename, nrows=numberOfRows)

#print(dSet.to_string())

#Turning Strings into ints with dictionary attached
le = preprocessing.LabelEncoder()
OneHotEncoder().fit_transform(dSet)
#Dictionary
d = defaultdict(LabelEncoder)

print(dSet.to_string())

#Splitting dataset by target and feature
dSet_target = dSet[["HeartDisease"]]
print(dSet_target.to_string())
dSet_features = dSet[["BMI", "Smoking", "Stroke", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "SleepTime", "Asthma"]]
print(dSet_features.to_string())

#Getting data ready for testing
X_train, X_test, y_train, y_test = train_test_split(dSet_features, dSet_target, test_size=0.3)


#normalize data
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler()
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, np.ravel(y_train,order='C'))
KNeighborsClassifier(n_neighbors=10)
y_pred = classifier.predict(X_test)

#Printing results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

