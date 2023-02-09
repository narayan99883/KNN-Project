#Data Set
#https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease 


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
bmiFullArray=[]
bmiArray=[]
bmiPositive=[]
bmiNegative=[]
filename='heart.csv'
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    print(header_row)
    for row in reader:
        bmiFullArray.append(row[1])
    

            
        

#plt.figure(figsize=(8,6))
#plt.scatter( bmiArray,bmiArray, s=50, color='r')
#plt.xlabel('BMI')

#plt.title('scatter plot R Value: '+str(R_sq))
#plt.plot(x,a*x+b)
#plt.show()
    
