import numpy as np
from math import sqrt
from collections import Counter
import warnings
import pandas as pd
import random
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv")
df = df[df.area != 0]
full_data=df.astype(float).values
X_train = full_data[:,:-1]  
Y_train = full_data[:,-1]
X_norm_train=preprocessing.scale(X_train)


df2=pd.read_csv("test.csv")
df2 = df2[df2.area != 0]
full_data2=df2.astype(float).values
X_test = full_data2[:,:-1]  
Y_test = full_data2[:,-1]
X_norm_test=preprocessing.scale(X_test)

print (len(X_norm_train),len(Y_train))
print (len(X_norm_test),len(Y_test))

clf=LinearRegression()
clf.fit(X_norm_train,Y_train)
accuracy=clf.score(X_norm_test,Y_test)

print(accuracy)
