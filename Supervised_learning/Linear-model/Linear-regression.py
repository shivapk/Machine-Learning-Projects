import numpy as np
from math import sqrt,log
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array, dot, transpose
from numpy.linalg import pinv, inv

def linear_regression(x_test):
    df=pd.read_csv("train.csv")
    df = df[df.area != 0]
    full_data=df.astype(float).values
    X_train = full_data[:,:-1]  
    Y_train = full_data[:,-1]
    
    X = np.array(X_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones,X))
    y = np.array(Y_train)
    
    Xt = transpose(X)
    product = dot(Xt, X)
    theInverse = pinv(product)
    w = dot(dot(theInverse, Xt), y)
    
    #prediction
    X_test = np.array(x_test)
    y_model=[]
    for i in X_test:
        components = w[1:] * i
        y_model.append(sum(components) + w[0])
        
    return y_model

  
def non_linear_regression(x_test):
     # on test data error increased with non-linear implementation..linear-> 103808.628527, non-linear->100639.40819(sudden increase at M=9)
    df=pd.read_csv("train.csv")
    df = df[df.area != 0]
    full_data=df.astype(float).values
    X_train = full_data[:,:-1]  
    Y_train = full_data[:,-1]
    #for M in range(9):
    U=X_train+X_train**2+X_train**3+X_train**4+X_train**5+X_train**6+X_train**7+X_train**8+X_train**9
    X = np.array(U)
    ones = np.ones(len(X))
    X = np.column_stack((ones,X))
    y = np.array(Y_train)
    Xt = transpose(X)
    product = dot(Xt, X)
    theInverse = pinv(product)
    w = dot(dot(theInverse, Xt), y)
    
    #prediction
    X_test = np.array(x_test+x_test**2+x_test**3+x_test**4+x_test**5+x_test**6+x_test**7+x_test**8+x_test**9)
    y_model=[]
    for i in X_test:
        components = w[1:] * (i)
        y_model.append(sum(components) + w[0])
        
    
        
    return y_model
   
    
    
def Rss_error(y_test,y_model):
    rss=[(y1-y2)**2 for y1,y2 in zip(y_test,y_model)]
    return sum(rss)
    
def histogram():
    df=pd.read_csv("train.csv")
    df = df[df.area != 0]
    full_data=df.astype(float).values  
    Y_train = full_data[:,-1] 
    print(Y_train)
    bins=[i for i in range(-50,50)]
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist([log(y,10) for y in Y_train],bins,histtype='bar',rwidth=0.7)
    
    axarr[1].hist(Y_train,bins,histtype='bar',rwidth=0.7)

    
    #plt.ylabel('Frequency')
    #plt.xlabel('x')
    plt.show()    
    
   
    #find f(x)=w0+wT*x
    #plot f(x) vs x
#histogram() 

#X_norm_train=preprocessing.scale(X_train)
df=pd.read_csv("train.csv")
df = df[df.area != 0]
full_data=df.astype(float).values
X_train = full_data[:,:-1]  
Y_train = full_data[:,-1]

df2=pd.read_csv("test.csv")
df2 = df2[df2.area != 0]
full_data2=df2.astype(float).values
X_test = full_data2[:,:-1]  
Y_test = full_data2[:,-1]
#X_norm_test=preprocessing.scale(X_test)
#non_linear_regression(X_test)
RSS_linear=Rss_error(Y_test,linear_regression(X_test))  #change it to test later
#RSS_non_linear=Rss_error(Y_test,non_linear_regression(X_test))

RSS_non_linear=Rss_error(Y_test,non_linear_regression(X_test))  #change it to test later
print (RSS_linear)
print (RSS_non_linear)
