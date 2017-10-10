import numpy as np
from math import sqrt
from collections import Counter
#import warnings
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
np.set_printoptions(precision=3) #just to make easy for the viewer


def model_knn(data,predict,k): # data passing is of type dictionary and predict is test data is lis
  distances=[]   #distance,group name
  for group in data:
    for features in data[group]:
      euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))  #euclidean_distance
      distances.append([euclidean_distance,group])
  votes = [i[1] for i in sorted(distances)[:k]] #sorting first k neighbours
  max_class=Counter(votes).most_common(1) 
  #confidence=(float(max_class[0][1])/k)*100
  vote_result=max_class[0][0]
  return vote_result


def feature_normalization(tr_data):
  scaler = MinMaxScaler(feature_range=(0, 1))
  rescaledX = scaler.fit_transform(tr_data)
  return rescaledX

def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out
'''
def cross_validation(X, Y):   #X is a list of lists [[],[],[],[]] and Y is []
  #we are doing 5 fold cross cross_validation
  #a=len(X)/5.0
  result={}
  for loop in range(2): 
      X_val=chunkIt(X,5)
      Y_val=chunkIt(Y,5)
      #X_val=[X[:a],X[a:2*a],X[2*a:3*a],X[3*a:4*a],X[4*a:5*a],X[5*a:]]  #X_val for validation set this contains training values X-val=[ [ [1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8] ], [ [],[],[] ], [ [],[],[] ], [ [],[],[] ]   ]
      #Y_val=[Y[:a],Y[a:2*a],Y[2*a:3*a],Y[3*a:4*a],Y[4*a:5*a],Y[5*a:]]  # Y is [ [2,2,2],[2,4,2],[2,4,2],[2,2,2]  ]
      
      for i in range(len(X_val)): # this for loop shoup be entire method
          test_size=0.2
          train_set={0:[],1:[]}
          test_set={0:[],1:[]}
          train_data_X = X_val[i] #first 80 percent by leaving 20 percent last data  type  [ [1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8] ]
          train_data_Y = Y_val[i]      #[2,2,4]
          temp1 = X_val[:i]+X_val[i+1:] #last 20 percent of data same as X_val syntax
          temp2 = Y_val[:i]+Y_val[i+1:]  #same as Y_val syntax
          test_data_X = [item for sublist in temp1 for item in sublist]   #[ [1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8] ]
          test_data_Y= [item for sublist in temp2 for item in sublist]  #[2,2,4]

      

          j=0  # for inserting in to the dictionary
          for i in train_data_X:
              t=train_data_Y[j]
              train_set[t].append(i[:]) #select last element which is class and append value to the one with this class
              j+=1

          j=0
          for i in test_data_X:
              t=test_data_Y[j]
              test_set[t].append(i[:])    # fill disctionary same as top for  

          set = list(range(1,15))
          odd_list = filter(lambda x: x % 2 != 0, set)   #choosing odd values of k as our number of classes is even
          
          
          for k in odd_list:
             correct=0
             total=0  
             for group in test_set:
                 for data in test_set[group]: #data is all features.
                     vote=model_knn(train_set,data,k)
                     if group==vote:
                         correct+=1
                     total+=1
             res1=correct/total
             if k in result.keys():
                 result[k]=result[k]+res1
             else:
                 result[k]=res1
  k_opt=max(result, key=result.get)  #optimum k for single train test loop
  E_performance=(result[k_opt])/10.0 #5 per loop
  plt.plot([x for x in result],[y/10.0 for y in result.values()],label="line")
  plt.title("K-NN")
  plt.xlabel("K")
  plt.ylabel("Accuracy")
  plt.show()
  print (k_opt)
  return k_opt
  
'''
df=pd.read_csv("C:/Users/MBD/Desktop/gre_college_applications/final-documents/TAMU/after-going-to-college/1st-sem/Machine Learning/projects/k-nn/train.csv")
df.loc[df['area'] > 0, 'area'] = 1
full_data=df.astype(float).values  #.tolist() #full_data contains row matrix of attributes
random.shuffle(full_data)
X = full_data[:,:-1]   #[[],[],[]]   train input
Y = full_data[:,-1]    #[]     train output



Xnorm=feature_normalization(X)  #normalized features for full data
#k_optimal=cross_validation(Xnorm,Y)

#below for test data
df2=pd.read_csv("C:/Users/MBD/Desktop/gre_college_applications/final-documents/TAMU/after-going-to-college/1st-sem/Machine Learning/projects/k-nn/test.csv")
df2.loc[df2['area'] > 0, 'area'] = 1
full_data_test=df2.astype(float).values  #.tolist() #full_data contains row matrix of attributes
#random.shuffle(full_data_test)
X_test = full_data_test[:,:-1]   #[[],[],[]]   train input
Y_test = full_data_test[:,-1] 
Xnorm_test=feature_normalization(X_test) 


train_set={0:[],1:[]}
test_set={0:[],1:[]}
train_data_X=Xnorm
train_data_Y=Y
test_data_X= Xnorm_test # change to Xnorm_test
test_data_Y=Y_test

j=0  # for inserting in to the dictionary
for i in train_data_X:
  t=train_data_Y[j]
  train_set[t].append(i[:]) #select last element which is class and append value to the one with this class
  j+=1 #select last element which is class and append value to the one with this class
      

j=0  # for inserting in to the dictionary
for i in test_data_X:
  t=test_data_Y[j]
  test_set[t].append(i[:]) #select last element which is class and append value to the one with this class
  j+=1

  
'''
for i in test_data:
  test_set[i[-1]].append(i[:-1])    # fill disctionary same as top for
'''
accuracies=[]
for i in range(1): #remove this loop if you dont want multiple values
    correct=0
    total=0
    for group in test_set:
      for data in test_set[group]: #data is all features.
          vote=model_knn(train_set,data,1)
          if group==vote:
              correct+=1
          #else:
              #print (confidence)
              
          total+=1
          
    #print ('Accuracy:',correct/total)
    accuracies.append(correct/total)   #outside for loop so for entire thing our model accuracy is what we are seeing
print (sum(accuracies)/len(accuracies))