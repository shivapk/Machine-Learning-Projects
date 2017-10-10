import numpy as np
from math import sqrt
from collections import Counter
import warnings
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

np.set_printoptions(precision=3) #just to make easy for the viewer


def model_knn(data,predict,k): # data passing is of type dictionary and predict is test data is list
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

def cross_validation(X, Y):   #X is a list of lists [[],[],[],[]] and Y is []
  #we are doing 5 fold cross cross_validation
  #a=len(X)/5.0
  X_val=chuckIt(X,5)
  Y_val=chunkIt(Y,5)
  #X_val=[X[:a],X[a:2*a],X[2*a:3*a],X[3*a:4*a],X[4*a:5*a],X[5*a:]]  #X_val for validation set this contains training values X-val=[ [ [1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8] ], [ [],[],[] ], [ [],[],[] ], [ [],[],[] ]   ]
  #Y_val=[Y[:a],Y[a:2*a],Y[2*a:3*a],Y[3*a:4*a],Y[4*a:5*a],Y[5*a:]]  # Y is [ [2,2,2],[2,4,2],[2,4,2],[2,2,2]  ]
  result={}
  for i in range(len(X_val)): # this for loop shoup be entire method
      test_size=0.2
      train_set={2:[],4:[]}
      test_set={2:[],4:[]}
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

      set = list(range(1,10))
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
  E_performance=(result[k_opt])/5.0
  for data_dict in result.values():
    x = data_dict.keys()
    y = data_dict.values()/5
    plt.scatter(x,y)

  plt.legend(result.keys())
  plt.show()
  
  return k_opt
  








names=['id','clump_thickness','unif_cell_size','unif_cell_shape','marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses','class']

df=pd.read_csv("breast-cancer-wisconsin.data", names=names)
df.replace('?',-99999, inplace=True)          # no need to worry about this in our training set
df.drop(['id'],1,inplace=True)
full_data=df.astype(float).values  #.tolist() #full_data contains row matrix of attributes
random.shuffle(full_data)
X = full_data[:,:-1]   #[[],[],[]]
Y = full_data[:,-1]    #[]



Xnorm=feature_normalization(X)  #normalized features for full data
k_optimal=cross_validation(X,Y)

accuracies=[]
test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))] #first 80 percent by leaving 20 percent last data
test_data=full_data[-int(test_size*len(full_data)):] #last 20 percent of data

for i in train_data:
  train_set[i[-1]].append(i[:-1]) #select last element which is class and append value to the one with this class
      
for i in test_data:
  test_set[i[-1]].append(i[:-1])    # fill disctionary same as top for

correct=0
total=0
for group in test_set:
    for data in test_set[group]: #data is all features.
        vote=model_knn(train_set,data,k_optimal)
        if group==vote:
            correct+=1
        #else:
            #print (confidence)
            
        total+=1
        
#print ('Accuracy:',correct/total)
accuracies.append(correct/total)   #outside for loop so for entire thing our model accuracy is what we are seeing
  
print (sum(accuracies)/len(accuracies))
  
