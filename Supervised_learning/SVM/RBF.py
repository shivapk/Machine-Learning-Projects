import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from svmutil import *
import time

names=[i for i in range(1,32)]
df=pd.read_csv("phishing_websites.csv",names=names)
Transform= [2,7,8,14,15,16,26,29]


df2=df.values.tolist()
#print (len(df2))
#print (df2[0])

for i in Transform:
    for j in range(len(df2)):
        if (df2[j][i-1]== -1):
            df2[j][i-1]=[1,0,0]
        elif (df2[j][i-1]==0):
            df2[j][i-1]=[0,1,0]
        else:
            df2[j][i-1]=[0,0,1]
            
Transform1=['c2_','c7_','c8_','c14_','c15_','c16_','c26_','c29_']   
df_new=pd.DataFrame(df2,columns=names)  
#print (df_new[2].apply(pd.Series))
#df_con=pd.concat([df_new[1],df_new.apply(pd.Series)],axis=1)
df_fb = pd.concat([pd.DataFrame(df_new[x].values.tolist()).add_prefix(x) for x in Transform1], axis=1)
df_f = pd.concat([df_fb, df_new.drop(Transform1, axis=1)], axis=1)
#intial columns=31 then removed 8 columns and added 24 colums=31-8+24 = 47

train, test = train_test_split(df_f, test_size=1/3)
full_test=test.values
full_train=train.values
x_train=full_train[:,:-1].tolist() 
y_train=full_train[:,-1].tolist() 
x_test=full_test[:,:-1].tolist() 
y_test=full_test[:,-1].tolist() 
#SVM_TYPE = ['C_SVC', 'NU_SVC', 'ONE_CLASS', 'EPSILON_SVR', 'NU_SVR' ]
#KERNEL_TYPE = ['LINEAR', 'POLY', 'RBF', 'SIGMOID', 'PRECOMPUTED']


def cross_val(y_train,x_train):
    prob  = svm_problem(y_test, x_test)  #t is 0 for linear svm  #set c as hyperparameter
    max=0
    c=g=1
    for c in range(1,20):
        c=str(c)
        for g in range(1,20):
            g=str(g)
            param = svm_parameter('-t 2 -c '+c+' -g '+g+' -v 3'+' -h 0')
            acc = svm_train(prob,param)
            if acc>max:
                max=acc
                best_c = c
                best_g=g
    print ('Best C is %s , Best Gamma is %s, Cross Validation Accuracy is %f'%(best_c,best_g,max))
    return [best_c,best_d]
    

def polynomial_svm(y_test,x_test,b_c,b_g): #gives best p
    prob  = svm_problem(y_test, x_test)  #t is 0 for linear svm  #set c as hyperparameter
    param = svm_parameter('-t 2 -c '+b_c+' -d '+b_g+' -h 0')
    model = svm_train(prob,param)  #v is cross validation 3 fold.
    p_labels, p_acc, p_vals = svm_predict(y_test,x_test, model) #replace m with cv_acc if cross validation is done
print ('this is RBF')
start=time.time()    
b_c,b_g=cross_val(y_train,x_train)
end=time.time()
print ('Average Training time = %f seconds'%((end-start)/400))

polynomial_svm(y_test,x_test,b_c,b_g)



