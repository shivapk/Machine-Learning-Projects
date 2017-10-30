import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import warnings

def get_values(data, attributes, attr):  #attr is the best attribute for splitting

    index = attributes.index(attr)
    values = []

    for entry in data:  #entry is one sample
        if entry[index] not in values:
            values.append(entry[index])

    return values    #values=['hot','cool'] all possible values of best attribute


def get_data(data, attributes, best, val):  #best='temperature', val='cool'

    new_data = [[]]
    index = attributes.index(best)

    for entry in data:  #entry is one sample
        if (entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])    
    return new_data   #[[temp,humidity],[each sample after removing the best column],[each sample],[each sample],[each sample]]


def majorClass(attributes, data, target):   #returns 'rainy' #performs majority voting and returns the class with highest value

    freq = {}  #freq={'rainy':23,'notrainy':17}
    index = attributes.index(target) # last index as class is the last column 

    for tuple in data:   #tuple is one sample
        if (tuple[index] in freq):
            freq[tuple[index]] += 1 
        else:
            freq[tuple[index]] = 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key]>max:
            max = freq[key]
            major = key

    return major
    
def entropy(attributes, data, targetAttr):
    
    freq = {}
   
    dataEntropy = 0.0

    for entry in data:
        if (entry[-1] in freq):
            freq[entry[-1]] += 1.0
        else:
            freq[entry[-1]]  = 1.0

    for freq in freq.values():
      dataEntropy += (-freq/float(len(data))) * math.log((freq/float(len(data))), 2)
   
    return dataEntropy
    
def info_gain(attributes, data, attr, targetAttr):
    freq = {}
    subsetEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if (entry[i] in freq):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0
    for val in freq.keys():
        valProb        = float(freq[val]) / sum(freq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)
   
    return (entropy(attributes, data, targetAttr) - subsetEntropy)
    
def attr_choose(data, attributes, target): #chooses best attribute based on information gain

    best = attributes[0]
    maxGain = 0;

    for attr in attributes[:-1]:  #for each attribute information gain is calculated
        newGain = info_gain(attributes, data, attr, target) 
        if newGain>maxGain:
            maxGain = newGain
            best = attr
    #print (best,maxGain)
    return best
best_fulltree=[]   
def build_tree(data, attributes, target):  #target is the last class column name, data is the current data at that level

    data = data[:]
    vals = [record[attributes.index(target)] for record in data]  #values of class or last column in the new data vals=[+1,-1,+1,-1]
    default = majorClass(attributes, data, target) #majority voting

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals): #if all values are same then 
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)    #gives best attribute
        best_fulltree.append(best)
        tree = {best:{}}
        
    
        for val in get_values(data, attributes, best):  #best has how many possible values -  val in ['hot','cool']
            new_data = get_data(data, attributes, best, val)  #gives new node data.like [[],[],[],[]]
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree                           #{'temperature':{'hot','cold',mild'}}
   
    return tree


class DecisionTree():

    def learn(self, training_set, attributes, target):
        self.tree_full = build_tree(training_set, attributes, target)
    def learn_nodewise(self, training_set, attributes, target, node_count,best_count):
        self.tree_p = Nodewise_acc_tree(training_set, attributes, target,node_count,best_count)
        

class Node():
    value = ""     #'sky'
    children = []  #[cloudy,clear]

    def __init__(self, val, dictionary):
        self.value = val   #value is sky / temperature
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()
            

def Nodewise_acc_tree(data, attributes, target, node_count,best_count): #how many nodes u want to select in the best
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]  #values of class or last column in the new data vals=[+1,-1,+1,-1]
    default = majorClass(attributes, data, target) #majority voting
    if len(best_count)==node_count:
        return default
    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals): #if all values are same then 
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)    #gives best attribute
        best_count.append(best)
        tree = {best:{}}
        for val in get_values(data, attributes, best):  #best has how many possible values -  val in ['hot','cool']
            new_data = get_data(data, attributes, best, val)  #gives new node data.like [[],[],[],[]]
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = Nodewise_acc_tree(new_data, newAttr, target,node_count,best_count)
            tree[best][val] = subtree                           #{'temperature':{'hot','cold',mild'}}
    return tree
    
def run_decision_tree():
    attributes=['clump_thickness','uniformity_cell_size','uniformity_cell_shape','marginal_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','normal_nucleoli','mitoses','class']
    df=pd.read_csv("C:/Users/MBD/Desktop/gre_college_applications/final-documents/TAMU/after-going-to-college/1st-sem/Machine Learning/project-2/hw2_question1.csv",names=attributes)
    target = attributes[-1]
    df_g=df.groupby('class')
    df_2=df_g.get_group(2)
    df_4=df_g.get_group(4)
    df_train=pd.concat([df_2[:296],df_4[:159]],ignore_index=True)
    df_test=pd.concat([df_2[296:],df_4[159:]],ignore_index=True)
    
    training_set=df_train.values.tolist()
    test_set=df_test.values.tolist()
  
    tree = DecisionTree()
    tree.learn(training_set, attributes, target)
    n=[] #for plotting
    accuracy_train=[]
    for node_count in range(1,len(best_fulltree)+1):
        n.append(node_count)
        tree.learn_nodewise( training_set, attributes, target,node_count,best_count=[])
        results = []
        for entry in training_set:  #for each sample go from top root to leaf
            tempDict = tree.tree_p.copy()
            result = ""
            while(isinstance(tempDict, dict)):   #for each sample its result tells the output
                root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])  #root is sky
                tempDict = tempDict[list(tempDict.keys())[0]]  #rest of the dictionary except sky
                index = attributes.index(root.value)
                value = entry[index]
                if(value in tempDict.keys()):  # checking value is either clear or cloudy
                    child = Node(value, tempDict[value])
                    result = tempDict[value]
                    tempDict = tempDict[value]
                else:
                    result = "Null"
                    break
            if result != "Null":
                results.append(result == entry[-1])

        accuracy = float(results.count(True))/float(len(results))
        accuracy_train.append(accuracy)
    accuracy_test=[]
    for node_count in range(1,len(best_fulltree)+1):
        tree.learn_nodewise( training_set, attributes, target,node_count,best_count=[])
        results = []
        for entry in test_set:  #for each sample go from top root to leaf
            tempDict = tree.tree_p.copy()
            result = ""
            while(isinstance(tempDict, dict)):   #for each sample its result tells the output
                root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])  #root is sky
                tempDict = tempDict[list(tempDict.keys())[0]]  #rest of the dictionary except sky
                index = attributes.index(root.value)
                value = entry[index]
                if(value in tempDict.keys()):  # checking value is either clear or cloudy
                    child = Node(value, tempDict[value])
                    result = tempDict[value]
                    tempDict = tempDict[value]
                else:
                    result = "Null"
                    break
            if result != "Null":
                results.append(result == entry[-1])

        accuracy = float(results.count(True))/float(len(results))
        accuracy_test.append(accuracy)
    print (accuracy_test)        
    plt.plot(n,accuracy_train,'r',marker='.',label='Train data')
    plt.plot(n,accuracy_test,'b',marker='.',label='Test data')
    plt.legend()
    
    plt.grid()
    plt.xlabel('Size of Tree(Number of Nodes)-------------->')
    plt.ylabel('Accuracy--------------------->')
    plt.title('Splitting Criteria: Entropy')
    plt.show()
           

if __name__ == "__main__":
   
    run_decision_tree()
    
    
