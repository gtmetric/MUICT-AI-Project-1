import numpy as np
import pandas as pd
import json

# Initialize dataframe
data =  {
            'Outlook':['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
            'Temperature':['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
            'Humidity':['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
            'Wind':['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
            'PlayTennis':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
        }
df = pd.DataFrame(data)
print("Dataset:")
print(df)

# Node class for creating the Decision Tree
class Node:
    def __init__(self):
        self.childs = []
        self.value = ''
        self.name = ''
        self.by = ''
        
    def addChild(self, node):
        self.childs.append(node)

    def toJSON(root):
        if root.value != '':
            return root.value
        else:
            results = []
            for child in root.childs:
                value = child.toJSON()
                results.append({child.by: value})
            return {root.name: results}
        
# Calculate the entropy of a dataset
def entropy1(data):
    nrow, ncol = data.shape
    groups = data.groupby(data.columns[ncol-1])
    rowNames = groups.first().index
    H = 0

    for row in rowNames:
        group = groups.get_group(row)
        p = len(group) / nrow
        h = p * np.log2(p)
        H += h

    return -H

# Calculate the entropy of a dataset given a column
def entropy2(data, colName):
    groups = data.groupby(colName)
    rowNames = groups.first().index
    H = 0
    
    for row in rowNames:
        group = groups.get_group(row)
        p = len(group) / len(data)
        h = p * entropy1(group)
        H += h

    return H
    
# Calculate the information gains of all columns in a dataset
def infoGains(data):
    colNames = data.columns
    nrow, ncol = data.shape
    H = entropy1(data)
    infoGains = []

    for col in colNames:
        if col != colNames[ncol-1]:
            gain = H - entropy2(data, col)
            infoGains.append(gain)

    return infoGains

# Create a decision tree
def decisionTree(data, root):
    nrow, ncol = data.shape
    if ncol > 1:
        colNames = data.columns

    if nrow == 0 or ncol == 0:
        root.value = 'Unclassified'

    elif ncol == 1:
        uniqueVals = pd.unique(data)
        nmax = 0
        maxVal = ''

        for val in uniqueVals:
            n = (data == val).sum()
            if n > nmax:
                nmax = n
                maxVal = val

        root.value = maxVal

    elif entropy1(data) == 0:
        root.value = data[colNames[ncol-1]][data.index[0]]

    else:
        gains = infoGains(data)
        index = -1
        max = -1

        for i in range(ncol-1):
            if max < gains[i]:
                max = gains[i]
                index = i

        root.name = colNames[index]
        groups = data.groupby(root.name)
        rowNames = groups.first().index

        for row in rowNames:
            group = groups.get_group(row)
            child = Node()
            child = decisionTree(group, child)
            child.by = row
            root.addChild(child)
            
    return root

root = Node()
root = decisionTree(df, root)
print("\nDecision Tree:")
print(json.dumps(root.toJSON(), indent=2))