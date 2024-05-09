import numpy as np
import pandas as pd
import json

# Initialize the dataset and create a data frame
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)


class Node:
    # Node class for creating the Decision Tree

    def __init__(self):
        self.children = []
        self.value = ''
        self.name = ''
        self.by = ''

    # Add a child to the root
    def add_child(self, node):
        self.children.append(node)

    # Convert the tree to Dict
    def to_dict(root_node):
        if root_node.value != '':
            return root_node.value

        else:
            results = []
            for child in root_node.children:
                value = Node.to_dict(child)
                results.append({child.by: value})
            return {root_node.name: results}

    # Convert the tree to JSON
    def to_JSON(root_node):
        return json.dumps(Node.to_dict(root_node), indent=2)


def entropy(data):
    # Calculate the entropy of a dataset

    n_row, n_col = data.shape
    groups = data.groupby(data.columns[n_col-1])
    row_names = groups.first().index
    H = 0

    for row in row_names:
        group = groups.get_group(row)
        p = len(group) / n_row
        h = p * np.log2(p)
        H += h

    return -H


def column_entropy(data, colName):
    # Calculate the entropy of a dataset given a column

    groups = data.groupby(colName)
    row_names = groups.first().index
    H = 0

    for row in row_names:
        group = groups.get_group(row)
        p = len(group) / len(data)
        h = p * entropy(group)
        H += h

    return H


def info_gains(data):
    # Calculate the information gains of all columns in a dataset

    colNames = data.columns
    _, n_col = data.shape
    H = entropy(data)
    info_gains = []

    for col in colNames:
        if col != colNames[n_col-1]:
            gain = H - column_entropy(data, col)
            info_gains.append(gain)

    return info_gains


def decision_tree(data, root_node):
    # Create a decision tree

    n_row, n_col = data.shape

    # If the dataset is empty, return 'Unclassified'.
    if n_row == 0 or n_col == 0:
        root_node.value = 'Unclassified'
        return root_node

    # If there is only one column, return the element occurred most the often.
    if n_col == 1:
        unique_values = pd.unique(data)
        n_max = 0
        max_value = ''

        for val in unique_values:
            n = (data == val).sum()
            if n > n_max:
                n_max = n
                max_value = val

        root_node.value = max_value

        return root_node

    col_names = data.columns

    # If the data set is pure, return the value.
    if entropy(data) == 0:
        root_node.value = data[col_names[n_col-1]][data.index[0]]
        return root_node

    gains = info_gains(data)
    index = -1
    max_value = -1

    for i in range(n_col-1):
        if max_value < gains[i]:
            max_value = gains[i]
            index = i

    root_node.name = col_names[index]
    groups = data.groupby(root_node.name)
    row_names = groups.first().index

    for row in row_names:
        group = groups.get_group(row)
        child = Node()
        child = decision_tree(group, child)
        child.by = row
        root_node.add_child(child)

    return root_node


# Create a decision tree of the dataset
root_node = Node()
root_node = decision_tree(df, root_node)
print("\nDecision Tree:")
print(root_node.to_JSON())
