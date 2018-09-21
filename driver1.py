# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17:16:34 2018

@author: Dell
"""

from DecisionTree import *
import pandas as pd
from sklearn import model_selection

header = ['Age of the patient', 'Spectacle prescription', 'Astigmatic', 'Tear production rate','Class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data', header=None, names=['Age of the patient', 'Spectacle prescription', 'Astigmatic', 'Tear production rate','Class'])
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc) +"%")

## TODO: You have to decide on a pruning strategy
t_pruned = prune_tree(t, [12, 10, 6])

print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t_pruned)
print("Accuracy on train = " + str(acc)+ "%")
