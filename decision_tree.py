# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import pandas as pd
from sklearn import tree as sk
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import graphviz

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    partition_dict =dict()
    length = len(x)
    for i in range(0,length):
        if x[i] in partition_dict:
            partition_dict[x[i]].append(i)
        else:
            partition_dict[x[i]] = []
            partition_dict[x[i]].append(i)
        
    return partition_dict
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    
    # INSERT YOUR CODE HERE
    ent = 0
    y = np.array(y)
    s = set(y)
    for i in s:
        numofval = (y==i)
        prob = numofval.sum()/len(y)
        ent = ent + (prob * np.log2(prob))
    ent = ent* (-1)
    return ent
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    x_part = partition(x)
    Entropy1 = entropy(y)
    Entropy2 = 0
    for j in x_part:
        y_i = []
        for i in x_part[j]:
            y_i.append(y[i])
        length=len(x)
        k = x.count(j)/length
        k=k*entropy(y_i)
        Entropy2 = Entropy2 + k
    MI= Entropy1-Entropy2
    return MI
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if attribute_value_pairs == None :
        attribute_value_pairs = []
        xshape = x.shape[1]
        for i in range(0, xshape):
            for j in set(x[:, i]):
                attribute_value_pairs.append((i,j))
    if len(attribute_value_pairs) == 0 or depth == max_depth :
        c = np.bincount(np.array(y))
        value = np.argmax(c)
        return value
    
    elif all(s == y[0] for s in y) :
        return y[0]
    
    else :
        maxv = 0
        for it in attribute_value_pairs:
            t = []
            k = it[0]
            length=len(x)
            for i in range(0, length):
                w = x[i][k]
                if w == it[1]:
                    t.append(1)
                else:
                    t.append(0)
            value = mutual_information(t,y)
            if value >= maxv:
                maxv = value
                find_split = it
        
        w = find_split[1]
        k = find_split[0]
        t = []
        for i in range(0, len(x)):
            t.append(x[i][k])
            
        final =partition(t)[w]
        
        x_true = []
        x_false = []
        y_true = []
        y_false = []
        
        length=len(x)
        for i in range(0, length):
            temp = np.asarray(x[i])
            if i in final:
                x_true.append(temp)
                y_true.append(y[i])
            else:
                x_false.append(temp)
                y_false.append(y[i])
        
        T = attribute_value_pairs.copy()
        F = attribute_value_pairs.copy()
        T.remove(find_split)
        F.remove(find_split)
        
        final_tree={}
        final_tree.update({(find_split[0], find_split[1], True): id3(x_true, y_true, T, depth+1, max_depth)})
        final_tree.update({(find_split[0], find_split[1], False): id3(x_false, y_false, F, depth+1, max_depth)})
        return final_tree
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    try:
        len(tree.keys())
        dict_keys=tree.keys()
        itemlist = list(dict_keys)[0]
    
        if x[itemlist[0]] == itemlist[1]:
            return predict_example(x, tree[(itemlist[0], itemlist[1], True)])
        else:
            return predict_example(x, tree[(itemlist[0], itemlist[1], False)])
        
    except Exception:
        return tree
    

    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    return (1/len(y_true)) * sum(y_true != y_pred)
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here C:/Program Files (x86)/Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + 'C:/Users/shara/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    print("Part a)")
    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    # end of part a of assignment

    print("Part b) Plotting training and testing error curves together for each monk dataset, with tree depth on the x-axis and error on the y-axis. ")
    for i in range(1,4):    
       # Load the training data
       M = np.genfromtxt('./monks-'+str(i)+'.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
       ytrn = M[:, 0]
       Xtrn = M[:, 1:]

       # Load the test data
       M = np.genfromtxt('./monks-'+str(i)+'.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
       ytst = M[:, 0]
       Xtst = M[:, 1:]
       
       trainerror = {}
       testerror = {}
       
       for j in range(1, 11):
           decision_tree = id3(Xtrn, ytrn, max_depth=j)
           trainy_pred = [predict_example(x, decision_tree) for x in Xtrn]
           trn_err = compute_error(ytrn, trainy_pred)
           testy_pred = [predict_example(x, decision_tree) for x in Xtst]
           tst_err = compute_error(ytst, testy_pred)
           
           trainerror[j] = trn_err
           testerror[j] = tst_err
       plt.figure() 
       plt.plot(list(trainerror.keys()), list(trainerror.values()), marker='o', linewidth=2, markersize=8) 
       plt.plot(list(testerror.keys()), list(testerror.values()), marker='s', linewidth=2, markersize=8) 
       plt.xlabel('Tree Depth', fontsize=15) 
       plt.ylabel('Error', fontsize=15) 
       plt.xticks(list(trainerror.keys()), fontsize=18) 
       plt.legend(['Train Error', 'Test Error'], fontsize=16) 
       plt.title("MONKS DATASET-"+str(i))

    #end of part b of assignment
    print("part c)")
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    for i in range(1, 6, 2):
        print("Tree visualisation for depth",i)
        fTree = id3(Xtrn, ytrn, max_depth=i)
        pretty_print(fTree)
        dot_str = to_graphviz(fTree)
        render_dot_file(dot_str, './monks1learn-'+str(i))
        y_pred = [predict_example(x, fTree) for x in Xtst]
        print("Printing the Confusion matrix using id3 alogrithm for monk 1 dataset for depth",i)
        print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                       columns=['Classifier Positives', 'Classifier Negatives'],
                       index=['True Positives', 'True Negatives']
                       ))
         
    #end of part c of assignment
    print("part d)")
    for i in range(1, 6, 2):
        feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
        ftree = sk.DecisionTreeClassifier(criterion='entropy', max_depth=i)
        ftree.fit(Xtrn, ytrn)
        y_pred = ftree.predict(Xtst)
    
        print("Confusion matrix obtained from scikit learn Tree for monk 1 dataset for depth ",i )
        print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                       columns=['Predicted Positives', 'Predicted Negatives'],
                       index=['True Positives', 'True Negatives']
                       ))
        
        
    #end of part d of assignment
    
    M = np.genfromtxt('./balance_train.txt', missing_values=0, skip_header=1, delimiter=',', dtype=int)
    X_train = M[:, 1:]
    Y_train = M[: ,0]

    M = np.genfromtxt('./balance_test.txt', missing_values=0, skip_header=1, delimiter=',', dtype=int)
    X_test = M[:, 1:]
    Y_test = M[: ,0 ]
    print("part e)")
    for i in range(1, 6, 2):
        
        ftree = sk.DecisionTreeClassifier(criterion='entropy', max_depth=i)
        ftree.fit(X_train, Y_train)
        y_pred = ftree.predict(X_test)
        print("Confusion matrix of of Balance Data using scikit learn Tree for balanced dataset for depth ",i )
        print(pd.DataFrame(confusion_matrix(Y_test, y_pred),columns=['Predicted Right', 'Predicted Balanced','Predicted Left'],
                       index=['True Right', 'True Balanced','True Left']
                       ))  
        
    for i in range(1, 6, 2):
        print("Tree visualisation for depth",i)
        fTree = id3(X_train, Y_train, max_depth=i)
        pretty_print(fTree)
        dot_str = to_graphviz(fTree)
        render_dot_file(dot_str, './balancedlearn-'+str(i))
        y_pred = [predict_example(x, fTree) for x in X_test]
        print("Printing the Confusion matrix of Balance Data for depth",i)
        print(pd.DataFrame(confusion_matrix(Y_test, y_pred), 
                       columns=['Predicted Right', 'Predicted Balanced','Predicted Left'],
                       index=['True Right', 'True Balanced','True Left']
                       ))
    
    #end of part e of assignment
    
    print("Report: The Confusion matrices are displayed for depth 1,3 and 5. The graphs for each monk dataset showing error and depth from 1 to 10 are plotted. The id3 and sklearn is used for balanced dataset(own dataset). The confusion matrices for it are shown. The id3 algorithm preditcs better than scikit learn, it can be infered using confusion matrix. Overfitting in decision tree is when it fits all the training samples. This affects the prediction accuracy. To reduce overfitting we can use pruning.")

