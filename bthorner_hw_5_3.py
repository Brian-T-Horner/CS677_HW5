"""
Brian Horner
CS 677 - Summer 2
Date: 8/10/2021
Week 5 Homework Question 3
This program uses A Decision Tree Classifier to predict if a fetus
is healthy or not from histograms using the following features:
LB - FHR baseline (beats per minute)
ALTV - % of time with abnormal long variability
Min - Minimum of FHR histogram
Mean - Histogram Mean
"""

# Imports
from bthoner_hw_5_1 import Y_ctg_data, X_ctg_data
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score


"""Decision Tree Classifier"""

X_train, X_test, Y_train, Y_test = train_test_split(X_ctg_data, Y_ctg_data,
                                                    test_size=.5,
                                                    random_state=33)

d_tree = tree.DecisionTreeClassifier(criterion='entropy')

# Fitting model and predicting
d_tree = d_tree.fit(X_train, Y_train)
y_predict = d_tree.predict(X_test)

# Computer decision trees accuracy and confusion matrix
tree_acc = round(accuracy_score(Y_test, y_predict), 2)
tree_conf = confusion_matrix(Y_test, y_predict)

# Computing TP, TN, FP, FN, TPR and TNR
tree_tp = tree_conf[0][0]; tree_fp = tree_conf[1][0]
tree_tn = tree_conf[1][1]; tree_fn = tree_conf[0][1]

tree_tpr = round(tree_tp/(tree_tp+tree_fn), 2)
tree_tnr = round(tree_tn/(tree_tn+tree_fp), 2)

# Adding computations to list for table printing
tree_list = ['Decision Tree', tree_tp, tree_fp, tree_tn,
             tree_fn, tree_acc, tree_tpr, tree_tnr
             ]

if __name__ == "__main__":
    # Print statements under here to avoid printing with bthorner_hw_5_5
    print(f"The Decision Tree Model's Accuracy Score is {tree_acc}")
    print(f"The confusion matrix for the Decision Tree Model is...")
    print(tree_conf)
