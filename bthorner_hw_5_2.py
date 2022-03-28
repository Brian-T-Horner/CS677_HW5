"""
Brian Horner
CS 677 - Summer 2
Date: 8/10/2021
Week 5 Homework Question 2
This program uses Gaussian Naive Bayesian Classifier to predict if a fetus
is healthy or not from histograms using the following features:
LB - FHR baseline (beats per minute)
ALTV - % of time with abnormal long variability
Min - Minimum of FHR histogram
Mean - Histogram Mean
"""

# Imports
from bthoner_hw_5_1 import Y_ctg_data, X_ctg_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


"""Naive Bayesian NB Classifier"""

X_train, X_test, Y_train, Y_test = train_test_split(X_ctg_data, Y_ctg_data,
                                                    test_size=.5,
                                                    random_state=33)

g_naive_bayes = GaussianNB()

# Fitting model. Used ravel instead of reshape on NP Array to avoid warning
gnb = g_naive_bayes.fit(X_train, Y_train.ravel())
y_predict = gnb.predict(X_test)

# Computing accuracy and Confusion Matrix
gnb_acc = round(accuracy_score(Y_test, y_predict), 2)
gnb_conf = confusion_matrix(Y_test, y_predict)

# Computing TP, TN, FP, FN, TPR and TNR
gnb_tp = gnb_conf[0][0]; gnb_fp = gnb_conf[1][0]
gnb_tn = gnb_conf[1][1]; gnb_fn = gnb_conf[0][1]

gnb_tpr = round(gnb_tp/(gnb_tp+gnb_fn), 2)
gnb_tnr = round(gnb_tn/(gnb_tn+gnb_fp), 2)

# Adding computations to list for table printing
gnb_list = ['Naive Bayesian', gnb_tp, gnb_fp,
            gnb_tn, gnb_fn, gnb_acc, gnb_tpr, gnb_tnr
            ]

if __name__ == "__main__":
    # Print statements here to avoid printing with bthorner_hw_5_5
    print(f"Gaussian Naive Bayesian Accuracy Score is {gnb_acc}")
    print(f"The confusion matrix for Naive Bayesian is...")
    print(gnb_conf)
