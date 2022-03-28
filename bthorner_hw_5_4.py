"""
Brian Horner
CS 677 - Summer 2
Date: 8/10/2021
Week 5 Homework Question 4
This program uses a Random Forest Classifier to predict if a fetus
is healthy or not from histograms using the following features:
LB - FHR baseline (beats per minute)
ALTV - % of time with abnormal long variability
Min - Minimum of FHR histogram
Mean - Histogram Mean
We run through n_estimators 1-10 and depth 1-5 in order to find the best
model. We plot the errors of each model and compute statistics for the best
model.
"""

# Imports
import matplotlib.pyplot as plt
from bthoner_hw_5_1 import Y_ctg_data, X_ctg_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

"""Random Forest Classifier"""

# Setting default values for variables used in iteration
lowest_error = 1
best_model = tuple()
previous_accuracy = 0

# Establishing Lists for scatter plot
n_list = []
d_list = []
error_list = []


# Looping through all options of n_estimator and max_depth
for n in range(1, 11):
    for d in range(1, 6):
        X_train, X_test, Y_train, Y_test = train_test_split(X_ctg_data, Y_ctg_data,
                                                            test_size=.5,
                                                            random_state=33)
        rf_model = RandomForestClassifier(n_estimators=n, max_depth=d,
                                          criterion='entropy')
        # Fitting and predicting with each model
        rf_model.fit(X_train, Y_train.ravel())
        Y_predict = rf_model.predict(X_test)
        # Calculating  models error for comparison
        model_error = np.mean(Y_predict != Y_test)
        # Adding n, d and error values to lists for scatter plots
        n_list.append(n); d_list.append(d)
        error_list.append(round(model_error, 2))

        # Comparing to previous lowest error rate
        if model_error < lowest_error:
            # Assigning model error to lowest error
            lowest_error = model_error
            # Grabbing n and d for the lower error rate
            best_model = (n, d)
        else:
            pass

# Splitting data to avoid problems
X_train, X_test, Y_train, Y_test = train_test_split(X_ctg_data, Y_ctg_data,
                                                    test_size=.5,
                                                    random_state=33)
# Getting best Random Forest model
rf_model = RandomForestClassifier(n_estimators=best_model[0],
                                  max_depth=best_model[1],
                                  criterion='entropy')
# Fitting and predicting with best model
rf_model.fit(X_train, Y_train.ravel())
Y_predict = rf_model.predict(X_test)

# Computing accuracy and Confusion Matrix for the best model
rf_acc = round(accuracy_score(Y_test, Y_predict), 2)
rf_conf = confusion_matrix(Y_test, Y_predict)

# Computing TP, TN, FP, FN, TPR and TNR
rf_tp = rf_conf[0][0]; rf_fp = rf_conf[1][0]
rf_tn = rf_conf[1][1]; rf_fn = rf_conf[0][1]
rf_tpr = round(rf_tp/(rf_tp+rf_fn), 2)
rf_tnr = round(rf_tn/(rf_tn+rf_fp), 2)

# Adding computations to list for table printing
rf_list = ['Random Forest', rf_tp, rf_fp,
           rf_tn, rf_fn, rf_acc, rf_tpr, rf_tnr
           ]


"""Scatter Plot of Models and Error Rates"""
plt.scatter(x=n_list, y=d_list, s=200, c=error_list, cmap='ocean')
plt.ylabel("Max Depth")
plt.xlabel("Number of n_estimators")
plt.title("Plot of Errors of Various Random Forest Models")
plt.colorbar()



if __name__ == "__main__":
    # Statements under here are to avoid running under bthorner_hw_5_5.py

    # Print Statements for model evaluation and the best model
    print(f"Lowest Error is {round(lowest_error, 2)} and the best model has an "
      f"n_estimator value of {best_model[0]} and a max_depth value of "
      f"{best_model[1]}. ")
    print(f"This model has an accuracy of {rf_acc}.")
    print(f"The confusion matrix for this model is:")
    print(rf_conf)

    # Plot show function
    plt.show()
