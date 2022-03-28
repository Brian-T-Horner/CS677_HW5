"""
Brian Horner
CS 677 - Summer 2
Date: 8/10/2021
Week 5 Homework Question 5
This program plots the statistics we calculated for the Naive Bayesian,
Decision Tree, and (Best) Random Forest models and prints them in a table for
discussion.
"""

# Imports
from bthorner_hw_5_2 import gnb_list
from bthorner_hw_5_3 import tree_list
from bthorner_hw_5_4 import rf_list


def table_printer():
    """Formats the computations for table printing."""
    # Header list for the top of the table
    header_list = ['-- Model --', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'TPR',
                   'TNR']
    # Adding each models list to the print_list
    print_list = [gnb_list, tree_list, rf_list]
    print("--- Summary of Models and Confusion Matrix Stats. ---\n")
    print_list.insert(0, header_list)
    for index, stuff in enumerate(print_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(15) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0:
            print('-' * len(row))


table_printer()

print("\nWith each run of the Random Forest program the TP and TN "
      "fluctuate wildly. It seems the random forest model has a disposition "
      "to focus on predicting the negative classifiers (Normal Fetus and "
      "always has a high rate of TNR. On the other hand it is pretty terrible at predicting "
      "positive classifiers (abnormal fetus). This could be troublesome in a "
      "real world application.\n")

print("The best model overall seems to be the Decision Tree. It constantly "
      "has an accuracy above 85% and has TPR and TNR rates that are "
      "respectable and close to each other. It is the only model that does "
      "not struggle predicting Positive classifiers.\n")

print("If I were to choose a model to use in a real world application of this "
      "data I would choose the random forest. While the decision tree seems "
      "like the better option, I believe successfully predicting if there is "
      "an abnormal fetus accurately is much more important in order to have "
      "specialists take a closer look at that group to affirm the findings "
      "and give the correct care that is needed.")

