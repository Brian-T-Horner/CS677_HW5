"""
Brian Horner
CS 677 - Summer 2
Date: 8/10/2021
Week 5 Homework Question 1
This program reads the sheet Raw Data from the excel file CTG.xls. We then
get the desired columns. We assign a value of 2 or 3 in the Column 'NSP' to 0.
We return the desired columns excluding 'NSP' as X_ctg_data and 'NSP' column
as Y_ctg_data.
"""

# Imports
import matplotlib.pyplot as plt
import pandas as pd


# Be sure pip install xlrd in order to use this pd.read_excel command
raw_data = pd.read_excel('CTG.xls', sheet_name='Raw Data')


# Slicing raw_data to get desired columns.
# 1: row slice to get rid of blank row
raw_data = raw_data.loc[1:, ['LB', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                             'Width', 'Min', 'Max', 'Mode', 'Mean',
                             'Median', 'Variance', 'NSP'
                             ]]
# Combining NSP abnormal and suspect under abnormal = 0
raw_data.loc[(raw_data['NSP'] == 2) | (raw_data['NSP'] == 3), 'NSP'] = 0

# Dropping columns with NANs
raw_data.dropna(inplace=True)

# Making a deep copy of specific rows to be used in models
ctg_data = raw_data.loc[1:, ['LB', 'ALTV', 'Min', 'Mean', 'NSP']].copy(
    deep=True)

# Splitting data into X and Y variables
Y_ctg_data = ctg_data[['NSP']].values
X_ctg_data = ctg_data[['LB', 'ALTV', 'Min', 'Mean']].values





