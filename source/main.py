# Import libraries
"""Packages"""
from introduction.intro import *

""" Main """
import pandas as pd

'''data'''
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('../data/weatherAUS.csv', sep=",")
dataX, dataY = main_introduction(df_train)


# Xtrain, Ytrain, Xtest, Ytest = train_test_split(dataX, dataY, test_size=0.33, random_state=2018,
#                                                 stratify=dataY)
Xtrain, Ytrain, Xtest, Ytest = train_test_split(dataX, dataY)

print(dataX.shape)
print(dataY.shape)
print(Xtrain.shape)
print(Ytrain.shape)

# for i in range(1):
    # test_number_of_features(Xtrain, Ytrain, i)
