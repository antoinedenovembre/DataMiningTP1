# Import libraries
"""Packages"""
from introduction.intro import *
from source.Algorithme.Kernel_PCA import Kernel_PCA
from source.Algorithme.fast_ICA import Fast_ICA

""" Main """
import pandas as pd

'''data'''
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('../data/weatherAUS.csv', sep=",")
dataX, dataY = main_introduction(df_train)


Xtrain,Xtest, Ytrain, Ytest = train_test_split(dataX, dataY, test_size=0.33, random_state=2018,
                                                 stratify=dataY)
print(dataX.shape)
print(dataY.shape)
print(Xtrain.shape)
print(Ytrain.shape)
n_components = 15

#PCA_func(Xtrain, Ytrain,n_components)
#Sparse_PCA(Xtrain,Ytrain,n_components)
#miniBatchDict(Xtrain,Ytrain)
#Fast_ICA(Xtrain,Ytrain,n_components)
Kernel_PCA(Xtrain,Ytrain,n_components)
