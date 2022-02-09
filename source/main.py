# Import libraries
"""Packages"""
from introduction.intro import *
from algorithms.gaussian_random_projection import gaussian_random_projection
from algorithms.pca_reduction import *
from algorithms.sparse_pca import *
from algorithms.mini_batch_dict import mini_batch_dict
from algorithms.kernel_pca import kernel_pca
from algorithms.fast_ica import fast_ica
from source.algorithms.Test import *

""" Main """
import pandas as pd

""" data """
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('../data/weatherAUS.csv', sep=",")
dataX, dataY = main_introduction(df_train)

# for i in range (25, 45, 5):
#     size = i / 100
#     Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataX, dataY, test_size=size, random_state=2018,
#                                                     stratify=dataY)
#     n_components = 15
#     pca_func(Xtrain, Ytrain, n_components)
# IDEAL TEST SIZE IS 0.25
Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataX, dataY, test_size=0.25, random_state=2018,
                                                stratify=dataY)
n_components = 15

#PCA
pca = pca_func(Xtrain, Ytrain, n_components)
Test(Xtest, Ytest, pca)

#Sparse PCA
sparse_pca = sparse_pca(Xtrain, Ytrain, n_components)
Test_custom_transform(Xtest, Ytest, sparse_pca)


#mini_batch_dict(Xtrain, Ytrain)

#Fast ICA
fast_ICA = fast_ica(Xtrain, Ytrain, n_components)
Test(Xtest,Ytest,fast_ICA)


#kernel_pca(Xtrain, Ytrain, n_components)
#gaussian_random_projection(Xtrain, Ytrain, n_components)
