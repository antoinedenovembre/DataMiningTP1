# Import libraries
"""Packages"""
from sklearn.metrics import accuracy_score

from introduction.intro import *
from algorithms.gaussian_random_projection import gaussian_random_projection
from algorithms.pca_reduction import pca_func
from algorithms.sparse_pca import sparse_pca
from algorithms.mini_batch_dict import mini_batch_dict
from algorithms.kernel_pca import kernel_pca
from algorithms.fast_ica import fast_ica
from source.algorithms.testing import test_custom_transform
from source.algorithms.testing import test
from utils.accuracy import print_accuracy

""" Main """
import pandas as pd

""" data """
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

# kernel_pca(Xtrain, Ytrain, n_components)
# gaussian_random_projection(Xtrain, Ytrain, n_components)
# mini_batch_dict(Xtrain, Ytrain)

# PCA
pca = pca_func(Xtrain, Ytrain, n_components)
test(Xtest, Ytest, pca)

# Sparse PCA
sparse_pca = sparse_pca(Xtrain, Ytrain, n_components)
test_custom_transform(Xtest, Ytest, sparse_pca)

# Fast ICA
fast_ICA = fast_ica(Xtrain, Ytrain, n_components)
test(Xtest, Ytest, fast_ICA)

# accu = []
# accu_range = range(1,100)
# for val in accu_range:
#     tree = DecisionTreeClassifier(max_depth=val, random_state=10)
#     classifier = tree.fit(Xtrain, Ytrain)
#     y_prediction = classifier.predict(Xtest)
#     accu.append(accuracy_score(Ytest, y_prediction))
#
# print_accuracy(accu_range, accu)
# MAX ACCURACY : 0.7956834532374101  is the maximum accuracy, reached for index  8

tree = DecisionTreeClassifier(max_depth=8, random_state=10)
classifier = tree.fit(Xtrain, Ytrain)
y_prediction = classifier.predict(Xtest)
print("MAX ACCURACY IS :", accuracy_score(Ytest, y_prediction))
