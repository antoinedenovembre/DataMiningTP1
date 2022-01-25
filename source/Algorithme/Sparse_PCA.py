from sklearn.decomposition import SparsePCA

import numpy as np
import pandas as pd

from source.anomaly_scores import anomaly_scores
from source.custom_inverse_transform import custom_inverse_transform
from source.plot_results import plot_results
from source.scatterPlot import scatterPlot


def Sparse_PCA(X_train,y_train,n_components):
    alpha = 0.0001
    random_state = 2018
    n_jobs = -1
    sparse_PCA = SparsePCA(n_components=n_components,
                           alpha=alpha, random_state=random_state,
                           n_jobs=n_jobs)
    sparse_PCA.fit(X_train.loc[:, :])
    X_train_sparse_PCA = sparse_PCA.transform(X_train)
    X_train_sparse_PCA = pd.DataFrame(data=X_train_sparse_PCA,
                                      index=X_train.index)
    scatterPlot(X_train_sparse_PCA, y_train, "Sparse PCA")

    X_train_sparse_PCA_inverse = custom_inverse_transform(X_train,X_train_sparse_PCA,sparse_PCA)
    anomaly_scores_sparse_PCA = \
        anomaly_scores(X_train, X_train_sparse_PCA_inverse)
    preds = plot_results(y_train, anomaly_scores_sparse_PCA, True)