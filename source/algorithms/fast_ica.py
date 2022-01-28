import pandas as pd
import numpy as np

from sklearn.decomposition import FastICA
from source.utils.anomaly_scores import anomaly_scores
from source.utils.plot_results import plot_results
from source.utils.scatterPlot import scatterPlot


def fast_ica(X_train, y_train, n_components):
    algorithm = 'parallel'
    whiten = True
    max_iter = 200
    random_state = 2018

    fast_ICA = FastICA(n_components=n_components,
                       algorithm=algorithm, whiten=whiten,
                       max_iter=max_iter,
                       random_state=random_state)
    X_train_fast_ICA = fast_ICA.fit_transform(X_train)
    X_train_fast_ICA = pd.DataFrame(data=X_train_fast_ICA,
                                    index=X_train.index)
    X_train_fast_ICA_inverse = fast_ICA.inverse_transform(X_train_fast_ICA)
    X_train_fast_ICA_inverse = pd.DataFrame(data=X_train_fast_ICA_inverse, index=X_train.index)
    scatterPlot(X_train_fast_ICA, y_train, "Independent Component Analysis")
    anomaly_scores_fast_ICA = anomaly_scores(X_train, X_train_fast_ICA_inverse)
    plot_results(y_train, anomaly_scores_fast_ICA, True)

