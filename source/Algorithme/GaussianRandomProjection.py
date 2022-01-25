import pandas as pd
import numpy as np

from sklearn.random_projection import GaussianRandomProjection

from source.anomaly_scores import anomaly_scores
from source.custom_inverse_transform import custom_inverse_transform
from source.plot_results import plot_results
from source.scatterPlot import scatterPlot


def Projection_gaussienne_aleatoire(X_train, y_train, n_components):

    eps = None
    random_state = 2018
    GRP = GaussianRandomProjection(n_components=n_components,
                                   eps=eps, random_state=random_state)
    X_train_GRP = GRP.fit_transform(X_train)
    X_train_GRP = pd.DataFrame(data=X_train_GRP, index=X_train.index)

    X_train_GRP_inverse = custom_inverse_transform(X_train,X_train_GRP,GRP)

    scatterPlot(X_train_GRP, y_train, "Gaussian Random Projection")
    anomaly_scores_PCA = anomaly_scores(X_train, X_train_GRP_inverse)
    preds = plot_results(y_train, anomaly_scores_PCA, True)