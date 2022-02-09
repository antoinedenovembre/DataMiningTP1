import pandas as pd
import numpy as np
from sklearn.decomposition import SparsePCA
from source.utils.anomaly_scores import anomaly_scores
from source.utils.custom_inverse_transform import custom_inverse_transform
from source.utils.plot_results import plot_results
from source.utils.scatterPlot import scatterPlot


def sparse_pca(X_train,y_train,n_components):
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
    cutoff = 350
    preds_Top = preds[:cutoff]
    print("Precision: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              cutoff, 2))
    print("Recall: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              y_train.sum(), 2))
    print("NoRainTomorrow days Caught out of 350 Cases with Sparse PCA:", preds_Top.trueLabel.sum())

    return sparse_PCA