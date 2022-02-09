import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from source.utils.anomaly_scores import anomaly_scores
from source.utils.plot_results import plot_results
from source.utils.scatterPlot import scatterPlot


def pca_func(Xtrain, Ytrain, components):
    train_index = range(0, len(Xtrain))
    print(Xtrain)
    print(Ytrain)

    whiten = False
    random_state = 2018

    pca = PCA(n_components=components, whiten=whiten,
              random_state=random_state)

    X_train_PCA = pca.fit_transform(Xtrain)
    X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)
    scatterPlot(X_train_PCA, Ytrain, "PCA")
    X_train_PCA_inverse = pca.inverse_transform(X_train_PCA)
    X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse,
                                       index=Xtrain.index)
    anomaly_scores_PCA = anomaly_scores(Xtrain, X_train_PCA_inverse)
    preds = plot_results(Ytrain, anomaly_scores_PCA, True)
    cutoff = 350
    preds_Top = preds[:cutoff]
    print("Precision: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              cutoff, 2))
    print("Recall: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              Ytrain.sum(), 2))
    print("NoRainTomorrow days Caught out of 350 Cases with PCA learning:", preds_Top.trueLabel.sum())
    return pca