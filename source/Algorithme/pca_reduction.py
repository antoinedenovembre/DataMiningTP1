
import pandas as pd


from sklearn.decomposition import PCA

from source.anomaly_scores import anomaly_scores
from source.plot_results import plot_results
from source.scatterPlot import scatterPlot


def PCA_func(Xtrain, Ytrain, components):
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
    plot_results(Ytrain, anomaly_scores_PCA, True)
