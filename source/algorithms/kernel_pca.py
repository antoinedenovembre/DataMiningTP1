from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np

from source.utils.anomaly_scores import anomaly_scores
from source.utils.plot_results import plot_results
from source.utils.scatterPlot import scatterPlot


def kernel_pca(X_data, y_data, n_components):
    kernel = 'rbf'
    gamma = None
    fit_inverse_transform = True
    random_state = 2018
    n_jobs = 1

    kernel_PCA = KernelPCA(n_components=n_components, kernel=kernel,
                           gamma=gamma,
                           fit_inverse_transform=fit_inverse_transform,
                           n_jobs=n_jobs, random_state=random_state)
    kernel_PCA.fit(X_data.iloc[:2000])
    X_data_kernel_PCA = kernel_PCA.transform(X_data)
    X_data_kernel_PCA = pd.DataFrame(data=X_data_kernel_PCA,
                                     index=X_data.index)
    X_data_kernel_PCA_inverse = kernel_PCA.inverse_transform(X_data_kernel_PCA)
    X_data_kernel_PCA_inverse = pd.DataFrame(data=X_data_kernel_PCA_inverse,
                                             index=X_data.index)
    scatterPlot(X_data_kernel_PCA, y_data, "Kernel PCA")
    anomaly_scores_PCA = anomaly_scores(X_data, X_data_kernel_PCA_inverse)
    preds = plot_results(y_data, anomaly_scores_PCA, True)
    cutoff = 350
    preds_Top = preds[:cutoff]
    print("Precision: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              cutoff, 2))
    print("Recall: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              y_data.sum(), 2))
    print("NoRainTomorrow days Caught out of 350 Cases with kernel PCA:", preds_Top.trueLabel.sum())
