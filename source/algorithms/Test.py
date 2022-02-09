import pandas as pd
import numpy as np
from source.utils.anomaly_scores import anomaly_scores
from source.utils.custom_inverse_transform import custom_inverse_transform
from source.utils.plot_results import plot_results
from source.utils.scatterPlot import scatterPlot

def Test(Xtest,Ytest,pca):
    X_test_PCA = pca.transform(Xtest)
    X_test_PCA = pd.DataFrame(data=X_test_PCA, index=Xtest.index)
    X_test_PCA_inverse = pca.inverse_transform(X_test_PCA)
    X_test_PCA_inverse = pd.DataFrame(data=X_test_PCA_inverse,
                                      index=Xtest.index)
    scatterPlot(X_test_PCA, Ytest, "PCA")

    anomaly_scores_PCA = anomaly_scores(Xtest, X_test_PCA_inverse)
    preds = plot_results(Ytest, anomaly_scores_PCA, True)
    cutoff = 350
    preds_Top = preds[:cutoff]
    print("Precision: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              cutoff, 2))
    print("Recall: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              Ytest.sum(), 2))
    print("NoRainTomorrow days Caught out of 350 Cases with PCA learning:", preds_Top.trueLabel.sum())

def Test_custom_transform(Xtest,Ytest,pca):
    X_test_PCA = pca.transform(Xtest)
    X_test_PCA = pd.DataFrame(data=X_test_PCA, index=Xtest.index)
    X_test_inverse = custom_inverse_transform(Xtest,X_test_PCA,pca)
    X_test_inverse = pd.DataFrame(data=X_test_inverse,
                                      index=Xtest.index)
    scatterPlot(X_test_PCA, Ytest, "PCA")

    anomaly_scores_PCA = anomaly_scores(Xtest, X_test_inverse)
    preds = plot_results(Ytest, anomaly_scores_PCA, True)
    cutoff = 350
    preds_Top = preds[:cutoff]
    print("Precision: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              cutoff, 2))
    print("Recall: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              Ytest.sum(), 2))
    print("NoRainTomorrow days Caught out of 350 Cases with PCA learning:", preds_Top.trueLabel.sum())