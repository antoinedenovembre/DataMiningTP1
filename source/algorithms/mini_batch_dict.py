import pandas as pd
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from source.utils.anomaly_scores import anomaly_scores
from source.utils.custom_inverse_transform import custom_inverse_transform
from source.utils.plot_results import plot_results
from source.utils.scatterPlot import scatterPlot


def mini_batch_dict(X_train, y_train):
    n_components = 28
    alpha = 1
    batch_size = 200
    n_iter = 10
    random_state = 2018

    miniBatchDictLearning = MiniBatchDictionaryLearning(
        n_components=n_components, alpha=alpha, batch_size=batch_size,
        n_iter=n_iter, random_state=random_state)
    miniBatchDictLearning.fit(X_train)
    X_train_miniBatchDictLearning = miniBatchDictLearning.fit_transform(X_train)
    X_train_miniBatchDictLearning = pd.DataFrame(data=X_train_miniBatchDictLearning,
                                                 index=X_train.index)
    scatterPlot(X_train_miniBatchDictLearning, y_train,
                "Mini-batch Dictionary Learning")

    X_train_miniBatchDictLearning_inverse = custom_inverse_transform(X_train, X_train_miniBatchDictLearning,
                                                                     miniBatchDictLearning)
    anomaly_scores_miniBatchDictLearning = anomaly_scores(X_train, X_train_miniBatchDictLearning_inverse)
    preds = plot_results(y_train, anomaly_scores_miniBatchDictLearning, True)
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
    print("NoRainTomorrow days Caught out of 350 Cases with Dictionnary learning:", preds_Top.trueLabel.sum())
