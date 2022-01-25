import pandas as pd


from sklearn.decomposition import MiniBatchDictionaryLearning

from source.anomaly_scores import anomaly_scores
from source.custom_inverse_transform import custom_inverse_transform
from source.plot_results import plot_results
from source.scatterPlot import scatterPlot

def miniBatchDict(X_train,y_train):
    n_components = 28
    alpha = 1
    batch_size = 200
    n_iter = 10
    random_state = 2018
    miniBatchDictLearning = MiniBatchDictionaryLearning(
    n_components=n_components, alpha=alpha, batch_size=batch_size,
    n_iter=n_iter, random_state=random_state)
    miniBatchDictLearning.fit(X_train)
    X_train_miniBatchDictLearning = \
    miniBatchDictLearning.fit_transform(X_train)
    X_train_miniBatchDictLearning = \
    pd.DataFrame(data=X_train_miniBatchDictLearning,
    index=X_train.index)
    scatterPlot(X_train_miniBatchDictLearning, y_train,
    "Mini-batch Dictionary Learning")

    X_train_miniBatchDictLearning_inverse = custom_inverse_transform(X_train,X_train_miniBatchDictLearning,miniBatchDictLearning)
    anomaly_scores_miniBatchDictLearning = anomaly_scores(X_train,X_train_miniBatchDictLearning_inverse)
    preds = plot_results(y_train, anomaly_scores_miniBatchDictLearning, True)