import numpy as np
import pandas as pd


def custom_inverse_transform(Xtrain,Xtrain_method,method):
    Xtrain_inverse = \
        np.array(Xtrain_method).dot(method.components_) + np.array(Xtrain.mean(axis=0))
    X_train_inverse = \
        pd.DataFrame(data=Xtrain_inverse, index=Xtrain.index)
    return X_train_inverse