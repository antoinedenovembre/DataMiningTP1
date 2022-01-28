import numpy as np
import pandas as pd


def anomaly_scores(original_DF, reduced_DF):
    loss = np.sum((np.array(original_DF) - np.array(reduced_DF)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=original_DF.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    return loss
