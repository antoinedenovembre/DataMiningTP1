import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve

# TODO make it work

def anomaly_scores(original_DF, reduced_DF):
    loss = np.sum((np.array(original_DF) - np.array(reduced_DF)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=original_DF.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    return loss


def scatterPlot(x_DF, y_DF, algo_name):
    temp_DF = pd.DataFrame(data=x_DF.loc[:, 0:1], index=x_DF.index)
    temp_DF = pd.concat((temp_DF, y_DF), axis=1, join="inner")
    print(x_DF)
    print(y_DF)
    # temp_DF.columns = ["First Vector", "Second Vector", "Label"]
    # sns.lmplot(x="First Vector", y="Second Vector", hue="Label",
    #            data=temp_DF, fit_reg=False)
    # ax = plt.gca()
    # ax.set_title("Separation of Observations using " + algo_name)
    # plt.show()


def plot_results(true_labels, anomaly_scores, return_preds=False):
    preds = pd.concat([true_labels, anomaly_scores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']

    precision, recall, thresholds = precision_recall_curve(preds['trueLabel'], preds['anomalyScore'])

    average_precision = average_precision_score(preds['trueLabel'], preds['anomalyScore'])

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision = ' '{0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['anomalyScore'])

    area_under_ROC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: ''Area under the curve = {0:0.2f}'.format(area_under_ROC))
    plt.legend(loc="lower right")
    plt.show()

    if return_preds:
        return preds


def test_number_of_features(Xtrain, Ytrain, components):
    train_index = range(0, len(Xtrain))
    print(Xtrain)
    print(Ytrain)

    whiten = False
    random_state = 2018

    pca = PCA(n_components=components, whiten=whiten,
              random_state=random_state)

    X_train_PCA = pca.fit_transform(Xtrain)
    X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)
    # scatterPlot(X_train_PCA, Ytrain, "PCA")
