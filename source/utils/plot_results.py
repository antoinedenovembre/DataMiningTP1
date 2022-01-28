import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve


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
