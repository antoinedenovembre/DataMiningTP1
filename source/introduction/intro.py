from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def data_drop(df_train):
    # Dropping every column we don't need
    df_train = df_train.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Location', 'Date'], axis=1)
    # Dropping every row having NA values
    df_train = df_train.dropna(axis=0)

    return df_train


def data_encode(df_train):
    # Encoding every non-numeric value in the dataset
    le = LabelEncoder()
    df_train['WindGustDir'] = le.fit_transform(df_train['WindGustDir'])
    df_train['WindDir9am'] = le.fit_transform(df_train['WindDir9am'])
    df_train['WindDir3pm'] = le.fit_transform(df_train['WindDir3pm'])
    df_train['RainTomorrow'] = le.fit_transform(df_train['RainTomorrow'])
    df_train['RainToday'] = le.fit_transform(df_train['RainToday'])

    return df_train


def data_rebalance(df_train):
    df_true = df_train[df_train["RainTomorrow"] == 1]
    df_false = df_train[df_train["RainTomorrow"] == 0]
    df_false = df_false.drop(df_false.index[df_true.shape[0]:])
    df_ret = df_true.append(df_false)
    
    return df_ret


def data_correlation(df_train):
    # Viewing the correlations between data columns
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_train.corr(), cmap="Blues")
    plt.show()


def data_presentation(df_train):
    # Printing some information on the dataset
    print('\033[91m' + 'Preview :' + '\033[0m' + '\n', df_train.head())
    print('\033[91m' + '\nNaNs :' + '\033[0m' + '\n', df_train.isnull().sum())
    print('\033[91m' + '\nStats :' + '\033[0m' + '\n', df_train.describe())
    print('\033[91m' + '\nColumns :' + '\033[0m' + '\n', df_train.columns)
    print('\033[91m' + '\nInfos :' + '\033[0m' + '\n', df_train.info())
    print('\033[91m' + '\nNumber of instances :' + '\033[0m' + '\n', df_train.shape[0])
    print('\033[91m' + '\nNumber of positive instances :' + '\033[0m' + '\n', sum(df_train['RainTomorrow'] == 1))
    print('\033[91m' + '\nNumber of negative instances :' + '\033[0m' + '\n', sum(df_train['RainTomorrow'] == 0))


def main_introduction(df_train):
    data_presentation(df_train)
    df_train = data_drop(df_train)
    df_train = data_encode(df_train)
    df_train = data_rebalance(df_train)
    data_presentation(df_train)
    data_correlation(df_train)
    dataX = df_train.copy().drop(['RainTomorrow'], axis=1)
    dataY = df_train['RainTomorrow']
    return dataX, dataY
