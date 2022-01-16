from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def datapresentation(df_train):
    df_train['RainTomorrow'] = le.fit_transform(df_train['RainTomorrow'])
    df_train['RainToday'] = le.fit_transform(df_train['RainToday'])
    print(df_train.head())
    print(df_train.describe())
    print(df_train.columns)
    print(df_train.shape[0])
    print(df_train.info())
    print(df_train['RainTomorrow'].sum())
