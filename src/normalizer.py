from sklearn.preprocessing import StandardScaler

columns = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']

def normalize(df_train, df_test, col=columns):

    for col in columns:
        scaler = StandardScaler()
        scaler.fit(df_train[col].values.reshape(-1,1))
        df_train[col] = scaler.transform(df_train[col].values.reshape(-1,1))
        df_test[col] = scaler.transform(df_test[col].values.reshape(-1,1))

    return df_train, df_test