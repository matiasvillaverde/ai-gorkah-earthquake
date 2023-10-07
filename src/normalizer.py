from sklearn.preprocessing import StandardScaler

columns = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']

def normalize(df_train, df_test, col=columns):
    """
    Normalize the specified columns in the train and test data using StandardScaler.

    Args:
        df_train (DataFrame): The training data.
        df_test (DataFrame): The test data.
        col (list): The columns to normalize.

    Returns:
        tuple: A tuple containing the normalized train and test data.
    """

    scaler = StandardScaler()
    scaler.fit(df_train[col])

    df_train[col] = scaler.transform(df_train[col])
    df_test[col] = scaler.transform(df_test[col])

    return df_train, df_test
