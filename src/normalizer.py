from sklearn.preprocessing import StandardScaler
import numpy as np

columns = [
    'count_floors_pre_eq',
    'age',
    'area_percentage',
    'height_percentage',
    'count_families']


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


def log_transform(df_train, df_test, col=columns):
    """
    Log transform the specified columns in the train and test data.

    Args:
        df_train (DataFrame): The training data.
        df_test (DataFrame): The test data.
        col (list): The columns to log transform.

    Returns:
        tuple: A tuple containing the log transformed train and test data.
    """

    for c in col:
        df_train[c] = np.log1p(df_train[c])
        df_test[c] = np.log1p(df_test[c])

    return df_train, df_test