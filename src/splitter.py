from sklearn.model_selection import train_test_split


def split(dataframe):
    """
    Splits a given dataframe into predictor variables (X) and target variable (y),
    and then further splits them into training and testing sets.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe containing the data.

    Returns:
    tuple: A tuple containing four elements - X_train, X_val, y_train, y_val.
        X_train (pandas.DataFrame): The training set of predictor variables.
        X_val (pandas.DataFrame): The validation set of predictor variables.
        y_train (pandas.Series): The training set of target variable.
        y_val (pandas.Series): The validation set of target variable.
    """
    # Define the predictor variables (X) and the target variable (y)
    X = dataframe.drop(['damage_grade'], axis=1)
    y = dataframe['damage_grade']

    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
    return X_train, X_val, y_train, y_val
