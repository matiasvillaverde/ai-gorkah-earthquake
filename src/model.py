def XGBoost(X_train, y_train):
    """
    Create a XGBoost classifier and fit it to the train data.

    Args:
        X_train (array-like): The training set of predictor variables.
        y_train (array-like): The training set of target variable.

    Returns:
        xgb.XGBClassifier: The trained XGBoost classifier.
    """

    import xgboost as xgb

    # Change categories of y to start from 0 because softmax likes it that way
    y_train = y_train - 1

    # Define the hyperparameters
    hyperparameters = {
        'n_estimators': 1000, 
        'max_depth': 10, 
        'learning_rate': 0.03428839361332356, 
        'subsample': 0.7846790551755336, 
        'colsample_bytree': 0.86244395361441,
        'objective': 'multi:softmax',  # Specify multi-class classification objective
        'num_class': 3  # Number of classes in your multi-class problem
    }

    # Create an XGBoost classifier with the specified hyperparameters
    classifier = xgb.XGBClassifier(**hyperparameters)

    # Fit the classifier to the training data (X_train, y_train)
    classifier.fit(X_train, y_train)

    return classifier
