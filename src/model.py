
def XGBoost(X_train, y_train):
    """
    Create a XGBoost classifier and fit it to the train data.

    Args:
        training data set

    Returns:
        classifier
    """

    import xgboost as xgb

    #change categories of y to start from 0 bc softmax likes it that way
    y_train = y_train - 1

    # Define the hyperparameters
    hyperparameters = {
        'n_estimators': 535,
        'max_depth': 10,
        'learning_rate': 0.051232294238614126,
        'subsample': 0.6796645277288101,
        'colsample_bytree': 0.7886065868653529,
        'objective': 'multi:softmax',  # Specify multi-class classification objective
        'num_class': 3  # Number of classes in your multi-class problem
    }

    # Create an XGBoost classifier with the specified hyperparameters
    classifier = xgb.XGBClassifier(**hyperparameters)

    # Fit the classifier to the training data (X_train, y_train)
    classifier.fit(X_train, y_train)

    return classifier