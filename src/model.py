from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

def neural_network(geo1_shape, geo2_shape, geo3_shape):
    """
    Create a neural network model.

    Args:
        geo1_shape (int): The shape of the output layer for geo1.
        geo2_shape (int): The shape of the output layer for geo2.
        geo3_shape (int): The shape of the input layer.

    Returns:
        tensorflow.keras.models.Model: The compiled neural network model.
    """
    inp = Input((geo3_shape,))
    intermediate_layer = Dense(16, name="intermediate")(inp)
    output2 = Dense(geo2_shape, activation='sigmoid')(intermediate_layer)
    output1 = Dense(geo1_shape, activation='sigmoid')(intermediate_layer)

    model = Model(inp, [output2, output1])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def random_forest(train_values):
    """
    Create a random forest classifier and fit it to the train data.

    Args:
        train_values (pandas.DataFrame): The training data.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The fitted random forest classifier.
    """
    # Create a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

    # Fit the model to predict the damage_grade_0 and damage_grade_1 columns
    # of the train data set
    rf.fit(train_values.drop(['damage_grade_0', 'damage_grade_1'],
            axis=1), train_values[['damage_grade_0', 'damage_grade_1']])
    return rf


def XGBoost(train_values):
    """
    Create a XGBoost classifier and fit it to the train data.

    Args:
        training data set

    Returns:
        ?
    """

    data=train_values

    # Define the predictor variables (X) and the target variable (y)
    X = data.drop(['building_id', 'damage_grade'], axis=1)
    y = data['damage_grade']

    #change categories of y to start from 0 bc softmax likes it that way
    y = y - 1

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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