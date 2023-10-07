import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, X_val, y_val): 
    """
    Evaluate the performance of a model on validation data.

    Parameters:
    model (object): The trained model object.
    X_val (array-like): The feature matrix of the validation data.
    y_val (array-like): The target array of the validation data.

    Returns:
    accuracy (float): The accuracy score of the model on the validation data.
    micro_f1 (float): The micro F1 score of the model on the validation data.
    """
    # Set y_val to start from 0 because softmax likes it that way
    y_val = y_val - 1

    # Predict on the validation data (X_val)
    y_pred = model.predict(X_val)

    # Calculate the accuracy
    accuracy = accuracy_score(y_val, y_pred)

    # Calculate the micro F1 score
    micro_f1 = f1_score(y_val, y_pred, average='micro')

    return accuracy, micro_f1
