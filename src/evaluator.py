import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def print_model_summary(model, test_data):
    """
    Print the model summary and return the predictions.

    Args:
        model: The trained model.
        test_data: The test data.

    Returns:
        ndarray: The predictions made by the model.
    """
    # Predict the damage_grade_0 and damage_grade_1 columns of the test data
    # set
    predictions = model.predict(test_data.drop(
        ['damage_grade_0', 'damage_grade_1'], axis=1))

    # Measure the F Score of the model
    f_score = f1_score(
        test_data[['damage_grade_0', 'damage_grade_1']], predictions, average='weighted')
    print("F-Score: ", f_score)

    return predictions
