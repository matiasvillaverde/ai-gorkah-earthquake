import pandas as pd


def predict(trained_model, new_data):
    """
    Make predictions on a new dataset using a trained XGBoost classifier.

    Args:
        trained_model (XGBoost classifier): The trained XGBoost classifier.
        new_data (DataFrame): The new dataset for prediction.

    Returns:
        DataFrame: A DataFrame with 'building_id' and 'predictions'.

    Example:
        >>> trained_model = XGBClassifier()
        >>> new_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> predict(trained_model, new_data)
           building_id  damage_grade
        0             1             2
        1             2             3
        2             3             1
    """

    # Create a copy of the new_data DataFrame to avoid modifying the original
    # DataFrame
    new_data_copy = new_data.copy()

    # Remove the 'building_id' column if present in the new data (assuming
    # it's not needed for prediction)
    if 'building_id' in new_data_copy.columns:
        new_data_copy = new_data_copy.drop(['building_id'], axis=1)

    # Make predictions on the new data
    predictions = trained_model.predict(new_data_copy)

    # Add 1 to the predictions to revert to the original damage_grade values
    predictions = predictions + 1

    # Create a DataFrame with 'building_id' and 'predictions'
    predictions_df = pd.DataFrame(
        {'building_id': new_data['building_id'], 'damage_grade': predictions})

    return predictions_df
