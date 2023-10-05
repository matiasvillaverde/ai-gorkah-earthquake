import pandas as pd

def clean(train_values, train_labels, test_values):
    """
    Clean the given train and test data by performing various operations.

    Args:
        train_values (DataFrame): The training data.
        train_labels (DataFrame): The training labels.
        test_values (DataFrame): The test data.

    Returns:
        tuple: A tuple containing the cleaned train and test data.
    """

    # Clean train data
    train_values = merge_data(train_values, train_labels)
    train_values = remove_geo_columns(train_values)
    train_values = drop_building_id_column(train_values)

    # Clean test data
    test_values = remove_geo_columns(test_values)
    test_values = drop_building_id_column(test_values)

    # Add damage_grade column with default value 0 because there is no damage_grade column in the test data
    test_values = add_damage_grade_column(test_values)

    return (train_values, test_values)

def merge_data(train_values, train_labels):
    """
    Merge data and labels using the building_id column.

    Args:
        train_values (DataFrame): The training data.
        train_labels (DataFrame): The training labels.

    Returns:
        DataFrame: The merged data and labels.
    """
    return pd.merge(train_values, train_labels, on='building_id')

def remove_geo_columns(data):
    """
    Remove geo columns from the given data.

    Args:
        data (DataFrame): The data to remove geo columns from.

    Returns:
        DataFrame: The data with geo columns removed.
    """
    geo_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    return data.drop(geo_columns, axis=1)

def drop_building_id_column(data):
    """
    Drop the building_id column from the given data.

    Args:
        data (DataFrame): The data to drop the building_id column from.

    Returns:
        DataFrame: The data with the building_id column dropped.
    """
    return data.drop('building_id', axis=1)

def add_damage_grade_column(data):
    """
    Add a damage_grade column with default value 0 to the given data.

    Args:
        data (DataFrame): The data to add the damage_grade column to.

    Returns:
        DataFrame: The data with the damage_grade column added.
    """
    data['damage_grade'] = 0
    return data
