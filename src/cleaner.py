import pandas as pd


def clean(train_values, train_labels):
    """
    Clean the given train by performing various operations.

    Args:
        train_values (DataFrame): The training data.
        train_labels (DataFrame): The training labels.

    Returns:
        tuple: A tuple containing the cleaned train and test data.
    """

    # Clean train data
    train_merged = merge_data(train_values, train_labels)

    # Drop building_id column
    train_merged = drop_building_id_column(train_merged)

    return (train_merged)


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


def drop_building_id_column(data):
    """
    Drop the building_id column from the given data.

    Args:
        data (DataFrame): The data to drop the building_id column from.

    Returns:
        DataFrame: The data with the building_id column dropped.
    """
    return data.drop('building_id', axis=1)



