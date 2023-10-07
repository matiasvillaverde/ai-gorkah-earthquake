import pandas as pd
from pathlib import Path


def import_data(directory="./Data"):
    """
    Import the data from CSV files and return them as tuples.

    Args:
        directory (str): The directory path where the CSV files are located.

    Returns:
        tuple: A tuple containing the train_x, train_y, and test_x dataframes.
    """
    # File paths
    file_names = ["train_values.csv", "train_labels.csv", "test_values.csv"]
    file_paths = [Path(directory) / file_name for file_name in file_names]

    # Read the data from the CSV files
    data_frames = [pd.read_csv(file_path) for file_path in file_paths]

    # Assign the data frames to variables
    train_x, train_y, test_x = data_frames

    return train_x, train_y, test_x
