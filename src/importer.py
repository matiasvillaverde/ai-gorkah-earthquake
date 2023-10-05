import os
import pandas as pd


def import_data(directory="./Data"):
    """
    Import the data from CSV files and return them as tuples.

    Args:
        directory (str): The directory path where the CSV files are located.

    Returns:
        tuple: A tuple containing the train_x, train_y, and test_x dataframes.
    """
    # File paths
    train_x_path = os.path.join(directory, "train_values.csv")
    train_y_path = os.path.join(directory, "train_labels.csv")
    test_x_path = os.path.join(directory, "test_values.csv")

    # Read the data from the CSV files
    train_x, train_y, test_x = [
        pd.read_csv(file_path) for file_path in [
            train_x_path, train_y_path, test_x_path]]

    return train_x, train_y, test_x


def main():
    # Import the data
    train_x, train_y, test_x = import_data()

    # Print the data
    print(train_x)
    print(train_y)
    print(test_x)


if __name__ == "__main__":
    main()
