# File that import the data from the csv file to the database

import pandas as pd

def import_data(directory="./Data"):
    # Import the data from the csv file
    # Read the data from the file
    train_x = pd.read_csv(directory + "/test_values.csv")  # Read training data
    train_y = pd.read_csv(directory + "/train_labels.csv")  # Read training labels
    test_x  = pd.read_csv(directory + "/test_values.csv")   # Read test data

    # Return the data
    return (train_x, train_y, test_x)

def main():
    # Import the data
    (train_x, train_y, test_x) = import_data()

    # Print the data
    print(train_x)
    print(train_y)
    print(test_x)


if __name__ == "__main__":
    main()

