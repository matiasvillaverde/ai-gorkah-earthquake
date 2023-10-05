import numpy as np
import pandas as pd

def encode(train_values, train_labels, test_values):

    # Encode the data

    return (train_values, train_labels, test_values)


def dummy_encode(train_x):

    # Dummy encode the combined column
    dummy_encoded_column = np.array(pd.get_dummies(train_x))

    return dummy_encoded_column