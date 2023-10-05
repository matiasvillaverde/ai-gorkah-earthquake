import numpy as np
import pandas as pd

def encode(train_values, test_values):

    # Encode the data
    train_values = binary_encode(train_values)
    test_values = binary_encode(test_values)

    # Add damage_grade_1 column to test_values because it is not present in the test data
    test_values["damage_grade_1"] = 0

    return (train_values, test_values)

def binary_encode(df):
    '''
    This function takes a dataframe and returns a binary encoded dataframe
    '''
    import category_encoders as ce
    cols =['damage_grade']
    encoder = ce.BinaryEncoder(cols=cols)
    binary_encoded = encoder.fit_transform(df)
    return binary_encoded