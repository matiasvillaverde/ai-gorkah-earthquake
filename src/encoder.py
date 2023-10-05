import numpy as np
import pandas as pd

def encode(train_values, train_labels, test_values):

    # Encode the data
    train_values = binary_encode(train_values)
    test_values = binary_encode(test_values)

    return (train_values, train_labels, test_values)

def binary_encode(df):
    '''
    This function takes a dataframe and returns a binary encoded dataframe
    '''
    import category_encoders as ce
    cols =['land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'legal_ownership_status', 'geo_level_1_id',
       'geo_level_2_id', 'geo_level_3_id']
    encoder = ce.BinaryEncoder(cols=cols)
    binary_encoded = encoder.fit_transform(df)
    return binary_encoded


def dummy_encode(train_x):

    # Dummy encode the combined column
    dummy_encoded_column = np.array(pd.get_dummies(train_x))

    return dummy_encoded_column
