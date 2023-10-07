import category_encoders as ce

def create_encoder(df):
    """
    Create a binary encoder for the given dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe to be encoded.

    Returns:
    encoder (category_encoders.BinaryEncoder): The fitted binary encoder.
    """

    cols = ['ground_floor_type', 'land_surface_condition', 'foundation_type', 'roof_type', 
            'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status', 
            'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    encoder = ce.BinaryEncoder(cols=cols)
    encoder.fit(df)

    return encoder


def encode(df, encoder):
    """
    Binary encode the given dataframe using a pre-fitted encoder.

    Parameters:
    df (pandas.DataFrame): The input dataframe to be encoded.
    encoder (category_encoders.BinaryEncoder): The pre-fitted binary encoder.

    Returns:
    encoded_df (pandas.DataFrame): The encoded dataframe.
    """

    encoded_df = encoder.transform(df)
    return encoded_df
