import category_encoders as ce

    
def create_encoder(df):
    cols = ['ground_floor_type', 'land_surface_condition', 'foundation_type', 'roof_type', 
        'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status', 
    'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    
    encoder = ce.BinaryEncoder(cols=cols)
    encoder.fit(df)

    return encoder


def encode(df, encoder):
    """
    Binary encode the given dataframe using a pre-fitted encoder.
    """

    return encoder.transform(df)



