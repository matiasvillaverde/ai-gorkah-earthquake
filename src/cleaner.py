import pandas as pd

def clean(train_values, train_labels, test_values):

    # Merge data and labels using the building_id column
    train_values = pd.merge(train_values, train_labels, on='building_id')

    # Remove non-numeric columns from train_values
    non_numeric_columns = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']

    # Remove Geo Data
    geo_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    # Remove the columns
    train_values = train_values.drop(non_numeric_columns, axis=1)
    train_values = train_values.drop(geo_columns, axis=1)
    test_values = test_values.drop(non_numeric_columns, axis=1)
    test_values = test_values.drop(geo_columns, axis=1)

    # Drop the building_id column
    train_values = train_values.drop('building_id', axis=1)
    test_values = test_values.drop('building_id', axis=1)

    # Add the damage_grade column to the test_values
    test_values['damage_grade'] = 0

    return (train_values, test_values)