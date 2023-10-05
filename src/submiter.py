import pandas as pd

def submit(model, test_values):

    # Create a data frame with the predictions
    predictions = pd.DataFrame(predictions, columns=['damage_grade_0', 'damage_grade_1'])

    # Add the building_id column to the data frame
    predictions['building_id'] = test_data['building_id']

    # Reorder the columns of the data frame
    predictions = predictions[['building_id', 'damage_grade_0', 'damage_grade_1']]

    # Save the data frame to a csv file
    predictions.to_csv('../Data/predictions.csv', index=False)

    print("Submission saved")

