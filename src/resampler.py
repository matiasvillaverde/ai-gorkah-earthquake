from imblearn.over_sampling import SMOTE
import pandas as pd

def resample(train_data):
    # Define the predictor variables (X) and the target variable (y)
    X = train_data.drop(columns=['damage_grade'])  # Exclude 'damage_grade'
    y = train_data['damage_grade']

    # Apply SMOTE to balance the classes
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine the resampled X and y back into one dataset
    resampled_df = pd.concat([X_resampled, y_resampled], axis=1)

    return resampled_df
