from sklearn.model_selection import train_test_split

def split (dataframe):
    # Define the predictor variables (X) and the target variable (y)
    X = dataframe.drop(['damage_grade'], axis=1)
    y = dataframe['damage_grade']
    
    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
    return X_train, X_val, y_train, y_val