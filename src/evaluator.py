from sklearn.metrics import f1_score

# Function that takes a model and prints the model summary
def print_model_summary(model, test_data):

    # Predict the damage_grade_0 and damage_grade_1 columns of the test data set
    predictions = model.predict(test_data.drop(['damage_grade_0', 'damage_grade_1'], axis=1))

    # Measure the F Score of the model
    print(f1_score(test_data[['damage_grade_0', 'damage_grade_1']], predictions, average='weighted'))

    predictions