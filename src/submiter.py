import pandas as pd

def submit(model, test_values):
    # Load the sample submission file
    # submission = pd.read_csv("../Data/submission_format.csv")

    # # Make a prediction
    # predictions = model.predict(test_values)

    # # Save the prediction in the submission file
    # submission.iloc[:, 1:] = predictions
    # submission.to_csv("submission.csv", index=False)

    print("Submission saved")