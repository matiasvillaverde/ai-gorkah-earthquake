import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, X_val, y_val): 
    #set y_val to start from 0 bc softmax likes it that way
    y_val = y_val - 1

    # Predict on the test data (X_test)
    y_pred = model.predict(X_val)

    # Calculate the accuracy
    accuracy = accuracy_score(y_val, y_pred)

    # Calculate the micro F1 score
    micro_f1 = f1_score(y_val, y_pred, average='micro')

    return accuracy, micro_f1

