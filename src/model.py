from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def neural_network(geo1_shape, geo2_shape, geo3_shape):
    inp = Input((geo3_shape,))
    intermediate_layer = Dense(16, name="intermediate")(inp)
    output2 = Dense(geo2_shape, activation='sigmoid')(intermediate_layer)
    output1 = Dense(geo1_shape, activation='sigmoid')(intermediate_layer)

    model = Model(inp, [output2, output1])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def random_forest(train_values):
    # Create a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

    # Fit the model to predict the damage_grade_0 and damage_grade_1 columns of the train data set
    rf.fit(train_values.drop(['damage_grade_0', 'damage_grade_1'], axis=1), train_values[['damage_grade_0', 'damage_grade_1']])
    return rf