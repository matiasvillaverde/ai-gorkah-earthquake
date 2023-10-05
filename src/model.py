from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def neural_network(geo1_shape, geo2_shape, geo3_shape):
    inp = Input((geo3_shape,))
    intermediate_layer = Dense(16, name="intermediate")(inp)
    output2 = Dense(geo2_shape, activation='sigmoid')(intermediate_layer)
    output1 = Dense(geo1_shape, activation='sigmoid')(intermediate_layer)

    model = Model(inp, [output2, output1])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model