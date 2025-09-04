from tensorflow import keras


def build_model(input_dim):
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model
