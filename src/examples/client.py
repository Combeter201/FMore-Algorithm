import numpy as np
import flwr as fl
import tensorflow as tf

from fmore.node import fmoreClient as fC

quality_vector = np.random.rand(3)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

fl.client.start_numpy_client(
    "localhost:8080",
    client=fC.FMoreClient(quality_vector, (x_train, y_train, x_test, y_test),
                          model))
