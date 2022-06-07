import os
import tensorflow as tf
print("TF:", tf.__version__)

if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="artefacts/logs/", update_freq='epoch', write_graph=True)


    history = model.fit(x_train,
              y_train,
              epochs=3,
              callbacks=[tensorboard_callback],
              verbose=2
              )