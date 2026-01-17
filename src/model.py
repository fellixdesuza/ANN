import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(img_size: int = 32) -> tf.keras.Model:
    """Simple CNN for grayscale images (img_size x img_size x 1)."""
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 1)),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
