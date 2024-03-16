# PokeGenerator Project
# Contains autoencoder model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def build_encoder(input_shape=(128, 128, 3), kernel_shape=(5, 5)):
    """Build the encoder architecture"""

    return Sequential(
        [
            Input(shape=input_shape),
            Conv2D(16, kernel_shape, activation="swish", padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(32, kernel_shape, activation="swish", padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(64, kernel_shape, activation="swish", padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(128, kernel_shape, activation="tanh", padding="same"),
            MaxPooling2D((2, 2), padding="same"),
        ],
        name="encoder",
    )


def build_decoder(encoder, kernel_shape=(5, 5)):
    """Bulid the decoder architecture"""

    return Sequential(
        [
            Input(shape=encoder.layers[-1].output_shape[1:]),
            Conv2D(128, kernel_shape, activation="swish", padding="same"),
            UpSampling2D((2, 2)),
            Conv2D(64, kernel_shape, activation="swish", padding="same"),
            UpSampling2D((2, 2)),
            Conv2D(32, kernel_shape, activation="swish", padding="same"),
            UpSampling2D((2, 2)),
            Conv2D(16, kernel_shape, activation="swish", padding="same"),
            UpSampling2D((2, 2)),
            Conv2D(3, kernel_shape, activation="sigmoid", padding="same"),
        ],
        name="decoder",
    )


def build_autoencoder(encoder, decoder):
    """Build the autoencoder architecture"""
    inputs = Input((128, 128, 3), name="autoencoder_input")
    embedding = encoder(inputs)
    reconstruction = decoder(embedding)
    return Model(inputs=inputs, outputs=reconstruction, name="autoencoder")


def build_model():
    """Return a trainable model and heads"""
    encoder = build_encoder()
    decoder = build_decoder(encoder)
    autoencoder = build_autoencoder(encoder, decoder)
    autoencoder.compile(Adam(3e-4), loss="binary_crossentropy")
    return autoencoder, encoder, decoder


if __name__ == "__main__":
    pass
