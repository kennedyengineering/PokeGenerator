# PokeGenerator Project
# Contains autoencoder model
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Conv2DTranspose,
    Flatten,
    Reshape,
    Dense,
    LeakyReLU,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def build_encoder(input_shape=(128, 128, 3), kernel_shape=(3, 3)):
    """Build the encoder architecture"""

    return Sequential(
        [
            Input(shape=input_shape),
            Conv2D(16, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(16, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(16, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(32, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(32, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(32, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(64, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(64, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(64, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(128, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(128, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(128, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(256, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(256, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(256, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(512, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(512, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(512, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Flatten(),
            Dense(8192),
            LeakyReLU(0.1),
            Dense(8192),
            LeakyReLU(0.1),
            Dense(6144),
            LeakyReLU(0.1),
            Dense(6144),
            LeakyReLU(0.1),
            Dense(4096),
            LeakyReLU(0.1),
            Dense(4096, activation=None),
            # Dense(8192, activation="swish"),
            # Dense(8192, activation="swish"),
            # Dense(1024, activation="tanh"),
        ],
        name="encoder",
    )


def build_decoder(encoder, kernel_shape=(5, 5)):
    """Bulid the decoder architecture"""

    return Sequential(
        [
            Input(shape=encoder.layers[-1].output_shape[1:]),
            # Dense(8192, activation="swish"),
            # Dense(8192, activation="swish"),
            Dense(4096),
            LeakyReLU(0.1),
            Dense(4096),
            LeakyReLU(0.1),
            Dense(6144),
            LeakyReLU(0.1),
            Dense(6144),
            LeakyReLU(0.1),
            Dense(8192),
            LeakyReLU(0.1),
            Dense(8192),
            LeakyReLU(0.1),
            Reshape(target_shape=encoder.layers[-14].output_shape[1:]),
            Conv2D(512, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(512, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(512, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            UpSampling2D((2, 2)),
            Conv2D(256, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(256, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(256, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            UpSampling2D((2, 2)),
            Conv2D(128, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(128, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(128, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            UpSampling2D((2, 2)),
            Conv2D(64, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(64, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(64, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            UpSampling2D((2, 2)),
            Conv2D(32, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(32, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(32, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            UpSampling2D((2, 2)),
            Conv2D(16, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(16, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(16, kernel_shape, padding="same"),
            LeakyReLU(0.1),
            Conv2D(3, kernel_shape, activation=None, padding="same"),
        ],
        name="decoder",
    )


def build_autoencoder(encoder, decoder):
    """Build the autoencoder architecture"""
    inputs = Input((128, 128, 3), name="autoencoder_input")
    embedding = encoder(inputs)
    reconstruction = decoder(tf.math.sigmoid(embedding))
    return Model(inputs=inputs, outputs=reconstruction, name="autoencoder")


def build_model():
    """Return a trainable model and heads"""
    encoder = build_encoder()
    decoder = build_decoder(encoder)
    autoencoder = build_autoencoder(encoder, decoder)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    autoencoder.compile(Adam(1e-4), loss=loss)
    return autoencoder, encoder, decoder


if __name__ == "__main__":
    pass
