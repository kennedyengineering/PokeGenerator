# PokeGenerator Project
# Contains autoencoder model

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def build_encoder(latent_dim):
    """Build the encoder architecture"""
    return Sequential(
        [
            Input(shape=(28, 20), name="encoder_input"),
            Flatten(),
            Dense(64, activation="relu", name="encoder_hidden_1"),
            Dense(32, activation="relu", name="encoder_hidden_2"),
            Dense(latent_dim, activation="tanh", name="encoder_embedding"),
        ],
        name="encoder",
    )


def build_decoder(latent_dim):
    """Bulid the decoder architecture"""
    return Sequential(
        [
            Input(latent_dim, name="decoder_input"),
            Dense(32, activation="relu", name="decoder_hidden_1"),
            Dense(64, activation="relu", name="decoder_hidden_2"),
            Dense(28 * 20, activation="linear", name="decoder_output"),
            Reshape((28, 20)),
        ],
        name="decoder",
    )


def build_autoencoder(encoder, decoder):
    """Build the autoencoder architecture"""
    inputs = Input((28, 20), name="autoencoder_input")
    embedding = encoder(inputs)
    reconstruction = decoder(embedding)
    return Model(inputs=inputs, outputs=reconstruction, name="autoencoder")


def build_model(latent_dim=2):
    """Return a trainable model and heads"""
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    autoencoder = build_autoencoder(encoder, decoder)
    autoencoder.compile(Adam(3e-4), loss="mean_squared_error")
    return autoencoder, encoder, decoder


if __name__ == "__main__":
    pass
