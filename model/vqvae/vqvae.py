# vqvae/vqvae.py

from tensorflow import keras
from tensorflow.keras import layers

from .vector_quantizer import VectorQuantizer

def get_encoder(image_shape=(128,128,3), kernel_size=(3,3), latent_dim=16):
    encoder_inputs = keras.Input(shape=image_shape)
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(kernel_size=(3,3), latent_dim=16):
    latent_inputs = keras.Input(shape=get_encoder(kernel_size=kernel_size, latent_dim=latent_dim).output.shape[1:])
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, kernel_size, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

def get_vqvae(image_shape=(128,128,3), kernel_size=(3,3), latent_dim=16, num_embeddings=64):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(image_shape= image_shape,kernel_size=kernel_size, latent_dim=latent_dim)
    decoder = get_decoder(kernel_size=kernel_size, latent_dim=latent_dim)
    inputs = keras.Input(shape=image_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    encoder.summary()
    decoder.summary()
    return keras.Model(inputs, reconstructions, name="vq_vae")

