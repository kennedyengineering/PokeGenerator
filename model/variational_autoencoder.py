from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Input, LeakyReLU,
    Conv2DTranspose, Reshape, UpSampling2D, Lambda, Activation,
    BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish, tanh
import tensorflow as tf

def build_encoder(input_shape=(128, 128, 3), kernel_size=(5, 5), strides=(2,2), latent_dim=256):
    model = tf.keras.Sequential([
        Input(shape=input_shape, name='encoder_input'),
        Conv2D(32, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2D(64, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2D(128, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2D(256, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2D(512, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2D(1024, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Flatten(),
        Dense(latent_dim, activation=tanh)
    ])

    encoder_input = model.input
    encoder_output = model.output

    encoder_mean = Dense(latent_dim, name='mean')(encoder_output)
    encoder_log_var = Dense(latent_dim, name='log_var')(encoder_output)

    return Model(inputs=encoder_input, outputs=[encoder_mean, encoder_log_var], name='encoder')

def build_decoder(encoder, kernel_size=(5,5), strides=(2,2), latent_dim=256):
    hidden_units = encoder.layers[-4].output_shape[1]
    hidden_shape = encoder.layers[-5].output_shape[1:]

    model = tf.keras.Sequential([
        Input(shape=(latent_dim,), name='decoder_input'),
        Dense(hidden_units, activation='relu'),
        Reshape(hidden_shape),
        Conv2DTranspose(512, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2DTranspose(256, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2DTranspose(128, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2DTranspose(64, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2DTranspose(32, kernel_size=kernel_size, strides=strides, padding='same', activation=swish),
        Conv2DTranspose(3, kernel_size=kernel_size, strides=strides,
        padding='same', activation='sigmoid', name='decoder_output')
    ], name='decoder')

    return model

def build_autoencoder(encoder, decoder, input_shape=(128, 128, 3), latent_dim=256, beta=1):
    # Define encoder input and get latent vectors
    encoder_inputs = Input(shape=input_shape, name='autoencoder_input')
    z_mean, z_log_var = encoder(encoder_inputs)

    # Reparameterization trick to sample z
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # Apply sampling function
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Obtain the reconstruction from the decoder
    reconstruction = decoder(z)

    # Build the VAE model
    vae = Model(inputs=encoder_inputs, outputs=reconstruction, name='vae_mlp')

    # Add VAE loss
    reconstruction_loss = mse(K.flatten(encoder_inputs), K.flatten(reconstruction))
    reconstruction_loss *= input_shape[0] * input_shape[1] 
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + beta * kl_loss)
    vae.add_loss(vae_loss)

    return vae

def build_model(latent_dim=256):
    encoder = build_encoder(latent_dim=latent_dim, kernel_size=4, strides=2)
    decoder = build_decoder(encoder=encoder, kernel_size=3, latent_dim=latent_dim,  strides=2)
    autoencoder = build_autoencoder(encoder,decoder)

    opt = Adam(3e-4)
    autoencoder.compile(opt)
    return autoencoder, encoder, decoder

# vae, encoder, decoder = build_model(latent_dim=512)

# encoder.summary()
# decoder.summary()
# vae.summary()