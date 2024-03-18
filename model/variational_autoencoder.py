from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Input, LeakyReLU,
    Conv2DTranspose, Reshape, UpSampling2D, Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam

def build_encoder(input_shape=(128,128,3), kernel_shape=(3,3), latent_dim=8192):
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(16, kernel_shape, activation='relu', padding='same')(encoder_inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, kernel_shape, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, kernel_shape, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    encoder_mean = Dense(latent_dim, name='mean')(x)
    encoder_log_var = Dense(latent_dim, name='log_var')(x)

    return Model(inputs=encoder_inputs,outputs=[encoder_mean,encoder_log_var], name='encoder')

def build_decoder(encoder, kernel_shape=(3,3), latent_dim=8192):
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    hidden_units = encoder.layers[-4].output_shape[1]
    x = Dense(hidden_units, activation='relu')(latent_inputs)
    hidden_shape = encoder.layers[-5].output_shape[1:]
    x = Reshape(hidden_shape)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, kernel_shape, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64, kernel_shape, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, kernel_shape, activation='relu', padding='same')(x)
    decoder_outputs = Conv2DTranspose(3, kernel_shape, activation='sigmoid', padding='same', name='decoder')(x)
    return Model(inputs=latent_inputs, outputs=decoder_outputs, name='decoder')

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
    encoder = build_encoder(latent_dim=latent_dim)
    encoder.summary()
    decoder = build_decoder(encoder=encoder, latent_dim=latent_dim)
    decoder.summary()
    autoencoder = build_autoencoder(encoder,decoder)
    autoencoder.summary()  

    opt = Adam(3e-4)
    autoencoder.compile(opt)
    return autoencoder, encoder, decoder

# vae, _, _ = build_model(latent_dim=256)