import json
from latent_diffusion import (
    build_reverse_process_mlp_model, 
    training,
    sample
)
import numpy as np
import tensorflow as tf
from train import load_dataset
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
# import matplotlib.pyplot as plt

# def display_images(images, titles=None, cols=5, figsize=(20, 10)):
#     """
#     Display a list of images in a grid.

#     Args:
#         images: A list or array of images to display.
#         titles: Optional list of titles for each image.
#         cols: Number of columns in the image grid.
#         figsize: Tuple indicating the size of the figure.
#     """
#     n_images = len(images)
#     rows = (n_images + cols - 1) // cols
#     plt.figure(figsize=figsize)
#     for i, image in enumerate(images):
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(image, cmap='gray')  # Adjust as needed
#         if titles is not None:
#             plt.title(titles[i])
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

def main():
    with open("training_config.json", "r") as f:
        config = json.load(f)
    images = load_dataset(config)  

    # TODO: Scrap from config file
    autoencoder = tf.keras.models.load_model("model_2024-03-16-17:16:18.keras")
    predictions = autoencoder.predict(images)
    autoencoder.summary()

    # Access the encoder and decoder directly
    encoder = autoencoder.get_layer('encoder')
    decoder = autoencoder.get_layer('decoder')

    # Summary of the encoder and decoder to verify
    encoder.summary()
    decoder.summary()

    # # rgb_predictions = safely_convert_to_rgb(predictions)
    # rgb_predictions = predictions
    # display_images(rgb_predictions[:10]) 

    input_dim = 8192
    num_layers = 32    # Number of hidden layers
    num_hidden = 800  # Number of neurons in each hidden layer
    T = 1000
    latent_model = build_reverse_process_mlp_model(input_dim, num_layers, num_hidden, T)
    latent_model.summary()

    betas = np.linspace(1e-4, .02, T)
    sigmas = np.sqrt(betas)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas, axis=-1)
    batch_size = 128
    

    latent_vectors = encoder.predict(images)
    latent_model = training(
        latent_vectors, batch_size, T, 
        alphas_cumprod, latent_model
    )

    # Sample latent vectors
    sampled_latent_vectors = sample(latent_model, shape=latent_vectors.shape)

    # Decode sampled latent vectors into images
    decoded_images = autoencoder.decoder.predict(sampled_latent_vectors)

    # # Visualize the latent space
    # plt.figure(figsize=(10, 5))
    # plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], alpha=0.5, label="Training Latent Vectors")
    # plt.scatter(sampled_latent_vectors[:, 0], sampled_latent_vectors[:, 1], alpha=0.5, label="Sampled Latent Vectors")
    # plt.legend()
    # plt.title("Latent Space Visualization")
    # plt.show()

if __name__ == "__main__":
    main()