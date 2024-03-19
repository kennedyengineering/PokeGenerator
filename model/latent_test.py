import json
import os
import numpy as np
import tensorflow as tf
from train import load_dataset
import matplotlib.pyplot as plt
import glob

def display_images(images, titles=None, cols=5, figsize=(20, 10)):
    """
    Display a list of images in a grid.

    Args:
        images: A list or array of images to display.
        titles: Optional list of titles for each image.
        cols: Number of columns in the image grid.
        figsize: Tuple indicating the size of the figure.
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')  # Adjust as needed
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def interpolate_vectors(v1, v2, num_steps=10):
    """Interpolates between two vectors with a specified number of steps."""
    ratios = np.linspace(0, 1, num_steps)
    interpolated_vectors = [(1 - ratio) * v1 + ratio * v2 for ratio in ratios]
    return np.array(interpolated_vectors)

def plot_interpolated_images(decoder, latent_vectors, figsize=(10, 2)):
    """Plots a series of images showing the transition between two points."""

    latent_vector_1 = latent_vectors[0]  # Replace 0 with a specific index
    latent_vector_2 = latent_vectors[1]  # Replace 1 with another index

    # Interpolate between the two latent vectors
    interpolated_vectors = interpolate_vectors(
        latent_vector_1, latent_vector_2, num_steps=10
    )

    decoded_images = decoder.predict(interpolated_vectors)

    plt.figure(figsize=figsize)
    num_images = len(decoded_images)
    for i, image in enumerate(decoded_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    with open("training_config.json", "r") as f:
        config = json.load(f)
    images = load_dataset(config)

    model_files = glob.glob("./checkpoints/*.keras")
    latest_model_file = max(model_files, key=os.path.getmtime)
    print(f"Attempting to load model from: {latest_model_file}")
    variational_autoencoder = tf.keras.models.load_model(latest_model_file)

    encoder = variational_autoencoder.get_layer('encoder')
    decoder = variational_autoencoder.get_layer('decoder')

    encoded_imgs = encoder.predict(images[:2]) 
    z_mean = encoded_imgs[0]  

    plot_interpolated_images(decoder, z_mean)

    reconstructed_images = variational_autoencoder.predict(images[:10])
    display_images(reconstructed_images, titles=['Reconstructed'] * 10)

if __name__ == "__main__":
    main()
