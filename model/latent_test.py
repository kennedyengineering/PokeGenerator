import json
import os
import numpy as np
import tensorflow as tf
from train import load_dataset
import matplotlib.pyplot as plt
import glob
import variational_autoencoder as vae
from latent_diffusion import build_reverse_process_mlp_model, train_model, sample
from datetime import datetime
from pathlib import Path

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

def generate_images(decoder, num_images=10, latent_dim=2048):
    """
    Generates images by sampling from the latent space and decoding.

    Args:
        decoder: The decoder model.
        num_images: Number of images to generate.
        latent_dim: The dimensionality of the latent space.
    """
    # Sample random vectors from the normal distribution
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
    print(random_latent_vectors.shape)

    # Decode the random latent vectors into images
    generated_images = decoder.predict(random_latent_vectors)

    # Display the generated images
    display_images(generated_images, cols=num_images, figsize=(20, 10))

def display_image_pairs(originals, reconstructions, page=1, pairs_per_page=4):
    """Displays original and reconstructed image pairs in a gallery format."""
    total_pairs = len(originals)
    start = (page - 1) * pairs_per_page * 2
    end = start + pairs_per_page * 2
    images_to_show = np.concatenate((originals[start//2:end//2], reconstructions[start//2:end//2]))
    titles = ['Original' if i < pairs_per_page else 'Reconstructed' for i in range(pairs_per_page * 2)]

    cols = 4  # Two columns per pair, original and reconstructed
    rows = (len(images_to_show) + cols - 1) // cols
    plt.figure(figsize=(10, 2.5 * rows))
    for i, image in enumerate(images_to_show):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')  # Adjust as needed
        plt.title(titles[i % pairs_per_page])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    with open("training_config.json", "r") as f:
        config = json.load(f)
    images = load_dataset(config)

    ae_model_files = glob.glob("checkpoints/model_*.keras")
    latest_ae_model_file = max(ae_model_files, key=os.path.getmtime)
    print(f"Attempting to load model from: {latest_ae_model_file}")
    variational_autoencoder = tf.keras.models.load_model(latest_ae_model_file)
    variational_autoencoder.summary()

    encoder = variational_autoencoder.get_layer('encoder')
    decoder = variational_autoencoder.get_layer('decoder')

    # Show Reconstruction
    reconstructions = variational_autoencoder.predict(images)

    # Show in gallery format of original compare to reconstructed
    display_image_pairs(images, reconstructions, page=1)
    
    latent_vectors = encoder.predict(images)
    z_mean, z_log_var = latent_vectors

    sampled_latent_vectors = vae.sampling(latent_vectors)

    # Ensure the shape is what your model expects
    print("Sampled latent vectors shape:", sampled_latent_vectors.shape) 
    # Demonison of Latent Space
    print("Latent Space Dimension:", sampled_latent_vectors.shape[1])

    # Reverse Process Model
    latent_diffusion_model = build_reverse_process_mlp_model(input_dim=sampled_latent_vectors.shape[1], num_layers=3, num_hidden=512, T=1000)
    latent_diffusion_model.summary()

    # # Training
    T = 1000
    betas = np.linspace(1e-4, .02, T)
    sigmas = np.sqrt(betas)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas, axis=-1)

    # Train or Load Model
    if config["train_latent_model"]:
        ld_model = train_model(
            sampled_latent_vectors, 
            batch_size=config["batch_size"], 
            T=T, 
            alphas_cumprod=alphas_cumprod, 
            model=latent_diffusion_model, 
            epochs=config['epochs']
        )
    else:
        latent_model_files = glob.glob("checkpoints/latent_model_*.keras")
        latest_latent_model_file = max(latent_model_files, key=os.path.getmtime)
        print(f"Attempting to load model from: {latest_latent_model_file}")
        ld_model = tf.keras.models.load_model("model/checkpoints/latent_model_*.keras")
        ld_model.summary()

    # Sample
    ld_sampled_vectors = sample(
        ld_model, 
        sampled_latent_vectors.shape, 
        T, 
        sigmas, 
        alphas, 
        alphas_cumprod
    ) 
    decode_images = decoder.predict(ld_sampled_vectors)

    print("Decoded Images Shape:", decode_images[0].shape)
    
    # # Display the generated images
    display_images(decode_images, cols=10, figsize=(20, 10))

    checkpoint_directory = Path(config["checkpoint_directory"])
    checkpoint_directory.mkdir(exist_ok=True)
    # Save the model
    ld_model.save(
        checkpoint_directory
        / ("latent_model_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S%z") + ".keras")
    )

if __name__ == "__main__":
    main()
