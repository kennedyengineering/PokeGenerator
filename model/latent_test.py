from datetime import datetime
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
import cv2
import matplotlib.pyplot as plt

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

    autoencoder = tf.keras.models.load_model("model_2024-03-17-18:52:31.keras")
    autoencoder.summary()

    encoder = autoencoder.get_layer('encoder')
    decoder = autoencoder.get_layer('decoder')

    latent_vectors = encoder.predict(images)

    plot_interpolated_images(decoder, latent_vectors)

    # # rgb_predictions = safely_convert_to_rgb(predictions)
    # rgb_predictions = predictions
    # display_images(rgb_predictions[:10]) 

    # input_dim = 8192
    # num_layers = 8    # Number of hidden layers
    # num_hidden = 4000  # Number of neurons in each hidden layer
    # T = 1000
    # # latent_model = build_reverse_process_mlp_model(input_dim, num_layers, num_hidden, T)
    # # latent_model.summary()

    # betas = np.linspace(1e-4, .02, T)
    # sigmas = np.sqrt(betas)
    # alphas = 1 - betas
    # alphas_cumprod = np.cumprod(alphas, axis=-1)
    # batch_size = 128

    latent_vectors = encoder.predict(images)
    # latent_model = training(
    #     latent_vectors, batch_size, T, 
    #     alphas_cumprod, latent_model, epochs=400
    # )

    # # TODO: Inspect the latent vectors

    # # Save Model
    # # latent_model.save("checkpoints/latent_model_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S%z") + ".keras")

    # diffusion_model = tf.keras.models.load_model(
    #     "latent_model_2024-03-17-00:02:10.keras"
    # )

    # # Sample latent vectors
    # sampled_latent_vectors = sample(
    #     diffusion_model, shape=latent_vectors.shape, T=T, sigmas=sigmas, alphas=alphas, alphas_cumprod=alphas_cumprod
    # )

    # # Save the sampled latent vectors
    # np.save("sampled_latent_vectors.npy", sampled_latent_vectors)

    # # Open Latent Space Visualization
    # # Load the data back from the .npy file
    # sampled_latent_vectors = np.load("sampled_latent_vectors.npy")
    
    # # Display the data
    # print(sampled_latent_vectors)
    
    

    # # # Decode sampled latent vectors into images
    # output = decoder.predict(sampled_latent_vectors)

    # for i in range(len(predictions)):
    #     cv2.imshow("yo", cv2.cvtColor(predictions[i], cv2.COLOR_RGB2BGR))
    #     if cv2.waitKey(-1) & 0xFF == ord("q"):
    #         break

    # # Visualize the latent space
    # # plt.figure(figsize=(10, 5))
    # # plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], alpha=0.5, label="Training Latent Vectors")
    # # plt.scatter(sampled_latent_vectors[:, 0], sampled_latent_vectors[:, 1], alpha=0.5, label="Sampled Latent Vectors")
    # # plt.legend()
    # # plt.title("Latent Space Visualization")
    # # plt.show()

if __name__ == "__main__":
    main()


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
