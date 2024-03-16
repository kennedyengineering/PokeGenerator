from latent_diffusion import build_reverse_process_mlp_model
import numpy as np

def main():
    input_dim = 2
    num_layers = 3    # Number of hidden layers
    num_hidden = 128  # Number of neurons in each hidden layer
    T = 1000
    model = build_reverse_process_mlp_model(input_dim, num_layers, num_hidden, T)
    model.summary()

    # T = 1000
    # betas = np.linspace(1e-4, .02, T)
    # sigmas = np.sqrt(betas)
    # alphas = 1 - betas
    # alphas_cumprod = np.cumprod(alphas, axis=-1)

    # batch_size = 128

    # # Sample latent vectors
    # sampled_latent_vectors = sample(model, shape=latent_vectors.shape)

    # # Decode sampled latent vectors into images
    # decoded_images = decoder.predict(sampled_latent_vectors)

    # # Visualize the latent space
    # plt.figure(figsize=(10, 5))
    # plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], alpha=0.5, label="Training Latent Vectors")
    # plt.scatter(sampled_latent_vectors[:, 0], sampled_latent_vectors[:, 1], alpha=0.5, label="Sampled Latent Vectors")
    # plt.legend()
    # plt.title("Latent Space Visualization")
    # plt.show()

if __name__ == "__main__":
    main()