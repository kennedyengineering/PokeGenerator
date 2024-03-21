import json
import os
import numpy as np
import tensorflow as tf
from model.autoencoder_train import load_dataset
import glob
import variational_autoencoder as vae
from latent_diffusion import (
    build_reverse_process_mlp_model, 
    train_model, 
    sample
)
from datetime import datetime
from pathlib import Path

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
    
    # Save the model
    checkpoint_directory = Path(config["checkpoint_directory"])
    checkpoint_directory.mkdir(exist_ok=True)
    ld_model.save(
        checkpoint_directory
        / ("latent_model_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S%z") + ".keras")
    )

if __name__ == "__main__":
    main()
