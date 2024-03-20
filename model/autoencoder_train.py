# PokeGenerator Project
# Trains model

import argparse
import json
from pathlib import Path
from datetime import datetime
import imageio.v3 as iio
import numpy as np
import tensorflow as tf

# TODO: Add option to select model to train
# from autoencoder import build_model
#from conv_autoencoder import build_model
from variational_autoencoder import build_model
# from vqvae.vqvae_trainer import VQVAETrainer

CONFIG_FILE = "training_config.json"


def load_dataset(config):
    """Load images from dataset directory and cache"""

    # TODO: Add cache toggle switch in config file

    # Load cache if exists
    if Path(config["dataset_cache"]).is_file():
        print("Dataset cache exists, loading from cache")

        images = np.load(config["dataset_cache"], allow_pickle=True)
    # Create cache if doesn't exist
    else:
        print("Dataset cache doesn't exist, building cache")

        # Load images
        images = list()
        for file in Path(config["dataset_directory"]).iterdir():
            if not file.is_file():
                continue

            images.append(iio.imread(file))

        # Convert to numpy array
        images = np.array(images)

        # Update cache
        images.dump(config["dataset_cache"])
    
    # Preprocess the dataset (Normalize to [-1, 1] range)   
    # FIXME: is [0 1 range better?] -- is necessary for binary_crossentropy loss
    # images = (images.astype(np.float32) / 127.5) - 1.0
    images = images.astype(np.float32) / 255.0

    return images


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config_file",
        help="Config JSON File",
        type=argparse.FileType("r"),
        default=CONFIG_FILE,
    )
    args = parser.parse_args()

    config = json.load(args.config_file)
    images = load_dataset(config)
    # images_variance = np.var(images / 255.0)

    # Build and train model, TODO: Add latent_dim to config file
    # model, encoder, decoder = build_model(latent_dim=1024)
    # model = VQVAETrainer(images_variance)
    # model.compile(optimizer=tf.keras.optimizers.Adam(3e-4))

    # history = model.fit(
    #     images,
    #     images,
    #     batch_size=config["batch_size"],
    #     epochs=config["epochs"],
    #     validation_split=config["validation_split"], 
    #     shuffle=True,
    #     callbacks=[
    #         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=(config["epochs"] // 50), restore_best_weights=True, verbose=1),
    #     ],
    # )

    # # Save the model
    # # TODO: Add checkpointing, save models every X epochs to an directory corresponding to a training run
    # # TODO: Use callbacks to save model checkpoints, and produce example inference image (to show progression)
    # checkpoint_directory = Path(config["checkpoint_directory"])
    # checkpoint_directory.mkdir(exist_ok=True)
    # model.save(
    #     checkpoint_directory
    #     / ("model_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S%z") + ".keras")
    # )

if __name__ == "__main__":
    main()