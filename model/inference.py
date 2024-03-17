# PokeGenerator Project
# Trains model

import argparse
import json
from pathlib import Path
from datetime import datetime
import imageio.v3 as iio
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# TODO: Add option to select model to train
# from autoencoder import build_model
from conv_autoencoder import build_model


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
    parser.add_argument(
        "--model_path",
        help="Path to .keras Checkpoint",
        type=str,
    )
    args = parser.parse_args()

    # Load JSON Data
    config = json.load(args.config_file)

    # Load the dataset
    images = load_dataset(config)

    # Preprocess the dataset (Normalize to [-1, 1] range)   # FIXME: is [0 1 range better?] -- is necessary for binary_crossentropy loss
    # images = (images.astype(np.float32) / 127.5) - 1.0
    images = images.astype(np.float32) / 255.0

    # Build and inference model
    model = load_model(args.model_path)
    model.summary()

    output = model.predict(images)

    for i in range(len(output)):
        cv2.imshow("yo", cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))
        if cv2.waitKey(-1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
