# PokeGenerator Project
# Trains model

import argparse
import json
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from autoencoder import build_model


CONFIG_FILE = "training_config.json"


def load_dataset(config):
    """Load images from dataset directory and cache"""

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
    args = parser.parse_args()

    # Load JSON Data
    config = json.load(args.config_file)

    # Load the dataset
    images = load_dataset(config)

    # Preprocess the dataset
    images = (images.astype(np.float32) / 127.5) - 1.0

    # Build and train model
    model, _, _ = build_model()
    np.random.shuffle(images)
    history = model.fit(
        images,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=config["validation_split"],
    )


if __name__ == "__main__":
    main()
