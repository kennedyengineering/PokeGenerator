# PokeGenerator Project
# Trains model

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from model.autoencoder import build_model
from model.dataset import load_dataset


CONFIG_FILE = "config.json"


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
    dataset_config = config["dataset"]
    training_config = config["training"]

    # Load the dataset
    images = load_dataset(dataset_config)

    # Preprocess the dataset (Normalize to [0, 1] range), is necessary for binary_crossentropy loss
    images = images.astype(np.float32) / 255.0

    # Build and train model
    model, _, _ = build_model()
    # TODO: Save history to file
    history = model.fit(
        images,
        images,
        batch_size=training_config["batch_size"],
        epochs=training_config["epochs"],
        shuffle=True,
    )

    # Save the model
    # TODO: Add checkpointing, save models every X epochs to an directory corresponding to a training run
    # TODO: Use callbacks to save model checkpoints, and produce example inference image (to show progression)
    checkpoint_directory = Path(training_config["checkpoint_directory"])
    checkpoint_directory.mkdir(exist_ok=True)
    model.save(
        checkpoint_directory
        / ("model_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S%z") + ".keras")
    )


if __name__ == "__main__":
    main()
