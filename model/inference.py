# PokeGenerator Project
# Runs model

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model

from model.dataset import load_dataset


CONFIG_FILE = "config.json"


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_path",
        help="Path to .keras File",
        type=str,
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
    inference_config = config["inference"]

    # Load the dataset
    images = load_dataset(dataset_config)

    # Preprocess the dataset (Normalize to [0, 1] range)
    images = images.astype(np.float32) / 255.0

    # Load and summarize model
    model = load_model(args.model_path)
    model.summary()

    # Run inference
    output = model.predict(images)

    # Show images
    for i in range(len(output)):
        cv2.imshow("Model Inference", cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR))
        if cv2.waitKey(-1) & 0xFF == ord("q"):
            break

    # TODO: Creation option to save images to disk
    # TODO: Show embedding space


if __name__ == "__main__":
    main()
