# PokeGenerator Project
# Loads dataset

from pathlib import Path
import imageio.v3 as iio
import numpy as np


def load_dataset(config):
    """Load images from dataset directory and cache"""

    # Load cache if exists
    if Path(config["dataset_cache"]).is_file() and config["dataset_cache_enable"]:
        print("Dataset cache exists, loading from cache")

        images = np.load(config["dataset_cache"], allow_pickle=True)
    # Create cache if doesn't exist
    else:
        if config["dataset_cache_enable"]:
            print("Dataset cache doesn't exist, building cache")
        else:
            print("Dataset cache disabled")

        # Load images
        images = list()
        for file in Path(config["dataset_directory"]).iterdir():
            if not file.is_file():
                continue

            images.append(iio.imread(file))

        # Convert to numpy array
        images = np.array(images)

        # Update cache
        if config["dataset_cache_enable"]:
            images.dump(config["dataset_cache"])

    return images
