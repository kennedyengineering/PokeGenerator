# PokeGenerator Project
# Main program for generating dataset

import os
import json
import argparse
import numpy as np
import imageio.v3 as iio


CONFIG_FILE = "dataset_config.json"
BLACKLIST_FILE = "blacklist.json"


def has_black_background(rgb_img):
    """Returns True if the image background is black, otherwise False
    It assumes the majority of the image's border pixels are the background color.
    """
    top_rows = rgb_img[0, :, :]
    bottom_rows = rgb_img[-1, :, :]
    left_cols = rgb_img[:, 0, :]
    right_cols = rgb_img[:, -1, :]
    together = np.reshape(
        np.concatenate([top_rows, bottom_rows, left_cols, right_cols], axis=0), [-1]
    )
    vals, counts = np.unique(together, return_counts=True)
    return vals[np.argmax(counts)] == 0.0


def RGBA_to_black_background(rgba_img):
    """Converts a non-black background rgba image to a black background rgb image"""
    rgb = rgba_img[:, :, :3]
    a = rgba_img[:, :, 3]
    black = rgb * (np.expand_dims(a, axis=-1) / 255)
    return black


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="Config JSON File",
        type=argparse.FileType("r"),
        default=CONFIG_FILE,
    )
    parser.add_argument(
        "--blacklist_file",
        help="Blacklist JSON File",
        type=argparse.FileType("r"),
        default=BLACKLIST_FILE,
    )
    args = parser.parse_args()

    # Load JSON Data
    config = json.load(args.config_file)
    blacklist = json.load(args.blacklist_file)

    # Load paths of images
    image_paths = []
    for path in blacklist[
        "good_directories"
    ]:  # FIXME: Blacklist unwanted directories, instead of manually specifing the goodones. Then the apply blacklist switch can turn off the filtering, and file discovery can happen organically, if desired.
        # Read all images
        raw_image_paths = [
            os.path.join(path, f)
            for f in os.listdir(os.path.join(config["source_directory"], path))
            if f.endswith(".png")
        ]

        # Filter bad images
        filtered_image_paths = [
            os.path.join(config["source_directory"], f)
            for f in raw_image_paths
            if f not in blacklist["bad_files"]
        ]

        image_paths.extend(filtered_image_paths)

    # Create output directory
    os.mkdir(config["output_directory"])

    # Process images
    for i, path in enumerate(image_paths):
        image = iio.imread(path)
        output_path = os.path.join(config["output_directory"], f"{i}.png")

        if has_black_background(image):
            # Save black background images to disk
            iio.imwrite(output_path, image)
        elif image.shape[2] == 4:
            # Convert RGBA images to black background images
            iio.imwrite(output_path, RGBA_to_black_background(image))
        else:
            # Report error condition
            print(f"ERROR: Cannot process image {path}")


if __name__ == "__main__":
    main()
