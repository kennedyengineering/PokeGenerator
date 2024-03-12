# PokeGenerator Project
# Main program for generating dataset

import os
import json
import argparse
import numpy as np
import imageio.v3 as iio


CONFIG_FILE = "dataset_config.json"
BLACKLIST_FILE = "blacklist.json"


def draw_bounding_box(bbox, rgb_img, color=(100, 100, 100)):
    """Renders bounding box on image

    Receives bounding box in form [x, y, w, h]
    """

    rgb_img[bbox[1], bbox[0] : bbox[0] + bbox[2]] = color
    rgb_img[bbox[1] : bbox[1] + bbox[3], bbox[0]] = color
    rgb_img[bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]] = color
    rgb_img[bbox[1] : bbox[1] + bbox[3], bbox[0] + bbox[2]] = color

    return rgb_img


def find_bounding_box(rgb_img):
    """Return bounding box of sprite

    In form [x, y, w, h] where x, y are bottom left coordinates
    """

    a = np.where(rgb_img != (255, 255, 255))
    bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
    bbox = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

    return bbox


def RGBA_to_RGB(rgba_img):
    """Converts a RGBA image to an RGB image

    NOTES:
    If the alpha channel does not create a clean mask (i.e, contains values other then 0 or 255)
    then the resulting image cannot be cleanly shown with a black background.

    For this reason, a white background is used instead of black.
    """

    # Split the RGB and A channels
    rgb = rgba_img[:, :, :3]
    alpha = rgba_img[:, :, 3]

    # Use A channel as mask of Pokemon
    img = np.ones_like(rgb) * 255
    idx = alpha != 0
    img[idx] = rgb[idx]

    return img


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
    for path in blacklist["good_directories"]:
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
            if f not in blacklist["bad_files"] or not config["apply_blacklist"]
        ]

        image_paths.extend(filtered_image_paths)

    # Create output directory
    os.mkdir(config["output_directory"])

    # Process images
    for i, path in enumerate(image_paths):
        # Load image from disk
        image = iio.imread(path)
        output_path = os.path.join(config["output_directory"], f"{i}.png")

        # Remove A channel
        if image.shape[2] == 4:
            image = RGBA_to_RGB(image)

        # Compute bounding box
        bbox = find_bounding_box(image)
        print(bbox)

        image = draw_bounding_box(bbox, image)

        # Save image to disk
        iio.imwrite(output_path, image)


if __name__ == "__main__":
    main()
