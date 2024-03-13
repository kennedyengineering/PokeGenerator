# PokeGenerator Project
# Main program for generating dataset

import os
import json
import argparse
import numpy as np
import imageio.v3 as iio
import cv2
from tqdm import tqdm


CONFIG_FILE = "dataset_config.json"
BLACKLIST_FILE = "blacklist.json"


def crop_bounding_square(bbox, rgb_img):
    """Returns the contents of the image inside the sqaure bounding box of the bounding box

    Receives bounding box in form [x, y, w, h]
    """

    side_length = max(bbox[2], bbox[3])
    center_point = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    new_bbox = (
        int(center_point[0] - side_length / 2),
        int(center_point[1] - side_length / 2),
        side_length,
        side_length,
    )

    return crop_bounding_box(new_bbox, rgb_img)


def crop_bounding_box(bbox, rgb_img):
    """Returns the contents of the image inside the bounding box

    Receives bounding box in form [x, y, w, h]
    """

    # Extend image dimensions if necessary
    x1 = bbox[0]
    x2 = bbox[0] + bbox[2] + 1
    y1 = bbox[1]
    y2 = bbox[1] + bbox[3] + 1

    if x1 < 0:
        padding = (
            np.ones((rgb_img.shape[0], abs(x1), rgb_img.shape[2]), dtype=rgb_img.dtype)
            * 255
        )
        x2 += abs(x1)
        x1 = 0
        rgb_img = np.concatenate([padding, rgb_img], axis=1)

    if y1 < 0:
        padding = (
            np.ones((abs(y1), rgb_img.shape[1], rgb_img.shape[2]), dtype=rgb_img.dtype)
            * 255
        )
        y2 += abs(y1)
        y1 = 0
        rgb_img = np.concatenate([padding, rgb_img], axis=0)

    if x2 > rgb_img.shape[1]:
        # TODO: populate if needed
        pass

    if y2 > rgb_img.shape[0]:
        # TODO: populate if needed
        pass

    return rgb_img[y1:y2, x1:x2]


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
    """Finds bounding box of sprite

    Returns bounding box in form [x, y, w, h] where x, y are bottom left coordinates
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
    for i, path in enumerate(
        tqdm(image_paths, desc="Generating Pokemon Sprite Dataset")
    ):
        # Load image from disk
        image = iio.imread(path, pilmode="RGBA")
        output_path = os.path.join(config["output_directory"], f"{i}.png")

        # Remove A channel
        if image.shape[2] == 4:
            image = RGBA_to_RGB(image)

        # Compute bounding box
        bbox = find_bounding_box(image)

        # Crop the image
        image = crop_bounding_square(bbox, image)

        # Resize the image
        image = cv2.resize(
            image,
            dsize=(config["sprite_size"], config["sprite_size"]),
            interpolation=cv2.INTER_CUBIC,
        )

        # Save image to disk
        iio.imwrite(output_path, image)


if __name__ == "__main__":
    main()
