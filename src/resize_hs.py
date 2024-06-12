import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from spectral import open_image
from hs_utils import read_calib, apply_calib, find_closest_wavelength

# Example arguments
# args.input = "data/sierra/images"
# args.output = "data/sierra/images"
# args.input = "data/sierra/images"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to folder containing the .bil and .hdr images",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Path where to store the resized file, as .npy arrays",
    )
    parser.add_argument(
        "--calib",
        "-c",
        required=False,
        type=str,
        help="Path to .csv calib file of the HS camera",
    )
    parser.add_argument(
        "--new-width",
        "-w",
        required=False,
        type=int,
        default=224,
        help="Width to which the HS images should be resized. Original aspect ratio is preserved",
    )
    parser.add_argument(
        "--save-rgb",
        required=False,
        type=bool,
        default=True,
        help="If True, a pseudo-rgb image will also be computed and saved in the output folder",
    )
    args = parser.parse_args()

    # Read args
    folder = args.input
    calib_path = args.calib
    new_w = args.new_width
    output_folder = args.output

    # Retrieve images
    filenames = [x.split(".")[0] for x in os.listdir(folder) if ".bil" in x]

    for fname in tqdm(filenames):
        filepath = os.path.join(folder, fname)
        bil_path = filepath + ".bil"
        hdr_path = filepath + ".bil.hdr"

        # Read HS image and calib
        img = open_image(hdr_path).load()
        h, w = img.shape[:2]
        calib = read_calib(calib_path)

        # Apply calibration
        calib_image = apply_calib(img, calib)

        # Resize image
        new_h = round(h / (w / new_w))
        resized_img = cv2.resize(calib_image, (new_w, new_h))

        # Save output
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, fname + ".npy")
        with open(output_path, "wb") as f:
            np.save(f, resized_img)

        if args.save_rgb:
            # Get pseudo-RGB
            rgb_targets = [630, 532, 465]
            rgb_wavs = [
                find_closest_wavelength(calib["wavelength"], wav) for wav in rgb_targets
            ]
            rgb_image = resized_img[:, :, rgb_wavs]
            rgb_image_norm = (rgb_image / rgb_image.max() * 255).astype(int)

            # Save pseudo-RGB
            output_path = os.path.join(output_folder, fname + ".png")
            cv2.imwrite(output_path, rgb_image_norm)
