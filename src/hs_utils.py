import re
from functools import partial
import numpy as np
import pandas as pd


def read_calib(calib_path):
    calibration_df = pd.read_csv(calib_path)
    return calibration_df


def apply_calib(image, calib_data):
    calib_values = calib_data["reflectance"].values
    calibrated_image = image * calib_values.reshape(1, 1, -1)
    return calibrated_image


def find_closest_wavelength(wavs, target):
    return (np.abs(np.array(wavs) - np.array(target))).argmin()


perc_mapping = {
    "0": "0.0",
    "8": "8.0",
    "16": "16.7",
    "32": "33.33",
    "64": "66.7",
    "100": "100.0",
}
elt_mapping = {"N": "N", "P": "P", "K": "K", "CA": "Ca", "S": "S", "MG": "Mg"}
rep_mapping = {"A": 1, "B": 2, "C": 3, "D": 4}


def get_label_from_fname(fname, gt_data):
    regex = r"([A-Z]+)([0-9]+)_([A-Z]+)*"

    # Find element, percentage and rep number in fname
    elt, perc, rep = re.match(regex, fname, re.I).groups()
    elt, perc, rep = elt_mapping[elt], perc_mapping[perc], rep_mapping[rep]

    # Find corresponding row in csv file
    match1 = gt_data[gt_data["Element"] == elt]
    match2 = match1[match1["Percentage"].astype(str) == perc]
    match3 = match2[match2["Rep#"] == rep]

    cols = ["N", "P", "K", "Ca", "Mg", "S"]
    try:
        label = {colname: float(match3[colname].iloc[0]) for colname in cols}
        return label
    except ValueError:
        return None


def hs_crop(image, crop_size=(224, 224)):
    """
    Takes a random crop from a hypercube.

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array of shape (h, w, c).
        crop_size (tuple): The desired crop size (height, width).

    Returns:
        numpy.ndarray: The cropped image.
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Ensure the crop size is not greater than the image size
    # print("(h, w)", (h, w))
    if crop_h > h or crop_w > w:
        raise ValueError("Crop size must be smaller than the image dimensions.")

    # Choose the top-left corner of the crop randomly
    start_h = np.random.randint(0, h - crop_h + 1)
    start_w = np.random.randint(0, w - crop_w + 1)

    # Perform the crop
    return image[start_h : start_h + crop_h, start_w : start_w + crop_w, :]


def hs_random_horizontal_flip(image, p=0.5):
    """
    Flips an image array horizontally (left to right).

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array of shape (h, w, c).

    Returns:
        numpy.ndarray: The horizontally flipped image.
    """
    if np.random.rand() < p:
        # Reverse the order of columns, using all rows and channels
        return image[:, ::-1, :]
    else:
        # Return the original image if no flip is performed
        return image


def hs_transforms(
    crop_size=(224, 224),
    flip_lr_prob=0.5,
):
    """
    Returns list of transforms to be applied sequentially to the HS image.
    """
    return [
        partial(hs_crop, crop_size=crop_size),
        partial(hs_random_horizontal_flip, p=flip_lr_prob),
    ]
