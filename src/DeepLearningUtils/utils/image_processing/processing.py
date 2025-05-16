
import cv2
import numpy as np


def create_gaussian_mask(old_resolution, image_size, center, sigma):
    """
    Create a gaussian mask with given center and sigma

    # of rows = height
    # of columns = width

    Args:
        old_resolution - tuple with (height,width)
        image_size - tuple with (height,width)
        center - tuple with (x,y)
        sigma -  tuple with (sigma_x,sigma_y)
    Returns:
        mask - numpy array with the gaussian mask
    """
    mask = np.zeros(old_resolution)

    if center[0] >= 0:
        mask[center[1], center[0]] = 1

    mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_AREA)

    mask = cv2.GaussianBlur(mask, sigma, 0)

    if mask.max() > 0:
        mask = mask / np.max(mask)

    return mask