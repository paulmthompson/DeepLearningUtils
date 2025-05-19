import numpy as np
from typing import List, Tuple, Union, Any


def convert_img_numpy(images: List[np.ndarray]) -> np.ndarray:
    """
    Convert a list of image arrays into a single stacked numpy array.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of image arrays to stack
        
    Returns
    -------
    np.ndarray
        Stacked array of shape (n_images, height, width, channels)
    """
    return np.stack(images)


def transpose_label_list(labels: List[List[Any]]) -> List[List[Any]]:
    """
    Transpose a list of label lists. Useful for converting from per-keypoint
    organization to per-image organization.
    
    Parameters
    ----------
    labels : List[List[Any]]
        List of label lists where each inner list corresponds to one keypoint type
        and contains labels for all images
        
    Returns
    -------
    List[List[Any]]
        Transposed list where each inner list corresponds to one image and contains
        labels for all keypoint types
    """
    return list(map(list, zip(*labels)))


def convert_labels_numpy(labels: List[List[np.ndarray]]) -> np.ndarray:
    """
    Convert a list of label lists into a single stacked numpy array.
    First transposes the list structure to group labels by image,
    then stacks the arrays.
    
    Parameters
    ----------
    labels : List[List[np.ndarray]]
        List of label lists where each inner list corresponds to one keypoint type
        and contains label arrays for all images
        
    Returns
    -------
    np.ndarray
        Stacked array of shape (n_images, n_keypoints, height, width)
    """
    # First transpose to get labels grouped by image
    labels_by_image = transpose_label_list(labels)
    
    # Stack the arrays for each image
    return np.stack([np.stack(img_labels) for img_labels in labels_by_image]) 