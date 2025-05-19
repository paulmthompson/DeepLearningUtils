import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import distance_transform_edt
import cv2


def interpolate_points(points: np.ndarray, max_distance: int) -> np.ndarray:
    """
    Interpolate between points if they are too far apart.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (n_points, 2) containing (x,y) coordinates
    max_distance : int
        Maximum allowed distance between consecutive points
        
    Returns
    -------
    np.ndarray
        Array of interpolated points
    """
    if len(points) < 2:
        return points
        
    interpolated = [points[0]]
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dist = np.linalg.norm(p2 - p1)
        
        if dist > max_distance:
            # Calculate number of points to insert
            n_points = int(np.ceil(dist / max_distance))
            # Generate interpolated points
            for t in np.linspace(0, 1, n_points + 1)[1:]:
                new_point = p1 + t * (p2 - p1)
                interpolated.append(new_point)
        else:
            interpolated.append(p2)
            
    return np.array(interpolated)


def create_line_mask(
    points: np.ndarray,
    height: int,
    width: int,
    line_width: int = 1
) -> np.ndarray:
    """
    Create a binary mask for a line.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (n_points, 2) containing (x,y) coordinates
    height : int
        Height of the output mask
    width : int
        Width of the output mask
    line_width : int
        Width of the line in pixels
        
    Returns
    -------
    np.ndarray
        Binary mask of shape (height, width)
    """
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Convert points to integer coordinates
    points = points.astype(int)
    
    # Draw line segments
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # Clip coordinates to image bounds
        x1 = np.clip(x1, 0, width - 1)
        y1 = np.clip(y1, 0, height - 1)
        x2 = np.clip(x2, 0, width - 1)
        y2 = np.clip(y2, 0, height - 1)
        
        # Draw line segment
        cv2.line(mask, (x1, y1), (x2, y2), 1.0, line_width)
    
    return mask


def create_distance_map(
    mask: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Create a distance map from a binary mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape (height, width)
    normalize : bool
        Whether to normalize the distance map to [0,1]
        
    Returns
    -------
    np.ndarray
        Distance map of shape (height, width)
    """
    # Calculate distance transform
    dist_map = distance_transform_edt(1 - mask)
    
    if normalize:
        # Normalize to [0,1]
        max_dist = np.max(dist_map)
        if max_dist > 0:
            dist_map = dist_map / max_dist
            
    return dist_map


def validate_line_data(
    images: np.ndarray,
    labels: List[List[np.ndarray]],
    n_lines: int
) -> None:
    """
    Validate line data format and dimensions.
    
    Parameters
    ----------
    images : np.ndarray
        Array of images of shape (n_samples, height, width, channels)
    labels : List[List[np.ndarray]]
        List of n_samples elements, each containing n_lines arrays of points
    n_lines : int
        Expected number of lines per image
        
    Raises
    ------
    ValueError
        If data format is invalid
    """
    if len(images.shape) != 4:
        raise ValueError(f"Images must be 4D array (n_samples, height, width, channels), got shape {images.shape}")
        
    if len(labels) != images.shape[0]:
        raise ValueError(f"Number of images ({images.shape[0]}) must match number of label sets ({len(labels)})")
        
    for i, image_labels in enumerate(labels):
        if len(image_labels) != n_lines:
            raise ValueError(f"Image {i} has {len(image_labels)} lines, expected {n_lines}")
            
        for j, line in enumerate(image_labels):
            if not isinstance(line, np.ndarray) or line.ndim != 2 or line.shape[1] != 2:
                raise ValueError(f"Line {j} in image {i} must be a 2D array of shape (n_points, 2)") 