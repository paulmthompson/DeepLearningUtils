import os
import cv2
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Set
from pathlib import Path

from src.DeepLearningUtils.utils.progress_bar import create_progress_bar


def load_line_data(
    data_folder: str,
    target_resolution: Tuple[int, int] = (256, 256),
    image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
    images_dir_name: str = 'images',
    labels_dir_name: str = 'labels',
    csv_delimiter: str = ',',
) -> pd.DataFrame:
    """
    Load line data from a folder structure into a pandas DataFrame.
    
    Parameters
    ----------
    data_folder : str
        Path to the root folder containing experiment subfolders
    target_resolution : Tuple[int, int]
        Target resolution to resize images to (height, width)
    image_extensions : Tuple[str, ...]
        Tuple of valid image file extensions
    images_dir_name : str
        Name of the directory containing images
    labels_dir_name : str
        Name of the directory containing label data
    csv_delimiter : str
        Delimiter character used in CSV files
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - folder_id: experiment folder name
        - image_name: name of the image file
        - image: resized image array
        - labels: dictionary mapping label names to coordinate arrays
    """
    experiment_folders = [f for f in os.listdir(data_folder) 
                         if os.path.isdir(os.path.join(data_folder, f))]
    
    data = []
    
    # Use tqdm progress bar
    iterable = enumerate(experiment_folders)
    progress = tqdm(iterable, desc='Loading', total=len(experiment_folders), 
                   ascii=True, leave=True, position=0)
    
    for i, experiment_folder in progress:
        print(f'Loading experiment folder: {experiment_folder}')
        
        # Get paths
        exp_path = os.path.join(data_folder, experiment_folder)
        label_folders = os.listdir(os.path.join(exp_path, labels_dir_name))
        img_folder = os.path.join(exp_path, images_dir_name)
        
        # Get image paths
        image_paths = [os.path.join(img_folder, img) for img in os.listdir(img_folder)
                      if any(img.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_paths:
            print(f'No images found in {experiment_folder}')
            continue
            
        # Load and resize all images first
        images_dict = {}
        old_resolution = None
        
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f'Could not read image: {image_path}')
                continue
                
            if old_resolution is None:
                old_resolution = (image.shape[0], image.shape[1])
                
            image_resized = cv2.resize(image, (target_resolution[1], target_resolution[0]),
                                     interpolation=cv2.INTER_AREA)
            
            # Extract image name (remove extension)
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            images_dict[img_name] = image_resized
        
        print(f'Original resolution: {old_resolution}')
        
        # Process each image
        for img_name, image in images_dict.items():
            labels_dict = {}
            
            for label_folder in label_folders:
                label_path = os.path.join(exp_path, labels_dir_name, label_folder, 
                                        f"{img_name}.csv")
                
                if os.path.exists(label_path):
                    try:
                        # Load coordinates and scale to new resolution
                        coords = np.loadtxt(label_path, delimiter=csv_delimiter)
                        if len(coords) > 0:
                            # Scale coordinates to new resolution
                            coords[:, 0] *= target_resolution[1] / old_resolution[1]
                            coords[:, 1] *= target_resolution[0] / old_resolution[0]
                            labels_dict[label_folder] = coords
                    except Exception as e:
                        print(f'Error loading {label_path}: {str(e)}')
            
            # Only add to dataframe if at least one label was found
            if labels_dict:
                data.append({
                    'folder_id': experiment_folder,
                    'image_name': img_name,
                    'image': image,
                    'labels': labels_dict
                })
    
    return pd.DataFrame(data)


def prepare_line_generator_data(
    df: pd.DataFrame,
    label_names: Optional[List[str]] = None,
    validation_folders: Optional[List[str]] = None,
    return_numpy: bool = True
) -> Union[Tuple[List[np.ndarray], List[List[np.ndarray]]],
           Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare data from DataFrame for use with LineDataGenerator.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from load_line_data
    label_names : Optional[List[str]]
        Names of labels to include. If None, uses all labels found
    validation_folders : Optional[List[str]]
        List of folder_ids to use for validation. If None, uses all data
    return_numpy : bool
        If True, returns numpy arrays instead of lists
        
    Returns
    -------
    Union[Tuple[List[np.ndarray], List[List[np.ndarray]]],
          Tuple[np.ndarray, np.ndarray]]
        If return_numpy is False:
        - List of images
        - List of lists of line coordinates (one list per image)
        
        If return_numpy is True:
        - Stacked numpy array of images
        - List of lists of line coordinates (one list per image)
    """
    # Filter by validation folders if specified
    if validation_folders is not None:
        df = df[~df['folder_id'].isin(validation_folders)]
    
    # Get label names if not specified
    if label_names is None:
        # Get all unique label names from the first row
        label_names = list(df.iloc[0]['labels'].keys())
    
    # Prepare data
    images = []
    labels = []
    
    for _, row in df.iterrows():
        # Get image
        images.append(row['image'])
        
        # Get labels for this image
        image_labels = []
        for label_name in label_names:
            if label_name in row['labels']:
                image_labels.append(row['labels'][label_name])
            else:
                # If label is missing, create empty array
                image_labels.append(np.zeros((0, 2)))
        
        labels.append(image_labels)
    
    if return_numpy:
        # Convert images to numpy array
        images = np.stack(images)
        # Labels remain as list of lists since they may have different lengths
        
    return images, labels


def load_line_data_for_generator(
    data_folder: str,
    target_resolution: Tuple[int, int] = (256, 256),
    label_names: Optional[List[str]] = None,
    validation_folders: Optional[List[str]] = None,
    image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
    images_dir_name: str = 'images',
    labels_dir_name: str = 'labels',
    csv_delimiter: str = ',',
    return_numpy: bool = True
) -> Union[Tuple[List[np.ndarray], List[List[np.ndarray]]],
           Tuple[np.ndarray, np.ndarray]]:
    """
    Load line data and prepare it for use with LineDataGenerator.
    
    Parameters
    ----------
    data_folder : str
        Path to the root folder containing experiment subfolders
    target_resolution : Tuple[int, int]
        Target resolution to resize images to (height, width)
    label_names : Optional[List[str]]
        Names of labels to include. If None, uses all labels found
    validation_folders : Optional[List[str]]
        List of folder_ids to use for validation. If None, uses all data
    image_extensions : Tuple[str, ...]
        Tuple of valid image file extensions
    images_dir_name : str
        Name of the directory containing images
    labels_dir_name : str
        Name of the directory containing label data
    csv_delimiter : str
        Delimiter character used in CSV files
    return_numpy : bool
        If True, returns numpy arrays instead of lists
        
    Returns
    -------
    Union[Tuple[List[np.ndarray], List[List[np.ndarray]]],
          Tuple[np.ndarray, np.ndarray]]
        If return_numpy is False:
        - List of images
        - List of lists of line coordinates (one list per image)
        
        If return_numpy is True:
        - Stacked numpy array of images
        - List of lists of line coordinates (one list per image)
    """
    # Load data into DataFrame
    df = load_line_data(
        data_folder=data_folder,
        target_resolution=target_resolution,
        image_extensions=image_extensions,
        images_dir_name=images_dir_name,
        labels_dir_name=labels_dir_name,
        csv_delimiter=csv_delimiter
    )
    
    # Prepare data for generator
    return prepare_line_generator_data(
        df=df,
        label_names=label_names,
        validation_folders=validation_folders,
        return_numpy=return_numpy
    ) 