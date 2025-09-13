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
    experiment_folders: Optional[List[str]] = None,
    image_prefix: Optional[str] = None,
    require_all_labels: Optional[List[str]] = None,
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
    experiment_folders : Optional[List[str]]
        List of experiment folder names to include. If None, all subfolders will be used.
    image_prefix : Optional[str]
        Prefix to remove from image filenames to match CSV filenames.
        For example, if images are named "img00001.png" and CSVs are "00001.csv",
        set image_prefix="img"
    require_all_labels : Optional[List[str]]
        If specified, only load frames that have all of these label categories present.
        This filtering is applied per experiment - a frame is only loaded if ALL
        specified labels exist for that frame within the experiment.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - folder_id: experiment folder name
        - image_name: name of the image file
        - image: resized image array
        - labels: dictionary mapping label names to coordinate arrays
    """
    # Get experiment folders to process
    if experiment_folders is None:
        experiment_folders = [f for f in os.listdir(data_folder) 
                            if os.path.isdir(os.path.join(data_folder, f))]
    else:
        # Validate that specified folders exist
        for folder in experiment_folders:
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                raise ValueError(f"Specified experiment folder does not exist: {folder}")
    
    data = []
    
    # Use tqdm progress bar
    iterable = enumerate(experiment_folders)
    progress = tqdm(iterable, desc='Loading', total=len(experiment_folders), 
                   ascii=True, leave=True, position=0)
    
    for i, experiment_folder in progress:
        print(f'Loading experiment folder: {experiment_folder}')
        
        # Get paths
        exp_path = os.path.join(data_folder, experiment_folder)
        labels_path = os.path.join(exp_path, labels_dir_name)
        label_folders = os.listdir(labels_path)
        img_folder = os.path.join(exp_path, images_dir_name)
        
        # Get image paths
        image_paths = [os.path.join(img_folder, img) for img in os.listdir(img_folder)
                      if any(img.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_paths:
            print(f'No images found in {experiment_folder}')
            continue

        # If require_all_labels is specified, find frames that have all required labels
        valid_frame_names = None
        if require_all_labels:
            print(f'Filtering frames that have all required labels: {require_all_labels}')

            # Check that all required label folders exist
            missing_label_folders = [label for label in require_all_labels
                                   if label not in label_folders]
            if missing_label_folders:
                print(f'Warning: Required label folders not found in {experiment_folder}: {missing_label_folders}')
                continue

            # Find frames that exist in ALL required label folders
            frame_sets = []
            for required_label in require_all_labels:
                label_folder_path = os.path.join(labels_path, required_label)
                if os.path.exists(label_folder_path):
                    # Get all CSV files in this label folder
                    csv_files = [f for f in os.listdir(label_folder_path)
                               if f.endswith('.csv')]
                    # Extract frame names (remove .csv extension)
                    frame_names = {os.path.splitext(f)[0] for f in csv_files}
                    frame_sets.append(frame_names)
                else:
                    print(f'Warning: Label folder {required_label} not found in {experiment_folder}')
                    frame_sets.append(set())

            # Find intersection of all frame sets
            if frame_sets:
                valid_frame_names = set.intersection(*frame_sets)
                print(f'Found {len(valid_frame_names)} frames with all required labels in {experiment_folder}')
            else:
                valid_frame_names = set()

            if not valid_frame_names:
                print(f'No frames found with all required labels in {experiment_folder}')
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
            
            # Extract image name (remove extension and optional prefix)
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            if image_prefix and img_name.startswith(image_prefix):
                img_name = img_name[len(image_prefix):]

            # If require_all_labels is specified, only include valid frames
            if valid_frame_names is not None and img_name not in valid_frame_names:
                continue

            images_dict[img_name] = image_resized
        
        print(f'Original resolution: {old_resolution}')
        print(f'Loading {len(images_dict)} images from {experiment_folder}')

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
                            # Handle case where there's only one point (1D array)
                            if coords.ndim == 1:
                                coords = coords.reshape(1, -1)
                            # Scale coordinates to new resolution
                            coords[:, 0] *= target_resolution[1] / old_resolution[1]
                            coords[:, 1] *= target_resolution[0] / old_resolution[0]
                            labels_dict[label_folder] = coords
                    except Exception as e:
                        print(f'Error loading {label_path}: {str(e)}')
            
            # Add validation for require_all_labels
            if require_all_labels:
                # Check that all required labels are present
                missing_labels = [label for label in require_all_labels
                                if label not in labels_dict]
                if missing_labels:
                    print(f'Warning: Frame {img_name} missing required labels: {missing_labels}')
                    continue

            # Only add to dataframe if at least one label was found
            # (or if require_all_labels is specified, all required labels are present)
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
        Names of labels to include. If None, uses all labels found.
        All specified labels must be present for an entry to be included.
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
        # Check if all required labels are present
        if not all(label_name in row['labels'] for label_name in label_names):
            continue
            
        # Get image
        images.append(row['image'])
        
        # Get labels for this image
        image_labels = []
        for label_name in label_names:
            image_labels.append(row['labels'][label_name])
        
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
    experiment_folders: Optional[List[str]] = None,
    image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
    images_dir_name: str = 'images',
    labels_dir_name: str = 'labels',
    csv_delimiter: str = ',',
    return_numpy: bool = True,
    image_prefix: Optional[str] = None,
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
        Names of labels to include. If None, uses all labels found.
        All specified labels must be present for an entry to be included.
    validation_folders : Optional[List[str]]
        List of folder_ids to use for validation. If None, uses all data
    experiment_folders : Optional[List[str]]
        List of experiment folder names to include. If None, all subfolders will be used.
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
    image_prefix : Optional[str]
        Prefix to remove from image filenames to match CSV filenames.
        For example, if images are named "img00001.png" and CSVs are "00001.csv",
        set image_prefix="img"
        
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
        csv_delimiter=csv_delimiter,
        experiment_folders=experiment_folders,
        image_prefix=image_prefix
    )
    
    # Prepare data for generator
    return prepare_line_generator_data(
        df=df,
        label_names=label_names,
        validation_folders=validation_folders,
        return_numpy=return_numpy
    )
