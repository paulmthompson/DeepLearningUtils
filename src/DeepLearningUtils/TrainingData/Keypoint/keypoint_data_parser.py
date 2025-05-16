import os
import cv2
import csv
import numpy as np
import re
import math
from tqdm import tqdm

from typing import List, Dict, Tuple, Optional, Union, Set

from src.DeepLearningUtils.utils.progress_bar import create_progress_bar
from src.DeepLearningUtils.utils.image_processing.processing import create_gaussian_mask


class KeypointDataParser:
    def __init__(self,
                 data_folder: str,
                 target_resolution: Tuple[int, int] = (480, 640),
                 keypoint_names: Optional[List[str]] = None,
                 gaussian_sigma: Tuple[int, int] = (25, 25),
                 csv_delimiter: str = ' ',
                 csv_has_header: bool = True,
                 frame_number_regex: Optional[str] = None,
                 image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
                 images_dir_name: str = 'images',
                 labels_dir_name: str = 'labels',
                 occlusion_markers: Tuple[str, ...] = ('nan', 'NaN', 'NAN', 'None')):
        """
        Parser for keypoint training data.

        Parameters
        ----------
        data_folder : str
            Path to the root folder containing experiment subfolders
        target_resolution : Tuple[int, int]
            Target resolution to resize images to (height, width)
        keypoint_names : List[str], optional
            Names of the keypoint folders to look for. If None, will use all folders found
        gaussian_sigma : Tuple[int, int]
            Sigma values for creating Gaussian masks (y_sigma, x_sigma)
        csv_delimiter : str
            Delimiter character used in CSV files
        csv_has_header : bool
            Whether CSV files have a header row
        frame_number_regex : Optional[str]
            Regex pattern to extract frame number from image filename.
            If None, assumes filename (without extension) is the frame number
        image_extensions : Tuple[str, ...]
            Tuple of valid image file extensions
        images_dir_name : str
            Name of the directory containing images
        labels_dir_name : str
            Name of the directory containing label data
        occlusion_markers : Tuple[str, ...]
            Values in CSV that indicate the keypoint is occluded
        """
        self.data_folder = data_folder
        self.target_resolution = target_resolution
        self.keypoint_names = keypoint_names
        self.gaussian_sigma = gaussian_sigma
        self.csv_delimiter = csv_delimiter
        self.csv_has_header = csv_has_header
        self.frame_number_regex = frame_number_regex
        self.image_extensions = image_extensions
        self.images_dir_name = images_dir_name
        self.labels_dir_name = labels_dir_name
        self.occlusion_markers = occlusion_markers

    def extract_frame_number(self, filename: str) -> Union[int, str]:
        """
        Extract frame number from image filename.
        
        Parameters
        ----------
        filename : str
            Image filename
            
        Returns
        -------
        Union[int, str]
            Frame number as integer if possible, otherwise as string
        """
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        if self.frame_number_regex:
            match = re.search(self.frame_number_regex, name)
            if match:
                frame_num = match.group(1)
            else:
                frame_num = name
        else:
            frame_num = name
            
        # Try to convert to int if possible
        try:
            return int(frame_num)
        except ValueError:
            return frame_num

    def parse_data(self) -> Tuple[List[np.ndarray], List[str], List[List[np.ndarray]]]:
        """
        Parse keypoint data from the data folder.

        Returns
        -------
        Tuple[List[np.ndarray], List[str], List[List[np.ndarray]]]
            Tuple containing:
            - List of training images
            - List of image filenames
            - List of lists of labels (one list per keypoint)
        """
        experiment_folders = [filename for filename in os.listdir(self.data_folder)
                              if os.path.isdir(os.path.join(self.data_folder, filename))]

        # Use tqdm progress bar
        iterable = enumerate(experiment_folders)
        progress = tqdm(iterable, desc='Loading', total=len(experiment_folders), ascii=True, leave=True, position=0)
        iterable = progress

        training_images = []
        training_image_filenames = []

        # Determine keypoint names if not provided
        if experiment_folders:
            first_exp_folder = os.path.join(self.data_folder, experiment_folders[0], self.labels_dir_name)
            if os.path.exists(first_exp_folder) and self.keypoint_names is None:
                self.keypoint_names = [folder for folder in os.listdir(first_exp_folder)
                                      if os.path.isdir(os.path.join(first_exp_folder, folder))]
        
        if not self.keypoint_names:
            print("No keypoint names provided or found in the first experiment folder")
            return [], [], []

        n_features = len(self.keypoint_names)
        training_labels = [[] for _ in range(n_features)]

        for i, experiment_folder in iterable:
            experiment_path = os.path.join(self.data_folder, str(experiment_folder))

            print(f'Loading experiment folder: {experiment_folder}')

            # Validate label folders exist
            label_folders_path = os.path.join(experiment_path, self.labels_dir_name)
            if not os.path.exists(label_folders_path):
                print(f'Skipping {experiment_folder}: No {self.labels_dir_name} folder found')
                continue

            label_folders = [folder for folder in os.listdir(label_folders_path)
                             if os.path.isdir(os.path.join(label_folders_path, folder))]

            # Check if all required label folders exist
            missing_folders = [name for name in self.keypoint_names if name not in label_folders]
            if missing_folders:
                print(f'Skipping {experiment_folder}: Missing label folders: {missing_folders}')
                continue

            # Process images
            experiment_images = []
            experiment_image_filenames = []
            experiment_labels = [[] for _ in range(n_features)]

            img_folder = os.path.join(experiment_path, self.images_dir_name)
            if not os.path.exists(img_folder):
                print(f'Skipping {experiment_folder}: No {self.images_dir_name} folder found')
                continue

            # Get image paths and their frame numbers
            image_paths = [os.path.join(img_folder, img) for img in os.listdir(img_folder)
                          if any(img.lower().endswith(ext) for ext in self.image_extensions)]

            if not image_paths:
                print(f'Skipping {experiment_folder}: No images found')
                continue

            img_names = [os.path.basename(img) for img in image_paths]
            img_frames = [self.extract_frame_number(img_name) for img_name in img_names]
            frame_to_path = dict(zip(img_frames, image_paths))

            # First collect all frames that have labels
            labeled_frames = set()
            keypoint_data = {}
            
            # Process each keypoint type to identify all labeled frames
            for feature_idx, keypoint_name in enumerate(self.keypoint_names):
                keypoint_folder = os.path.join(experiment_path, self.labels_dir_name, keypoint_name)
                
                # Find CSV file
                csv_files = [f for f in os.listdir(keypoint_folder) if f.endswith('.csv')]
                
                if not csv_files:
                    print(f'No CSV file found for {keypoint_name} in {experiment_folder}')
                    continue
                
                csv_file = csv_files[0]
                keypoint_coords = {}
                
                # Parse CSV file to identify labeled frames
                with open(os.path.join(keypoint_folder, csv_file), mode='r') as file:
                    # Skip header if needed
                    if self.csv_has_header:
                        next(file)
                    
                    reader = csv.reader(file, delimiter=self.csv_delimiter)
                    for row in reader:
                        if not row or len(row) < 3:
                            continue
                        
                        try:
                            # Get frame ID
                            frame_id = row[0].strip()
                            try:
                                frame_num = int(frame_id)
                            except ValueError:
                                frame_num = frame_id
                            
                            # Check if coordinates are marked as occluded
                            is_occluded = (
                                str(row[1]).strip() in self.occlusion_markers or 
                                str(row[2]).strip() in self.occlusion_markers
                            )
                            
                            if is_occluded:
                                # Store None to indicate occlusion
                                keypoint_coords[frame_num] = None
                            else:
                                try:
                                    x = int(float(row[1]))
                                    y = int(float(row[2]))
                                    keypoint_coords[frame_num] = [x, y]
                                except (ValueError, IndexError):
                                    print(f'Invalid coordinate in row {row} in {csv_file}')
                                    continue
                            
                            # Mark this frame as having a label entry
                            labeled_frames.add(frame_num)
                            
                        except Exception as e:
                            print(f'Error parsing row {row} in {csv_file}: {str(e)}')
                
                keypoint_data[keypoint_name] = keypoint_coords

            # Get only images that have label entries
            valid_frames = [frame for frame in img_frames if frame in labeled_frames]
            if not valid_frames:
                print(f'No labeled frames found in {experiment_folder}')
                continue
            
            print(f'Found {len(valid_frames)} labeled frames out of {len(img_frames)} total images')
            
            # Get first valid image to determine original resolution
            if not valid_frames:
                continue
                
            first_valid_path = frame_to_path[valid_frames[0]]
            first_img = cv2.imread(first_valid_path)
            original_resolution = (first_img.shape[0], first_img.shape[1])
            print(f'Original resolution: {original_resolution}')
            
            # Now process only the labeled images
            for frame_num in valid_frames:
                image_path = frame_to_path[frame_num]
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f'Could not read image: {image_path}')
                    continue
                
                try:
                    image_resized = cv2.resize(image, (self.target_resolution[1], self.target_resolution[0]),
                                           interpolation=cv2.INTER_AREA)
                    experiment_image_filenames.append(image_path)
                    experiment_images.append(image_resized)
                    
                    # Create label masks for this image
                    for feature_idx, keypoint_name in enumerate(self.keypoint_names):
                        keypoint_coords = keypoint_data.get(keypoint_name, {})
                        
                        if frame_num in keypoint_coords:
                            coords = keypoint_coords[frame_num]
                            
                            if coords is None:
                                # Occluded keypoint - create empty mask
                                mask = np.zeros(self.target_resolution, dtype=np.uint8)
                                experiment_labels[feature_idx].append(mask)
                                print(f'Occluded keypoint for frame {frame_num} in {keypoint_name}')
                            else:
                                # Create Gaussian mask for visible keypoint
                                mask = create_gaussian_mask(
                                    original_resolution,
                                    self.target_resolution,
                                    coords,
                                    self.gaussian_sigma
                                )
                                mask = (mask * 255).astype(np.uint8)
                                experiment_labels[feature_idx].append(mask)
                        else:
                            # Should not happen since we're only processing labeled frames
                            mask = np.zeros(self.target_resolution, dtype=np.uint8)
                            experiment_labels[feature_idx].append(mask)
                            print(f'Warning: No keypoint data for frame {frame_num} in {keypoint_name}')
                                
                except Exception as e:
                    print(f'Error processing {image_path}: {str(e)}')

            # Add experiment data to training data
            training_images.extend(experiment_images)
            training_image_filenames.extend(experiment_image_filenames)

            for i in range(n_features):
                if experiment_labels[i]:
                    training_labels[i].extend(experiment_labels[i])

        print(f'Loaded {len(training_images)} images and {n_features} keypoint types')
        return training_images, training_image_filenames, training_labels


def load_keypoint_data(data_folder: str,
                       target_resolution: Tuple[int, int] = (480, 640),
                       keypoint_names: Optional[List[str]] = None,
                       gaussian_sigma: Tuple[int, int] = (25, 25),
                       csv_delimiter: str = ' ',
                       csv_has_header: bool = True,
                       frame_number_regex: Optional[str] = None,
                       image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
                       images_dir_name: str = 'images',
                       labels_dir_name: str = 'labels',
                       occlusion_markers: Tuple[str, ...] = ('nan', 'NaN', 'NAN', 'None')) -> Tuple[
    List[np.ndarray], List[str], List[List[np.ndarray]]]:
    """
    Load keypoint training data from a folder.

    Parameters
    ----------
    data_folder : str
        Path to the root folder containing experiment subfolders
    target_resolution : Tuple[int, int]
        Target resolution to resize images to (height, width)
    keypoint_names : List[str], optional
        Names of the keypoint folders to look for. If None, will use all folders found
    gaussian_sigma : Tuple[int, int]
        Sigma values for creating Gaussian masks (y_sigma, x_sigma)
    csv_delimiter : str
        Delimiter character used in CSV files
    csv_has_header : bool
        Whether CSV files have a header row
    frame_number_regex : Optional[str]
        Regex pattern to extract frame number from image filename.
        If None, assumes filename (without extension) is the frame number
    image_extensions : Tuple[str, ...]
        Tuple of valid image file extensions
    images_dir_name : str
        Name of the directory containing images
    labels_dir_name : str
        Name of the directory containing label data
    occlusion_markers : Tuple[str, ...]
        Values in CSV that indicate the keypoint is occluded

    Returns
    -------
    Tuple[List[np.ndarray], List[str], List[List[np.ndarray]]]
        Tuple containing:
        - List of training images
        - List of image filenames
        - List of lists of labels (one list per keypoint)
    """
    parser = KeypointDataParser(
        data_folder=data_folder,
        target_resolution=target_resolution,
        keypoint_names=keypoint_names,
        gaussian_sigma=gaussian_sigma,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
        frame_number_regex=frame_number_regex,
        image_extensions=image_extensions,
        images_dir_name=images_dir_name,
        labels_dir_name=labels_dir_name,
        occlusion_markers=occlusion_markers
    )

    return parser.parse_data()
