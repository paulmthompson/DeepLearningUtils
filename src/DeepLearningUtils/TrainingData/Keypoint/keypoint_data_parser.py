import os
import cv2
import csv
import numpy as np

from typing import List, Dict, Tuple, Optional

from src.DeepLearningUtils.utils.progress_bar import create_progress_bar
from src.DeepLearningUtils.utils.image_processing.processing import create_gaussian_mask


class KeypointDataParser:
    def __init__(self,
                 data_folder: str,
                 target_resolution: Tuple[int, int] = (480, 640),
                 keypoint_names: Optional[List[str]] = None,
                 gaussian_sigma: Tuple[int, int] = (25, 25)):
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
        """
        self.data_folder = data_folder
        self.target_resolution = target_resolution
        self.keypoint_names = keypoint_names
        self.gaussian_sigma = gaussian_sigma

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
        iterable = create_progress_bar(len(experiment_folders), label='Loading Experiments')

        training_images = []
        training_image_filenames = []

        # Determine keypoint names if not provided
        first_exp_folder = os.path.join(self.data_folder, experiment_folders[0], 'labels')
        if self.keypoint_names is None:
            self.keypoint_names = [folder for folder in os.listdir(first_exp_folder)
                                   if os.path.isdir(os.path.join(first_exp_folder, folder))]

        n_features = len(self.keypoint_names)
        training_labels = [[] for _ in range(n_features)]

        for i, experiment_folder in iterable:
            experiment_path = os.path.join(self.data_folder, experiment_folder)

            print(f'Loading experiment folder: {experiment_folder}')

            # Validate label folders exist
            label_folders_path = os.path.join(experiment_path, 'labels')
            if not os.path.exists(label_folders_path):
                print(f'Skipping {experiment_folder}: No labels folder found')
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

            img_folder = os.path.join(experiment_path, 'images')
            if not os.path.exists(img_folder):
                print(f'Skipping {experiment_folder}: No images folder found')
                continue

            # Get image paths
            image_paths = [os.path.join(img_folder, img) for img in os.listdir(img_folder)
                           if img.endswith(('.png', '.jpg', '.jpeg'))]

            if not image_paths:
                print(f'Skipping {experiment_folder}: No images found')
                continue

            # Process images
            for image_path in image_paths:
                image = cv2.imread(image_path)
                if image is None:
                    print(f'Could not read image: {image_path}')
                    continue

                try:
                    image_resized = cv2.resize(image, (self.target_resolution[1], self.target_resolution[0]),
                                               interpolation=cv2.INTER_AREA)
                    experiment_image_filenames.append(image_path)
                    experiment_images.append(image_resized)
                except Exception as e:
                    print(f'Error processing {image_path}: {str(e)}')

            if not experiment_images:
                print(f'No valid images in {experiment_folder}, skipping')
                continue

            # Get original resolution from first image
            first_img = cv2.imread(image_paths[0])
            original_resolution = (first_img.shape[0], first_img.shape[1])
            print(f'Original resolution: {original_resolution}')

            # Extract frame numbers from image paths
            img_names = [os.path.basename(img) for img in image_paths]
            img_nums = []

            for img_name in img_names:
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_name = os.path.splitext(img_name)[0]

                if img_name.startswith('scene'):
                    img_nums.append(img_name[5:])
                else:
                    img_nums.append(img_name)

            # Process each keypoint type
            for feature_idx, keypoint_name in enumerate(self.keypoint_names):
                keypoint_folder = os.path.join(experiment_path, 'labels', keypoint_name)

                # Find CSV file
                csv_files = [f for f in os.listdir(keypoint_folder) if f.endswith('.csv')]

                if not csv_files:
                    print(f'No CSV file found for {keypoint_name} in {experiment_folder}')
                    continue

                csv_file = csv_files[0]
                keypoint_coords = {}

                # Parse CSV file
                with open(os.path.join(keypoint_folder, csv_file), mode='r') as file:
                    # Skip header
                    next(file)
                    reader = csv.reader(file, delimiter=' ')
                    for row in reader:
                        if not row:
                            continue
                        try:
                            frame_num = int(row[0])
                            x = int(row[1])
                            y = int(row[2])
                            keypoint_coords[frame_num] = [x, y]
                        except (ValueError, IndexError) as e:
                            print(f'Error parsing row {row}: {str(e)}')

                # Generate masks for each frame
                for idx, frame_num in enumerate(img_nums):
                    try:
                        frame_int = int(frame_num)
                        if frame_int in keypoint_coords:
                            mask = self.create_gaussian_mask(
                                original_resolution,
                                self.target_resolution,
                                keypoint_coords[frame_int],
                                self.gaussian_sigma
                            )

                            # Convert to uint8
                            mask = (mask * 255).astype(np.uint8)
                            experiment_labels[feature_idx].append(mask)
                        else:
                            # No keypoint for this frame, create empty mask
                            empty_mask = np.zeros(self.target_resolution, dtype=np.uint8)
                            experiment_labels[feature_idx].append(empty_mask)
                            print(f'No keypoint for frame {frame_num} in {keypoint_name}')
                    except ValueError:
                        print(f'Invalid frame number: {frame_num}')

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
                       gaussian_sigma: Tuple[int, int] = (25, 25)) -> Tuple[
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
        gaussian_sigma=gaussian_sigma
    )

    return parser.parse_data()
