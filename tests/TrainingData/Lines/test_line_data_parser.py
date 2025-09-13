import os
import cv2
import csv
import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from src.DeepLearningUtils.TrainingData.Lines.line_data_parser import load_line_data


@pytest.fixture
def simulated_line_data():
    """
    Creates simulated line data in the expected folder structure:

    temp_dir/
    ├── experiment_1/
    │   ├── images/
    │   │   ├── img0000001.png
    │   │   ├── img0000002.png
    │   │   └── ...
    │   └── labels/
    │       ├── line_category_1/
    │       │   ├── 0000001.csv
    │       │   ├── 0000002.csv
    │       │   └── ...
    │       └── line_category_2/
    │           ├── 0000001.csv
    │           ├── 0000002.csv
    │           └── ...
    ├── experiment_2/
    │   └── ... (same structure)
    └── experiment_3/
        └── ... (same structure)
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Configuration
        experiments = ['experiment_1', 'experiment_2', 'experiment_3']
        label_categories = ['line_category_1', 'line_category_2']
        num_samples_per_experiment = 10
        image_size = (256, 256, 3)  # Height, Width, Channels

        # Store information for validation
        created_data = {
            'temp_dir': temp_dir,
            'experiments': experiments,
            'label_categories': label_categories,
            'num_samples': num_samples_per_experiment,
            'image_files': {},
            'label_files': {}
        }

        for exp in experiments:
            exp_path = os.path.join(temp_dir, exp)
            images_path = os.path.join(exp_path, 'images')
            labels_path = os.path.join(exp_path, 'labels')

            # Create directories
            os.makedirs(images_path)
            os.makedirs(labels_path)

            created_data['image_files'][exp] = []
            created_data['label_files'][exp] = {}

            # Create images and corresponding labels
            for i in range(1, num_samples_per_experiment + 1):
                frame_num = f"{i:07d}"  # Zero-padded to 7 digits

                # Create image with some simple pattern
                image = np.zeros(image_size, dtype=np.uint8)

                # Add some visual content (diagonal lines with different colors for each experiment)
                if exp == 'experiment_1':
                    # Red diagonal line
                    cv2.line(image, (0, 0), (255, 255), (0, 0, 255), thickness=3)
                elif exp == 'experiment_2':
                    # Green diagonal line
                    cv2.line(image, (0, 255), (255, 0), (0, 255, 0), thickness=3)
                else:  # experiment_3
                    # Blue horizontal line
                    cv2.line(image, (0, 128), (255, 128), (255, 0, 0), thickness=3)

                # Add some noise to make it more realistic
                noise = np.random.randint(0, 50, image_size, dtype=np.uint8)
                image = cv2.add(image, noise)

                # Save image with prefix
                image_filename = f"img{frame_num}.png"
                image_path = os.path.join(images_path, image_filename)
                cv2.imwrite(image_path, image)
                created_data['image_files'][exp].append(image_filename)

                # Create label folders and CSV files for each category
                for category in label_categories:
                    category_path = os.path.join(labels_path, category)
                    os.makedirs(category_path, exist_ok=True)

                    if category not in created_data['label_files'][exp]:
                        created_data['label_files'][exp][category] = []

                    # Generate sample line coordinates
                    line_coords = generate_sample_line_coords(category, i)

                    # Save CSV file without prefix (just frame number)
                    csv_filename = f"{frame_num}.csv"
                    csv_path = os.path.join(category_path, csv_filename)

                    with open(csv_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        for x, y in line_coords:
                            writer.writerow([x, y])

                    created_data['label_files'][exp][category].append(csv_filename)

        yield created_data

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def generate_sample_line_coords(category: str, sample_index: int) -> List[Tuple[float, float]]:
    """
    Generate sample line coordinates for testing.

    Parameters
    ----------
    category : str
        The label category name
    sample_index : int
        The sample index (1-based)

    Returns
    -------
    List[Tuple[float, float]]
        List of (x, y) coordinate pairs representing a line
    """
    np.random.seed(sample_index * 42)  # Deterministic but varied

    num_points = 20  # Number of points in the line

    if category == 'line_category_1':
        # Create a diagonal line with some variation
        x_coords = np.linspace(10, 245, num_points)
        y_coords = np.linspace(10, 245, num_points)

        # Add some noise to make it more realistic
        noise_x = np.random.normal(0, 2, num_points)
        noise_y = np.random.normal(0, 2, num_points)

        x_coords += noise_x
        y_coords += noise_y

    else:  # line_category_2
        # Create a curved line
        t = np.linspace(0, 2 * np.pi, num_points)
        x_coords = 128 + 50 * np.cos(t) + np.random.normal(0, 1, num_points)
        y_coords = 128 + 50 * np.sin(t) + np.random.normal(0, 1, num_points)

    # Ensure coordinates are within image bounds
    x_coords = np.clip(x_coords, 0, 255)
    y_coords = np.clip(y_coords, 0, 255)

    return list(zip(x_coords.astype(float), y_coords.astype(float)))


def test_load_line_data_basic_functionality(simulated_line_data):
    """
    Test basic functionality of load_line_data with simulated data.
    """
    data_info = simulated_line_data
    temp_dir = data_info['temp_dir']

    # Test loading all experiments
    df = load_line_data(
        data_folder=temp_dir,
        target_resolution=(256, 256),
        image_prefix="img"
    )

    # Verify DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Check expected columns
    expected_columns = ['folder_id', 'image_name', 'image', 'labels']
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

    # Verify we have data from all experiments
    unique_folders = df['folder_id'].unique()
    expected_folders = set(data_info['experiments'])
    assert set(unique_folders) == expected_folders

    # Verify we have the expected number of samples per experiment
    for exp in data_info['experiments']:
        exp_data = df[df['folder_id'] == exp]
        assert len(exp_data) == data_info['num_samples'], f"Experiment {exp} has wrong number of samples"

    # Check image data
    sample_row = df.iloc[0]
    assert isinstance(sample_row['image'], np.ndarray)
    assert sample_row['image'].shape == (256, 256, 3), "Image should be resized to target resolution"

    # Check labels data
    assert isinstance(sample_row['labels'], dict)
    for category in data_info['label_categories']:
        assert category in sample_row['labels'], f"Missing label category: {category}"
        coords = sample_row['labels'][category]
        assert isinstance(coords, np.ndarray), "Label coordinates should be numpy array"
        assert coords.shape[1] == 2, "Coordinates should have 2 columns (x, y)"


def test_load_line_data_specific_experiments(simulated_line_data):
    """
    Test loading specific experiments only.
    """
    data_info = simulated_line_data
    temp_dir = data_info['temp_dir']

    # Test loading only specific experiments
    specific_experiments = ['experiment_1', 'experiment_3']
    df = load_line_data(
        data_folder=temp_dir,
        experiment_folders=specific_experiments,
        image_prefix="img"
    )

    # Verify only specified experiments are loaded
    unique_folders = set(df['folder_id'].unique())
    assert unique_folders == set(specific_experiments)

    # Verify experiment_2 is not included
    assert 'experiment_2' not in unique_folders


def test_load_line_data_different_resolution(simulated_line_data):
    """
    Test loading with different target resolution.
    """
    data_info = simulated_line_data
    temp_dir = data_info['temp_dir']

    # Test with different resolution
    target_res = (128, 128)
    df = load_line_data(
        data_folder=temp_dir,
        target_resolution=target_res,
        experiment_folders=['experiment_1'],
        image_prefix="img"
    )

    # Verify image is resized correctly
    sample_image = df.iloc[0]['image']
    assert sample_image.shape[:2] == target_res, f"Image should be resized to {target_res}"


def test_load_line_data_without_prefix(simulated_line_data):
    """
    Test what happens when image_prefix is not specified.
    Since image files have "img" prefix (e.g., img0000001.png) but CSV files don't (e.g., 0000001.csv),
    the function should not be able to match them and should return an empty DataFrame.
    """
    data_info = simulated_line_data
    temp_dir = data_info['temp_dir']

    # Test without specifying image prefix
    # This should result in no matches since filenames don't align
    df = load_line_data(
        data_folder=temp_dir,
        experiment_folders=['experiment_1'],
        image_prefix=None
    )

    # The function should return an empty DataFrame since files can't be matched
    assert isinstance(df, pd.DataFrame)
    assert df.empty, "DataFrame should be empty when image_prefix is not specified and filenames don't match"


def test_load_line_data_nonexistent_folder():
    """
    Test error handling for nonexistent folders.
    """
    # Test with nonexistent data folder
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        load_line_data(data_folder="/nonexistent/path")


def test_load_line_data_nonexistent_experiment_folder(simulated_line_data):
    """
    Test error handling for specifying nonexistent experiment folders.
    """
    data_info = simulated_line_data
    temp_dir = data_info['temp_dir']

    # Test with nonexistent experiment folder
    with pytest.raises(ValueError):
        load_line_data(
            data_folder=temp_dir,
            experiment_folders=['nonexistent_experiment'],
            image_prefix="img"
        )
