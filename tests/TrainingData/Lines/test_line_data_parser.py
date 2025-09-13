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


def test_load_line_data_require_all_labels(simulated_line_data):
    """
    Test the require_all_labels feature to ensure only frames with all specified labels are loaded.
    """
    data_info = simulated_line_data
    temp_dir = data_info['temp_dir']

    # Test with requiring all labels - should load all frames since our fixture has all labels
    df_all_required = load_line_data(
        data_folder=temp_dir,
        image_prefix="img",
        require_all_labels=['line_category_1', 'line_category_2']
    )

    # Should have all samples since all frames have both labels in our fixture
    expected_total = len(data_info['experiments']) * data_info['num_samples']
    assert len(df_all_required) == expected_total

    # Verify that all loaded frames have both required labels
    for _, row in df_all_required.iterrows():
        labels = row['labels']
        assert 'line_category_1' in labels, "All frames should have line_category_1"
        assert 'line_category_2' in labels, "All frames should have line_category_2"


def test_load_line_data_require_all_labels_missing(simulated_line_data):
    """
    Test require_all_labels when requiring a label that doesn't exist.
    Should return empty DataFrame or skip experiments.
    """
    data_info = simulated_line_data
    temp_dir = data_info['temp_dir']

    # Test with requiring a non-existent label
    df_missing = load_line_data(
        data_folder=temp_dir,
        image_prefix="img",
        require_all_labels=['line_category_1', 'nonexistent_category']
    )

    # Should be empty since 'nonexistent_category' doesn't exist
    assert df_missing.empty, "DataFrame should be empty when requiring non-existent labels"


def test_load_line_data_require_all_labels_partial_overlap():
    """
    Test require_all_labels with a custom fixture that has partial label overlap.
    """
    # Create a temporary directory with partial label overlap
    temp_dir = tempfile.mkdtemp()

    try:
        # Create experiment with partial label overlap
        exp_path = os.path.join(temp_dir, 'test_experiment')
        images_path = os.path.join(exp_path, 'images')
        labels_path = os.path.join(exp_path, 'labels')

        os.makedirs(images_path)
        os.makedirs(os.path.join(labels_path, 'label_A'))
        os.makedirs(os.path.join(labels_path, 'label_B'))

        # Create 5 images
        for i in range(1, 6):
            frame_num = f"{i:07d}"

            # Create image
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image_path = os.path.join(images_path, f"img{frame_num}.png")
            cv2.imwrite(image_path, image)

            # Create label_A for frames 1, 2, 3, 4 (not 5)
            if i <= 4:
                csv_path_A = os.path.join(labels_path, 'label_A', f"{frame_num}.csv")
                with open(csv_path_A, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([10, 10])
                    writer.writerow([20, 20])

            # Create label_B for frames 2, 3, 4, 5 (not 1)
            if i >= 2:
                csv_path_B = os.path.join(labels_path, 'label_B', f"{frame_num}.csv")
                with open(csv_path_B, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([15, 15])
                    writer.writerow([25, 25])

        # Test without require_all_labels - should load all 5 frames
        df_all = load_line_data(
            data_folder=temp_dir,
            image_prefix="img"
        )
        assert len(df_all) == 5, "Should load all 5 frames without filtering"

        # Test with require_all_labels - should only load frames 2, 3, 4 (intersection)
        df_filtered = load_line_data(
            data_folder=temp_dir,
            image_prefix="img",
            require_all_labels=['label_A', 'label_B']
        )
        assert len(df_filtered) == 3, "Should only load 3 frames that have both labels"

        # Verify that all loaded frames have both labels
        for _, row in df_filtered.iterrows():
            labels = row['labels']
            assert 'label_A' in labels, "All filtered frames should have label_A"
            assert 'label_B' in labels, "All filtered frames should have label_B"

        # Verify the specific frame names that were loaded
        loaded_frame_names = set(df_filtered['image_name'].values)
        expected_frames = {'0000002', '0000003', '0000004'}
        assert loaded_frame_names == expected_frames, f"Expected frames {expected_frames}, got {loaded_frame_names}"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_line_data_require_all_labels_per_experiment():
    """
    Test that require_all_labels filtering is applied per experiment, not globally.
    """
    temp_dir = tempfile.mkdtemp()

    try:
        # Create two experiments with different label patterns
        for exp_name, label_pattern in [('exp1', 'A_only'), ('exp2', 'both')]:
            exp_path = os.path.join(temp_dir, exp_name)
            images_path = os.path.join(exp_path, 'images')
            labels_path = os.path.join(exp_path, 'labels')

            os.makedirs(images_path)
            os.makedirs(os.path.join(labels_path, 'label_A'))
            os.makedirs(os.path.join(labels_path, 'label_B'))

            # Create 3 images per experiment
            for i in range(1, 4):
                frame_num = f"{i:07d}"

                # Create image
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                image_path = os.path.join(images_path, f"img{frame_num}.png")
                cv2.imwrite(image_path, image)

                # Create label_A for all frames in both experiments
                csv_path_A = os.path.join(labels_path, 'label_A', f"{frame_num}.csv")
                with open(csv_path_A, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([10, 10])
                    writer.writerow([20, 20])

                # Create label_B only for exp2
                if label_pattern == 'both':
                    csv_path_B = os.path.join(labels_path, 'label_B', f"{frame_num}.csv")
                    with open(csv_path_B, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([15, 15])
                        writer.writerow([25, 25])

        # Test with require_all_labels - should only load from exp2
        df_filtered = load_line_data(
            data_folder=temp_dir,
            image_prefix="img",
            require_all_labels=['label_A', 'label_B']
        )

        # Should only have frames from exp2 (3 frames)
        assert len(df_filtered) == 3, "Should only load frames from exp2"

        # Verify all frames are from exp2
        unique_folders = df_filtered['folder_id'].unique()
        assert len(unique_folders) == 1, "Should only have frames from one experiment"
        assert unique_folders[0] == 'exp2', "Should only have frames from exp2"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

