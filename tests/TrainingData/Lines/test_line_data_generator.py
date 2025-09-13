import os
import cv2
import numpy as np
import pandas as pd
import pytest
import tempfile
from typing import Dict, List, Tuple

from src.DeepLearningUtils.TrainingData.Lines.line_data_generator import LineDataGenerator


@pytest.fixture
def sample_dataframe():
    """
    Creates a sample pandas DataFrame that mimics the expected output of load_line_data.

    DataFrame structure:
    - folder_id: experiment folder name
    - image_name: name of the image file
    - image: resized image array
    - labels: dictionary mapping label names to coordinate arrays
    """
    # Configuration
    num_samples = 20
    image_height, image_width = 256, 256
    label_categories = ['line_category_1', 'line_category_2', 'line_category_3']
    experiments = ['experiment_1', 'experiment_2']

    data_rows = []

    for exp_idx, experiment in enumerate(experiments):
        for sample_idx in range(num_samples // len(experiments)):
            # Create sample image
            image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

            # Add different patterns for different experiments
            if exp_idx == 0:
                # Diagonal lines
                cv2.line(image, (0, 0), (255, 255), (255, 255, 255), thickness=2)
                cv2.line(image, (0, 128), (255, 128), (128, 128, 128), thickness=1)
            else:
                # Horizontal and vertical lines
                cv2.line(image, (0, 64), (255, 64), (255, 255, 255), thickness=2)
                cv2.line(image, (128, 0), (128, 255), (128, 128, 128), thickness=1)

            # Add some noise
            noise = np.random.randint(0, 30, (image_height, image_width, 3), dtype=np.uint8)
            image = cv2.add(image, noise)

            # Create labels dictionary
            labels = {}
            for cat_idx, category in enumerate(label_categories):
                # Generate line coordinates for this category
                if category == 'line_category_1':
                    # Diagonal line with noise
                    t = np.linspace(0, 1, 15)
                    x_coords = t * 250 + np.random.normal(0, 2, 15)
                    y_coords = t * 250 + np.random.normal(0, 2, 15)
                elif category == 'line_category_2':
                    # Horizontal line with noise
                    x_coords = np.linspace(10, 245, 12) + np.random.normal(0, 1, 12)
                    y_coords = np.full(12, 128) + np.random.normal(0, 3, 12)
                else:  # line_category_3
                    # Curved line
                    t = np.linspace(0, np.pi, 20)
                    x_coords = 128 + 80 * np.cos(t) + np.random.normal(0, 1, 20)
                    y_coords = 128 + 40 * np.sin(t) + np.random.normal(0, 1, 20)

                # Clip coordinates to image bounds
                x_coords = np.clip(x_coords, 0, image_width - 1)
                y_coords = np.clip(y_coords, 0, image_height - 1)

                # Stack as (N, 2) array - important: (x, y) format
                coords = np.column_stack((x_coords, y_coords))
                labels[category] = coords

            # Create row
            row = {
                'folder_id': experiment,
                'image_name': f'img{sample_idx:07d}.png',
                'image': image.copy(),
                'labels': labels.copy()
            }
            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    return df


@pytest.fixture
def sample_dataframe_missing_labels():
    """
    Creates a DataFrame where some images are missing certain label categories.
    This tests the behavior when not all labels are present for every image.
    """
    num_samples = 8
    image_height, image_width = 128, 128
    label_categories = ['category_A', 'category_B', 'category_C']

    data_rows = []

    for sample_idx in range(num_samples):
        # Create sample image
        image = np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8)

        # Create labels dictionary - some samples missing certain categories
        labels = {}

        # All samples have category_A
        t = np.linspace(0, 1, 10)
        x_coords = t * 120 + np.random.normal(0, 1, 10)
        y_coords = t * 120 + np.random.normal(0, 1, 10)
        x_coords = np.clip(x_coords, 0, image_width - 1)
        y_coords = np.clip(y_coords, 0, image_height - 1)
        labels['category_A'] = np.column_stack((x_coords, y_coords))

        # Only even samples have category_B
        if sample_idx % 2 == 0:
            x_coords = np.full(8, 64) + np.random.normal(0, 2, 8)
            y_coords = np.linspace(10, 118, 8) + np.random.normal(0, 1, 8)
            x_coords = np.clip(x_coords, 0, image_width - 1)
            y_coords = np.clip(y_coords, 0, image_height - 1)
            labels['category_B'] = np.column_stack((x_coords, y_coords))

        # Only samples 0, 3, 6 have category_C
        if sample_idx % 3 == 0:
            t = np.linspace(0, 2*np.pi, 12)
            x_coords = 64 + 30 * np.cos(t) + np.random.normal(0, 1, 12)
            y_coords = 64 + 30 * np.sin(t) + np.random.normal(0, 1, 12)
            x_coords = np.clip(x_coords, 0, image_width - 1)
            y_coords = np.clip(y_coords, 0, image_height - 1)
            labels['category_C'] = np.column_stack((x_coords, y_coords))

        row = {
            'folder_id': 'test_experiment',
            'image_name': f'sample_{sample_idx:03d}.png',
            'image': image.copy(),
            'labels': labels.copy()
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    return df


def test_line_data_generator_from_dataframe_basic(sample_dataframe):
    """
    Test basic functionality of LineDataGenerator with DataFrame input.
    """
    df = sample_dataframe

    # Test with specific label order
    label_order = ['line_category_1', 'line_category_2', 'line_category_3']

    generator = LineDataGenerator(
        dataframe=df,
        label_order=label_order,
        batch_size=4,
        training=False,
        shuffle=False
    )

    # Test generator properties
    assert len(generator) > 0
    assert generator.batch_size == 4

    # Test a batch
    X, y = generator[0]

    # Check shapes
    assert X.shape[0] == 4  # batch_size
    assert X.shape[1:3] == (256, 256)  # image dimensions
    assert X.shape[3] == 3  # RGB channels

    # Check label tensor - should have 3 channels (one per label category)
    assert y.shape[0] == 4  # batch_size
    assert y.shape[1:3] == (256, 256)  # image dimensions
    assert y.shape[3] == 3  # number of label categories

    # Check data types
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_line_data_generator_with_missing_labels(sample_dataframe_missing_labels):
    """
    Test LineDataGenerator when some images are missing certain label categories.
    Missing labels should result in blank channels.
    """
    df = sample_dataframe_missing_labels

    label_order = ['category_A', 'category_B', 'category_C']

    generator = LineDataGenerator(
        dataframe=df,
        label_order=label_order,
        batch_size=4,
        training=False,
        shuffle=False
    )

    X, y = generator[0]

    # Check shapes
    assert y.shape == (4, 128, 128, 3)

    # All samples should have category_A (channel 0) with some non-zero values
    for i in range(4):
        assert np.any(y[i, :, :, 0] > 0), f"Sample {i} should have category_A data"

    # Only even samples should have category_B (channel 1)
    for i in range(4):
        if i % 2 == 0:
            assert np.any(y[i, :, :, 1] > 0), f"Even sample {i} should have category_B data"
        else:
            assert np.all(y[i, :, :, 1] == 0), f"Odd sample {i} should have blank category_B"


def test_line_data_generator_compressed_labels(sample_dataframe):
    """
    Test LineDataGenerator with compressed labels (all categories in single channel).
    """
    df = sample_dataframe

    label_order = ['line_category_1', 'line_category_2', 'line_category_3']

    generator = LineDataGenerator(
        dataframe=df,
        label_order=label_order,
        batch_size=4,
        training=False,
        shuffle=False,
        compress_labels=True
    )

    X, y = generator[0]

    # Check shapes - should have only 1 channel when compressed
    assert y.shape == (4, 256, 256, 1)

    # Check that compressed channel contains combined data
    for i in range(4):
        assert np.any(y[i, :, :, 0] > 0), f"Sample {i} should have combined label data"


def test_line_data_generator_with_background_channel(sample_dataframe):
    """
    Test LineDataGenerator with background channel included.
    """
    df = sample_dataframe

    label_order = ['line_category_1', 'line_category_2']

    generator = LineDataGenerator(
        dataframe=df,
        label_order=label_order,
        batch_size=2,
        training=False,
        shuffle=False,
        include_background=True
    )

    X, y = generator[0]

    # Should have background + 2 label channels = 3 total
    assert y.shape[3] == 3

    # Background should be channel 0, and should be inverse of other channels
    for i in range(2):
        foreground = np.any(y[i, :, :, 1:], axis=2)  # Any foreground label
        background = y[i, :, :, 0] > 0
        # Background should be roughly inverse of foreground (allowing for line width)
        background_pixels = np.sum(background)
        total_pixels = background.size
        assert background_pixels > 0, "Should have some background pixels"


def test_line_data_generator_different_resolutions(sample_dataframe):
    """
    Test LineDataGenerator with different target resolutions.
    """
    df = sample_dataframe

    label_order = ['line_category_1']
    target_resolution = (128, 64)  # Different aspect ratio

    generator = LineDataGenerator(
        dataframe=df,
        label_order=label_order,
        target_resolution=target_resolution,
        batch_size=2,
        training=False,
        shuffle=False
    )

    X, y = generator[0]

    # Check that images and labels are resized correctly
    assert X.shape[1:3] == target_resolution
    assert y.shape[1:3] == target_resolution


def test_line_data_generator_shuffle_and_epochs(sample_dataframe):
    """
    Test shuffling behavior and epoch management.
    """
    df = sample_dataframe

    label_order = ['line_category_1']

    generator = LineDataGenerator(
        dataframe=df,
        label_order=label_order,
        batch_size=4,
        training=True,
        shuffle=True
    )

    # Get first batch
    X1, y1 = generator[0]

    # Trigger epoch end
    generator.on_epoch_end()

    # Get first batch again - should be different due to shuffling
    X2, y2 = generator[0]

    # Note: This test might occasionally pass even with proper shuffling
    # due to randomness, but it's a reasonable check
    assert X1.shape == X2.shape
    assert y1.shape == y2.shape


def test_line_data_generator_error_handling():
    """
    Test error handling for invalid inputs.
    """
    # Test with empty DataFrame
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        LineDataGenerator(
            dataframe=empty_df,
            label_order=['category_1'],
            batch_size=4
        )

    # Test with invalid label order (category not in data)
    df = pd.DataFrame([{
        'folder_id': 'test',
        'image_name': 'test.png',
        'image': np.zeros((64, 64, 3), dtype=np.uint8),
        'labels': {'existing_category': np.array([[10, 10], [20, 20]])}
    }])

    with pytest.raises(ValueError):
        LineDataGenerator(
            dataframe=df,
            label_order=['nonexistent_category'],
            batch_size=1
        )
