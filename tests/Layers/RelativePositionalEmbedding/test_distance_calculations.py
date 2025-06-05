"""Tests for distance calculation utilities."""

import pytest
import numpy as np

from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.distance_calculations import (
    calculate_spatial_distances,
    calculate_temporal_distances,
    calculate_all_distances,
    calculate_key_distances,
    validate_embedding_shapes
)
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_config import (
    RelativePositionalEmbeddingConfig
)


def test_spatial_distances():
    """Test spatial distance calculation."""
    distances = calculate_spatial_distances(4, 4)
    assert distances.shape == (4, 4)
    assert distances.dtype == np.int32
    assert np.all(distances >= 0)


def test_temporal_distances():
    """Test temporal distance calculation."""
    distances = calculate_temporal_distances(3, 3)
    assert distances.shape == (3, 3)
    assert distances.dtype == np.int32
    assert np.all(distances >= 0)


def test_all_distances():
    """Test complete distance calculation."""
    config = RelativePositionalEmbeddingConfig(
        query_shape=(2, 4, 6, 64),
        key_shape=(3, 8, 12, 128),
        query_dim=64,
        heads=8
    )
    
    height_dist, width_dist, time_dist = calculate_all_distances(config)
    
    assert height_dist.shape == (4, 8)
    assert width_dist.shape == (6, 12)
    assert time_dist.shape == (2, 3)
    assert np.all(height_dist >= 0)
    assert np.all(width_dist >= 0)
    assert np.all(time_dist >= 0)


def test_key_distances():
    """Test key perspective distance calculation."""
    config = RelativePositionalEmbeddingConfig(
        query_shape=(2, 4, 6, 64),
        key_shape=(3, 8, 12, 128),
        query_dim=64,
        heads=8
    )
    
    height_dist, width_dist, time_dist = calculate_key_distances(config)
    
    assert height_dist.shape == (8, 4)
    assert width_dist.shape == (12, 6)
    assert time_dist.shape == (3, 2)
    assert np.all(height_dist >= 0)
    assert np.all(width_dist >= 0)
    assert np.all(time_dist >= 0)


def test_validate_embedding_shapes():
    """Test embedding shape validation."""
    config = RelativePositionalEmbeddingConfig(
        query_shape=(2, 8, 12, 64),
        key_shape=(3, 16, 24, 128),
        query_dim=256,
        heads=8
    )
    
    height_shape, width_shape, time_shape = validate_embedding_shapes(config)
    
    expected_height_rows = 2 * (max(8, 16) - 1) + 1
    expected_width_rows = 2 * (max(12, 24) - 1) + 1
    expected_time_rows = 2 * (max(2, 3) - 1) + 1
    
    assert height_shape == (expected_height_rows, 256)
    assert width_shape == (expected_width_rows, 256)
    assert time_shape == (expected_time_rows, 256)


def test_distance_validation():
    """Test input validation for distance functions."""
    with pytest.raises(ValueError):
        calculate_spatial_distances(0, 5)
    
    with pytest.raises(ValueError):
        calculate_temporal_distances(-1, 5)


if __name__ == "__main__":
    pytest.main([__file__]) 