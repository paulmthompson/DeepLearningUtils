"""
Tests for RelativePositionalEmbeddingConfig dataclass.

This module provides comprehensive tests for the configuration dataclass used by
RelativePositionalEmbedding layers, ensuring proper validation and immutability.
"""

import pytest
from typing import Tuple

from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_config import (
    RelativePositionalEmbeddingConfig
)


class TestRelativePositionalEmbeddingConfig:
    """Test suite for RelativePositionalEmbeddingConfig dataclass."""
    
    def test_valid_config_creation(self):
        """Test creation of valid configuration."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(1, 32, 32, 64),
            key_shape=(1, 32, 32, 64),
            query_dim=64,
            heads=8,
            drop_rate=0.1
        )
        
        assert config.query_shape == (1, 32, 32, 64)
        assert config.key_shape == (1, 32, 32, 64)
        assert config.query_dim == 64
        assert config.heads == 8
        assert config.drop_rate == 0.1
    
    def test_default_drop_rate(self):
        """Test default value for drop_rate."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(1, 16, 16, 128),
            key_shape=(1, 16, 16, 128),
            query_dim=128,
            heads=4
        )
        
        assert config.drop_rate == 0.0
    
    def test_config_immutability(self):
        """Test that configuration is immutable (frozen)."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(1, 32, 32, 64),
            key_shape=(1, 32, 32, 64),
            query_dim=64,
            heads=8
        )
        
        with pytest.raises(AttributeError):
            config.query_dim = 128
        
        with pytest.raises(AttributeError):
            config.heads = 16
    
    @pytest.mark.parametrize("invalid_query_shape", [
        "not_a_tuple",
        [1, 32, 32, 64],  # List instead of tuple
        (1, 32, 32),      # Too few elements
        (1, 32, 32, 64, 3),  # Too many elements
        (0, 32, 32, 64),  # Zero dimension
        (-1, 32, 32, 64), # Negative dimension
        (1.5, 32, 32, 64), # Float instead of int
    ])
    def test_invalid_query_shape(self, invalid_query_shape):
        """Test validation of invalid query shapes."""
        with pytest.raises((ValueError, TypeError)):
            RelativePositionalEmbeddingConfig(
                query_shape=invalid_query_shape,
                key_shape=(1, 32, 32, 64),
                query_dim=64,
                heads=8
            )
    
    @pytest.mark.parametrize("invalid_key_shape", [
        "not_a_tuple",
        [1, 32, 32, 64],  # List instead of tuple
        (1, 32, 32),      # Too few elements
        (1, 32, 32, 64, 3),  # Too many elements
        (0, 32, 32, 64),  # Zero dimension
        (-1, 32, 32, 64), # Negative dimension
        (1.5, 32, 32, 64), # Float instead of int
    ])
    def test_invalid_key_shape(self, invalid_key_shape):
        """Test validation of invalid key shapes."""
        with pytest.raises((ValueError, TypeError)):
            RelativePositionalEmbeddingConfig(
                query_shape=(1, 32, 32, 64),
                key_shape=invalid_key_shape,
                query_dim=64,
                heads=8
            )
    
    @pytest.mark.parametrize("invalid_query_dim", [
        0, -1, -10, 1.5, "64", None
    ])
    def test_invalid_query_dim(self, invalid_query_dim):
        """Test validation of invalid query dimensions."""
        with pytest.raises((ValueError, TypeError)):
            RelativePositionalEmbeddingConfig(
                query_shape=(1, 32, 32, 64),
                key_shape=(1, 32, 32, 64),
                query_dim=invalid_query_dim,
                heads=8
            )
    
    @pytest.mark.parametrize("invalid_heads", [
        0, -1, -8, 1.5, "8", None
    ])
    def test_invalid_heads(self, invalid_heads):
        """Test validation of invalid number of heads."""
        with pytest.raises((ValueError, TypeError)):
            RelativePositionalEmbeddingConfig(
                query_shape=(1, 32, 32, 64),
                key_shape=(1, 32, 32, 64),
                query_dim=64,
                heads=invalid_heads
            )
    
    @pytest.mark.parametrize("invalid_drop_rate", [
        -0.1, -1.0, 1.1, 2.0, "0.1", None
    ])
    def test_invalid_drop_rate(self, invalid_drop_rate):
        """Test validation of invalid dropout rates."""
        with pytest.raises((ValueError, TypeError)):
            RelativePositionalEmbeddingConfig(
                query_shape=(1, 32, 32, 64),
                key_shape=(1, 32, 32, 64),
                query_dim=64,
                heads=8,
                drop_rate=invalid_drop_rate
            )
    
    def test_boundary_drop_rates(self):
        """Test boundary values for dropout rate."""
        # Test minimum boundary
        config_min = RelativePositionalEmbeddingConfig(
            query_shape=(1, 32, 32, 64),
            key_shape=(1, 32, 32, 64),
            query_dim=64,
            heads=8,
            drop_rate=0.0
        )
        assert config_min.drop_rate == 0.0
        
        # Test maximum boundary
        config_max = RelativePositionalEmbeddingConfig(
            query_shape=(1, 32, 32, 64),
            key_shape=(1, 32, 32, 64),
            query_dim=64,
            heads=8,
            drop_rate=1.0
        )
        assert config_max.drop_rate == 1.0
    
    def test_property_accessors(self):
        """Test property accessors for shape components."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(2, 16, 24, 128),
            key_shape=(3, 8, 12, 256),
            query_dim=64,
            heads=4
        )
        
        # Query shape properties
        assert config.query_seq_len == 2
        assert config.query_height == 16
        assert config.query_width == 24
        assert config.query_channels == 128
        
        # Key shape properties
        assert config.key_seq_len == 3
        assert config.key_height == 8
        assert config.key_width == 12
        assert config.key_channels == 256
    
    def test_distance_calculations(self):
        """Test maximum distance calculations."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(2, 16, 24, 128),
            key_shape=(3, 8, 12, 256),
            query_dim=64,
            heads=4
        )
        
        # Test max distance calculations
        assert config.max_height_dist == max(16, 8) - 1 == 15
        assert config.max_width_dist == max(24, 12) - 1 == 23
        assert config.max_time_dist == max(2, 3) - 1 == 2
    
    def test_different_query_key_shapes(self):
        """Test configuration with different query and key shapes."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(1, 64, 64, 256),
            key_shape=(2, 32, 32, 512),
            query_dim=128,
            heads=16
        )
        
        assert config.query_shape == (1, 64, 64, 256)
        assert config.key_shape == (2, 32, 32, 512)
        assert config.query_dim == 128
        assert config.heads == 16
    
    def test_large_dimensions(self):
        """Test configuration with large dimension values."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(10, 512, 512, 2048),
            key_shape=(20, 256, 256, 1024),
            query_dim=1024,
            heads=32
        )
        
        assert config.max_height_dist == 511
        assert config.max_width_dist == 511
        assert config.max_time_dist == 19
    
    def test_minimal_dimensions(self):
        """Test configuration with minimal dimension values."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(1, 1, 1, 1),
            key_shape=(1, 1, 1, 1),
            query_dim=1,
            heads=1
        )
        
        assert config.max_height_dist == 0
        assert config.max_width_dist == 0
        assert config.max_time_dist == 0
    
    def test_config_equality(self):
        """Test equality comparison between configurations."""
        config1 = RelativePositionalEmbeddingConfig(
            query_shape=(1, 32, 32, 64),
            key_shape=(1, 32, 32, 64),
            query_dim=64,
            heads=8,
            drop_rate=0.1
        )
        
        config2 = RelativePositionalEmbeddingConfig(
            query_shape=(1, 32, 32, 64),
            key_shape=(1, 32, 32, 64),
            query_dim=64,
            heads=8,
            drop_rate=0.1
        )
        
        config3 = RelativePositionalEmbeddingConfig(
            query_shape=(1, 32, 32, 64),
            key_shape=(1, 32, 32, 64),
            query_dim=128,  # Different query_dim
            heads=8,
            drop_rate=0.1
        )
        
        assert config1 == config2
        assert config1 != config3
    
    def test_config_hashing(self):
        """Test that configuration objects are hashable."""
        config = RelativePositionalEmbeddingConfig(
            query_shape=(1, 32, 32, 64),
            key_shape=(1, 32, 32, 64),
            query_dim=64,
            heads=8
        )
        
        # Should be able to use as dictionary key
        config_dict = {config: "test_value"}
        assert config_dict[config] == "test_value"
        
        # Should be able to add to set
        config_set = {config}
        assert config in config_set


if __name__ == "__main__":
    pytest.main([__file__]) 