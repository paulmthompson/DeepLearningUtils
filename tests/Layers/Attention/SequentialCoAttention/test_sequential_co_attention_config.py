"""
Tests for sequential co-attention configuration dataclasses.

This module tests the configuration objects for sequential co-attention implementations,
ensuring proper validation, type safety, and immutability according to design guidelines.
"""

import pytest
from typing import Tuple

from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_config import (
    CoMemoryAttentionConfig,
    CoAttentionConfig
)


class TestCoMemoryAttentionConfig:
    """Test cases for CoMemoryAttentionConfig dataclass."""
    
    def test_valid_configuration(self):
        """Test creating valid configuration."""
        config = CoMemoryAttentionConfig(
            query_shape=(1, 16, 16, 128),
            memory_shape=(1, 16, 16, 128),
            key_dim=64,
            value_dim=128,
            attention_heads=4
        )
        
        assert config.query_shape == (1, 16, 16, 128)
        assert config.memory_shape == (1, 16, 16, 128)
        assert config.key_dim == 64
        assert config.value_dim == 128
        assert config.attention_heads == 4
    
    def test_default_parameters(self):
        """Test default parameter values."""
        config = CoMemoryAttentionConfig(
            query_shape=(1, 8, 8, 64),
            memory_shape=(2, 8, 8, 64)
        )
        
        assert config.key_dim == 128
        assert config.value_dim == 256
        assert config.use_norm is False
        assert config.attention_drop_rate == 0.0
        assert config.use_positional_embedding is True
        assert config.use_key_positional_embedding is True
        assert config.attention_heads == 8
        assert config.use_qkv_embedding is False
    
    def test_property_accessors(self):
        """Test configuration property accessors."""
        config = CoMemoryAttentionConfig(
            query_shape=(2, 32, 16, 128),
            memory_shape=(3, 16, 32, 256)
        )
        
        # Query properties
        assert config.query_seq_len == 2
        assert config.query_height == 32
        assert config.query_width == 16
        assert config.query_channels == 128
        assert config.query_spatial_size == 512  # 32 * 16
        assert config.query_total_size == 1024  # 2 * 32 * 16
        
        # Memory properties
        assert config.memory_seq_len == 3
        assert config.memory_height == 16
        assert config.memory_width == 32
        assert config.memory_channels == 256
        assert config.memory_spatial_size == 512  # 16 * 32
        assert config.memory_total_size == 1536  # 3 * 16 * 32
    
    def test_invalid_query_shape(self):
        """Test validation of invalid query shapes."""
        with pytest.raises(ValueError, match="query_shape must be a 4-tuple"):
            CoMemoryAttentionConfig(
                query_shape=(16, 16, 128),  # Missing seq_len
                memory_shape=(1, 16, 16, 128)
            )
        
        with pytest.raises(ValueError, match="positive integers"):
            CoMemoryAttentionConfig(
                query_shape=(0, 16, 16, 128),  # Zero dimension
                memory_shape=(1, 16, 16, 128)
            )
    
    def test_invalid_memory_shape(self):
        """Test validation of invalid memory shapes."""
        with pytest.raises(ValueError, match="memory_shape must be a 4-tuple"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(16, 16, 128, 64, 1)  # Too many dimensions
            )
        
        with pytest.raises(ValueError, match="positive integers"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(1, -16, 16, 128)  # Negative dimension
            )
    
    def test_invalid_dimensions(self):
        """Test validation of invalid key/value dimensions."""
        with pytest.raises(ValueError, match="key_dim must be a positive integer"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(1, 16, 16, 128),
                key_dim=0
            )
        
        with pytest.raises(ValueError, match="value_dim must be a positive integer"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(1, 16, 16, 128),
                value_dim=-64
            )
    
    def test_divisibility_validation(self):
        """Test validation of head divisibility requirements."""
        with pytest.raises(ValueError, match="key_dim .* must be divisible by attention_heads"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(1, 16, 16, 128),
                key_dim=64,
                attention_heads=5  # 64 not divisible by 5
            )
        
        with pytest.raises(ValueError, match="value_dim .* must be divisible by attention_heads"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(1, 16, 16, 128),
                value_dim=100,
                attention_heads=8  # 100 not divisible by 8
            )
    
    def test_dropout_validation(self):
        """Test validation of dropout rate."""
        with pytest.raises(ValueError, match="attention_drop_rate must be in range"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(1, 16, 16, 128),
                attention_drop_rate=1.5  # > 1.0
            )
        
        with pytest.raises(ValueError, match="attention_drop_rate must be in range"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(1, 16, 16, 128),
                attention_drop_rate=-0.1  # < 0.0
            )
    
    def test_boolean_flag_validation(self):
        """Test validation of boolean flags."""
        with pytest.raises(TypeError, match="use_norm must be a boolean"):
            CoMemoryAttentionConfig(
                query_shape=(1, 16, 16, 128),
                memory_shape=(1, 16, 16, 128),
                use_norm="True"  # String instead of bool
            )
    
    def test_immutability(self):
        """Test that configuration is immutable."""
        config = CoMemoryAttentionConfig(
            query_shape=(1, 16, 16, 128),
            memory_shape=(1, 16, 16, 128)
        )
        
        with pytest.raises(AttributeError):
            config.key_dim = 256  # Should not be allowed


class TestCoAttentionConfig:
    """Test cases for CoAttentionConfig dataclass."""
    
    def test_valid_configuration(self):
        """Test creating valid configuration."""
        config = CoAttentionConfig(
            query_shape=(2, 1, 16, 16, 128),
            memory_shape=(2, 5, 16, 16, 128),
            key_dim=64,
            value_dim=64,
            hidden_dim=256
        )
        
        assert config.query_shape == (2, 1, 16, 16, 128)
        assert config.memory_shape == (2, 5, 16, 16, 128)
        assert config.key_dim == 64
        assert config.value_dim == 64
        assert config.hidden_dim == 256
    
    def test_default_parameters(self):
        """Test default parameter values."""
        config = CoAttentionConfig(
            query_shape=(1, 1, 8, 8, 64),
            memory_shape=(1, 3, 8, 8, 64)
        )
        
        assert config.key_dim == 128
        assert config.value_dim == 128
        assert config.hidden_dim == 512
        assert config.layer_norm_eps == 1e-3
    
    def test_property_accessors(self):
        """Test configuration property accessors."""
        config = CoAttentionConfig(
            query_shape=(4, 2, 32, 16, 128),
            memory_shape=(4, 10, 16, 32, 256)
        )
        
        # Query properties
        assert config.batch_size == 4
        assert config.query_seq_len == 2
        assert config.query_height == 32
        assert config.query_width == 16
        assert config.query_channels == 128
        
        # Memory properties
        assert config.memory_frames == 10
        assert config.memory_height == 16
        assert config.memory_width == 32
        assert config.memory_channels == 256
    
    def test_invalid_query_shape(self):
        """Test validation of invalid query shapes."""
        with pytest.raises(ValueError, match="query_shape must be a 5-tuple"):
            CoAttentionConfig(
                query_shape=(1, 16, 16, 128),  # Missing dimensions
                memory_shape=(1, 5, 16, 16, 128)
            )
        
        with pytest.raises(ValueError, match="positive integers"):
            CoAttentionConfig(
                query_shape=(1, 0, 16, 16, 128),  # Zero dimension
                memory_shape=(1, 5, 16, 16, 128)
            )
    
    def test_invalid_memory_shape(self):
        """Test validation of invalid memory shapes."""
        with pytest.raises(ValueError, match="memory_shape must be a 5-tuple"):
            CoAttentionConfig(
                query_shape=(1, 1, 16, 16, 128),
                memory_shape=(1, 5, 16)  # Too few dimensions
            )
        
        with pytest.raises(ValueError, match="positive integers"):
            CoAttentionConfig(
                query_shape=(1, 1, 16, 16, 128),
                memory_shape=(1, -5, 16, 16, 128)  # Negative dimension
            )
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="key_dim must be a positive integer"):
            CoAttentionConfig(
                query_shape=(1, 1, 16, 16, 128),
                memory_shape=(1, 5, 16, 16, 128),
                key_dim=0
            )
        
        with pytest.raises(ValueError, match="hidden_dim must be a positive integer"):
            CoAttentionConfig(
                query_shape=(1, 1, 16, 16, 128),
                memory_shape=(1, 5, 16, 16, 128),
                hidden_dim=-512
            )
    
    def test_epsilon_validation(self):
        """Test validation of layer norm epsilon."""
        with pytest.raises(ValueError, match="layer_norm_eps must be a positive float"):
            CoAttentionConfig(
                query_shape=(1, 1, 16, 16, 128),
                memory_shape=(1, 5, 16, 16, 128),
                layer_norm_eps=0.0
            )
        
        with pytest.raises(ValueError, match="layer_norm_eps must be a positive float"):
            CoAttentionConfig(
                query_shape=(1, 1, 16, 16, 128),
                memory_shape=(1, 5, 16, 16, 128),
                layer_norm_eps=-1e-5
            )
    
    def test_immutability(self):
        """Test that configuration is immutable."""
        config = CoAttentionConfig(
            query_shape=(1, 1, 16, 16, 128),
            memory_shape=(1, 5, 16, 16, 128)
        )
        
        with pytest.raises(AttributeError):
            config.hidden_dim = 1024  # Should not be allowed


class TestConfigurationCompatibility:
    """Test compatibility between different configurations."""
    
    def test_memory_config_compatibility(self):
        """Test that memory and co-attention configs are compatible."""
        memory_config = CoMemoryAttentionConfig(
            query_shape=(1, 16, 16, 128),
            memory_shape=(1, 16, 16, 128),
            key_dim=64,
            value_dim=128
        )
        
        coattention_config = CoAttentionConfig(
            query_shape=(2, 1, 16, 16, 128),
            memory_shape=(2, 5, 16, 16, 128),
            key_dim=64,
            value_dim=128
        )
        
        # Should be compatible for spatial dimensions
        assert memory_config.query_height == coattention_config.query_height
        assert memory_config.query_width == coattention_config.query_width
        assert memory_config.key_dim == coattention_config.key_dim
        assert memory_config.value_dim == coattention_config.value_dim
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimal valid configuration
        config = CoMemoryAttentionConfig(
            query_shape=(1, 1, 1, 1),
            memory_shape=(1, 1, 1, 1),
            key_dim=1,
            value_dim=1,
            attention_heads=1
        )
        assert config.query_spatial_size == 1
        assert config.memory_total_size == 1
        
        # Large configuration
        config = CoAttentionConfig(
            query_shape=(16, 8, 512, 512, 2048),
            memory_shape=(16, 32, 256, 256, 1024),
            key_dim=2048,
            value_dim=2048,
            hidden_dim=8192
        )
        assert config.query_height == 512
        assert config.memory_frames == 32 