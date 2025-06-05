"""
Tests for MVit2 attention configuration dataclasses.

This module tests the configuration objects for multi-head attention implementations,
ensuring proper validation, type safety, and immutability according to design guidelines.
"""

import pytest
from typing import Tuple

from src.DeepLearningUtils.Layers.Attention.MHA_mvit2.attention_config import (
    DotProductAttentionConfig,
    MultiHeadAttentionConfig,
    AttentionType
)


class TestDotProductAttentionConfig:
    """Test cases for DotProductAttentionConfig dataclass."""
    
    def test_valid_configuration(self):
        """Test creating valid configuration."""
        config = DotProductAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            query_dim=16,
            heads=8
        )
        
        assert config.query_shape == (1, 8, 8, 64)
        assert config.key_shape == (1, 8, 8, 64)
        assert config.query_dim == 16
        assert config.heads == 8
        assert config.use_scale is True  # default
        assert config.drop_rate == 0.0  # default
    
    def test_property_accessors(self):
        """Test property accessor methods."""
        config = DotProductAttentionConfig(
            query_shape=(2, 16, 16, 128),
            key_shape=(3, 8, 8, 128),
            query_dim=32,
            heads=4
        )
        
        # Query properties
        assert config.query_seq_len == 2
        assert config.query_height == 16
        assert config.query_width == 16
        assert config.query_channels == 128
        
        # Key properties
        assert config.key_seq_len == 3
        assert config.key_height == 8
        assert config.key_width == 8
        assert config.key_channels == 128
        
        # Computed properties
        assert config.query_spatial_size == 256  # 16 * 16
        assert config.key_spatial_size == 64    # 8 * 8
        assert config.query_total_size == 512   # 2 * 16 * 16
        assert config.key_total_size == 192     # 3 * 8 * 8
        assert config.head_dim == 8             # 32 / 4
    
    def test_immutability(self):
        """Test that configuration is immutable."""
        config = DotProductAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            query_dim=16,
            heads=8
        )
        
        with pytest.raises(AttributeError):
            config.query_dim = 32
    
    def test_invalid_query_shape(self):
        """Test validation of query_shape parameter."""
        # Wrong length
        with pytest.raises(ValueError, match="query_shape must be a 4-tuple"):
            DotProductAttentionConfig(
                query_shape=(1, 8, 8),  # Missing channel dimension
                key_shape=(1, 8, 8, 64),
                query_dim=16,
                heads=8
            )
        
        # Negative dimensions
        with pytest.raises(ValueError, match="All query_shape dimensions must be positive"):
            DotProductAttentionConfig(
                query_shape=(1, -8, 8, 64),
                key_shape=(1, 8, 8, 64),
                query_dim=16,
                heads=8
            )
    
    def test_invalid_key_shape(self):
        """Test validation of key_shape parameter."""
        with pytest.raises(ValueError, match="key_shape must be a 4-tuple"):
            DotProductAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8),  # Wrong length
                query_dim=16,
                heads=8
            )
    
    def test_invalid_query_dim(self):
        """Test validation of query_dim parameter."""
        with pytest.raises(ValueError, match="query_dim must be a positive integer"):
            DotProductAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                query_dim=-16,  # Negative
                heads=8
            )
    
    def test_invalid_heads(self):
        """Test validation of heads parameter."""
        with pytest.raises(ValueError, match="heads must be a positive integer"):
            DotProductAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                query_dim=16,
                heads=0  # Zero heads
            )
    
    def test_query_dim_not_divisible_by_heads(self):
        """Test validation that query_dim must be divisible by heads."""
        with pytest.raises(ValueError, match="query_dim \\(17\\) must be divisible by heads \\(8\\)"):
            DotProductAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                query_dim=17,  # Not divisible by 8
                heads=8
            )
    
    def test_invalid_drop_rate(self):
        """Test validation of drop_rate parameter."""
        with pytest.raises(ValueError, match="drop_rate must be a float in range \\[0.0, 1.0\\]"):
            DotProductAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                query_dim=16,
                heads=8,
                drop_rate=1.5  # Out of range
            )
    
    def test_invalid_attention_type(self):
        """Test validation of attention_type parameter."""
        with pytest.raises(ValueError, match="attention_type must be 'softmax' or 'linear'"):
            DotProductAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                query_dim=16,
                heads=8,
                attention_type="invalid"  # type: ignore
            )
    
    def test_type_validation(self):
        """Test type validation for boolean parameters."""
        with pytest.raises(TypeError, match="use_scale must be a boolean"):
            DotProductAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                query_dim=16,
                heads=8,
                use_scale="true"  # type: ignore
            )


class TestMultiHeadAttentionConfig:
    """Test cases for MultiHeadAttentionConfig dataclass."""
    
    def test_valid_configuration(self):
        """Test creating valid configuration."""
        config = MultiHeadAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            heads=8,
            value_dim=128,
            key_dim=128
        )
        
        assert config.query_shape == (1, 8, 8, 64)
        assert config.key_shape == (1, 8, 8, 64)
        assert config.heads == 8
        assert config.value_dim == 128
        assert config.key_dim == 128
    
    def test_head_dimensions(self):
        """Test head dimension calculations."""
        config = MultiHeadAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            heads=8,
            value_dim=128,
            key_dim=64
        )
        
        assert config.head_dim == 16      # value_dim / heads = 128 / 8
        assert config.key_head_dim == 8   # key_dim / heads = 64 / 8
    
    def test_get_attention_config(self):
        """Test creating DotProductAttentionConfig from MultiHeadAttentionConfig."""
        mha_config = MultiHeadAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            heads=8,
            value_dim=128,
            key_dim=64,
            attention_drop_rate=0.1,
            use_positional_embedding=False,
            attention_type="linear"
        )
        
        attention_config = mha_config.get_attention_config()
        
        assert isinstance(attention_config, DotProductAttentionConfig)
        assert attention_config.query_shape == (1, 8, 8, 64)
        assert attention_config.key_shape == (1, 8, 8, 64)
        assert attention_config.query_dim == 8  # key_head_dim
        assert attention_config.heads == 8
        assert attention_config.drop_rate == 0.1
        assert attention_config.use_positional_embedding is False
        assert attention_config.attention_type == "linear"
        assert attention_config.name == "attention"
    
    def test_dimension_divisibility_validation(self):
        """Test that dimensions must be divisible by heads."""
        # value_dim not divisible by heads
        with pytest.raises(ValueError, match="value_dim \\(130\\) must be divisible by heads \\(8\\)"):
            MultiHeadAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                heads=8,
                value_dim=130,  # Not divisible by 8
                key_dim=128
            )
        
        # key_dim not divisible by heads
        with pytest.raises(ValueError, match="key_dim \\(65\\) must be divisible by heads \\(8\\)"):
            MultiHeadAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                heads=8,
                value_dim=128,
                key_dim=65  # Not divisible by 8
            )
    
    def test_boolean_flag_validation(self):
        """Test validation of boolean flags."""
        with pytest.raises(TypeError, match="use_query_embedding must be a boolean"):
            MultiHeadAttentionConfig(
                query_shape=(1, 8, 8, 64),
                key_shape=(1, 8, 8, 64),
                use_query_embedding=1  # type: ignore
            )
    
    def test_equality_and_hashing(self):
        """Test equality comparison and hashing."""
        config1 = MultiHeadAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            heads=8,
            value_dim=128,
            key_dim=128
        )
        
        config2 = MultiHeadAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            heads=8,
            value_dim=128,
            key_dim=128
        )
        
        config3 = MultiHeadAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            heads=4,  # Different
            value_dim=128,
            key_dim=128
        )
        
        assert config1 == config2
        assert config1 != config3
        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)


class TestAttentionConfigIntegration:
    """Integration tests for attention configurations."""
    
    def test_config_with_different_query_key_shapes(self):
        """Test configuration with different query and key shapes."""
        config = MultiHeadAttentionConfig(
            query_shape=(1, 16, 16, 128),
            key_shape=(2, 8, 8, 64),
            heads=4,
            value_dim=256,
            key_dim=128
        )
        
        # Verify properties are calculated correctly
        assert config.query_height == 16
        assert config.query_width == 16
        assert config.key_height == 8
        assert config.key_width == 8
        assert config.query_seq_len == 1
        assert config.key_seq_len == 2
        assert config.query_channels == 128
        assert config.key_channels == 64
    
    @pytest.mark.parametrize("attention_type", ["softmax", "linear"])
    def test_attention_types(self, attention_type: AttentionType):
        """Test both attention types."""
        config = DotProductAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64),
            query_dim=16,
            heads=8,
            attention_type=attention_type
        )
        
        assert config.attention_type == attention_type
    
    def test_configuration_serialization_compatibility(self):
        """Test that configurations work well with standard operations."""
        config = MultiHeadAttentionConfig(
            query_shape=(1, 8, 8, 64),
            key_shape=(1, 8, 8, 64)
        )
        
        # Test string representation
        config_str = str(config)
        assert "MultiHeadAttentionConfig" in config_str
        assert "(1, 8, 8, 64)" in config_str
        
        # Test repr
        config_repr = repr(config)
        assert "MultiHeadAttentionConfig" in config_repr 