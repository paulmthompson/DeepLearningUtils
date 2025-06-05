"""
Tests for Keras 2D Rotary Positional Encoding implementation.

This module tests the RotaryPositionalEncoding2D layer to ensure it adheres to
company coding guidelines and functions correctly.
"""

import pytest
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras

from src.DeepLearningUtils.Layers.RotaryPositionalEncoding.rotary_config import (
    RotaryPositionalEncodingConfig
)
from src.DeepLearningUtils.Layers.RotaryPositionalEncoding.rotary2d_keras import (
    RotaryPositionalEncoding2D
)


class TestRotaryPositionalEncodingConfig:
    """Test suite for RotaryPositionalEncodingConfig dataclass."""

    def test_config_valid_parameters(self) -> None:
        """Test configuration with valid parameters."""
        config = RotaryPositionalEncodingConfig(
            dim=64, height=16, width=16, theta=10.0, rotate=True, max_freq=64
        )
        
        assert config.dim == 64
        assert config.height == 16
        assert config.width == 16
        assert config.theta == 10.0
        assert config.rotate is True
        assert config.max_freq == 64
        assert config.spatial_size == 256
        assert config.freq_dim == 32

    def test_config_default_parameters(self) -> None:
        """Test configuration with default parameters."""
        config = RotaryPositionalEncodingConfig(dim=32, height=8, width=8)
        
        assert config.theta == 10.0
        assert config.rotate is True
        assert config.max_freq == 64

    @pytest.mark.parametrize("dim,expected_error", [
        (-1, ValueError),
        (0, ValueError),
        (33, ValueError),  # Odd number
        (2.5, TypeError),  # Float
        ("64", TypeError),  # String
    ])
    def test_config_invalid_dim(self, dim, expected_error) -> None:
        """Test configuration validation for invalid dim values."""
        with pytest.raises(expected_error):
            RotaryPositionalEncodingConfig(dim=dim, height=16, width=16)

    @pytest.mark.parametrize("height,expected_error", [
        (-1, ValueError),
        (0, ValueError),
        (2.5, TypeError),
        ("16", TypeError),
    ])
    def test_config_invalid_height(self, height, expected_error) -> None:
        """Test configuration validation for invalid height values."""
        with pytest.raises(expected_error):
            RotaryPositionalEncodingConfig(dim=64, height=height, width=16)

    @pytest.mark.parametrize("width,expected_error", [
        (-1, ValueError),
        (0, ValueError),
        (2.5, TypeError),
        ("16", TypeError),
    ])
    def test_config_invalid_width(self, width, expected_error) -> None:
        """Test configuration validation for invalid width values."""
        with pytest.raises(expected_error):
            RotaryPositionalEncodingConfig(dim=64, height=16, width=width)

    @pytest.mark.parametrize("theta,expected_error", [
        (-1.0, ValueError),
        (0.0, ValueError),
        ("10.0", TypeError),
    ])
    def test_config_invalid_theta(self, theta, expected_error) -> None:
        """Test configuration validation for invalid theta values."""
        with pytest.raises(expected_error):
            RotaryPositionalEncodingConfig(dim=64, height=16, width=16, theta=theta)

    def test_config_invalid_rotate(self) -> None:
        """Test configuration validation for invalid rotate values."""
        with pytest.raises(TypeError):
            RotaryPositionalEncodingConfig(dim=64, height=16, width=16, rotate="true")

    @pytest.mark.parametrize("max_freq,expected_error", [
        (-1, ValueError),
        (0, ValueError),
        (2.5, TypeError),
        ("64", TypeError),
    ])
    def test_config_invalid_max_freq(self, max_freq, expected_error) -> None:
        """Test configuration validation for invalid max_freq values."""
        with pytest.raises(expected_error):
            RotaryPositionalEncodingConfig(dim=64, height=16, width=16, max_freq=max_freq)


class TestRotaryPositionalEncoding2D:
    """Test suite for RotaryPositionalEncoding2D layer."""

    def test_layer_initialization_with_config(self) -> None:
        """Test layer initialization using configuration object."""
        config = RotaryPositionalEncodingConfig(dim=64, height=16, width=16)
        layer = RotaryPositionalEncoding2D(config=config)
        
        assert layer.config == config
        assert layer.dim == 64
        assert layer.height == 16
        assert layer.width == 16

    def test_layer_initialization_with_parameters(self) -> None:
        """Test layer initialization using direct parameters."""
        layer = RotaryPositionalEncoding2D(dim=128, height=8, width=8, theta=5.0)
        
        assert layer.dim == 128
        assert layer.height == 8
        assert layer.width == 8
        assert layer.theta == 5.0

    def test_layer_initialization_missing_parameters(self) -> None:
        """Test layer initialization with missing required parameters."""
        with pytest.raises(ValueError, match="Either config must be provided"):
            RotaryPositionalEncoding2D(dim=64, height=16)  # Missing width

    @pytest.mark.parametrize("batch_size,seq_len,height,width,dim", [
        (1, 10, 8, 8, 64),
        (4, 5, 16, 16, 128),
        (2, 1, 4, 8, 32),
    ])
    def test_layer_forward_pass_valid_shapes(
        self, batch_size: int, seq_len: int, height: int, width: int, dim: int
    ) -> None:
        """Test forward pass with valid input shapes."""
        layer = RotaryPositionalEncoding2D(dim=dim, height=height, width=width)
        
        # Create input tensor
        input_shape = (batch_size, seq_len, height * width, dim)
        q = keras.ops.convert_to_tensor(
            np.random.randn(*input_shape).astype(np.float32)
        )
        
        # Forward pass
        output = layer(q)
        
        # Check output shape
        assert keras.ops.shape(output) == keras.ops.shape(q)
        assert output.dtype == q.dtype

    def test_layer_build_validation_wrong_dimensions(self) -> None:
        """Test layer build validation with wrong input dimensions."""
        layer = RotaryPositionalEncoding2D(dim=64, height=8, width=8)
        
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer.build((32, 64, 128))

    def test_layer_build_validation_wrong_dim(self) -> None:
        """Test layer build validation with wrong last dimension."""
        layer = RotaryPositionalEncoding2D(dim=64, height=8, width=8)
        
        # Wrong last dimension
        with pytest.raises(ValueError, match="Expected last dimension to be 64"):
            layer.build((1, 10, 64, 128))

    def test_layer_build_validation_wrong_spatial_size(self) -> None:
        """Test layer build validation with wrong spatial dimension."""
        layer = RotaryPositionalEncoding2D(dim=64, height=8, width=8)
        
        # Wrong spatial dimension (should be 8*8=64)
        with pytest.raises(ValueError, match="Expected spatial dimension to be 64"):
            layer.build((1, 10, 32, 64))

    def test_layer_call_validation_wrong_ndim(self) -> None:
        """Test call validation with wrong number of dimensions."""
        layer = RotaryPositionalEncoding2D(dim=64, height=8, width=8)
        
        # 3D tensor instead of 4D
        q = keras.ops.convert_to_tensor(np.random.randn(32, 64, 128).astype(np.float32))
        
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer(q)

    def test_layer_get_config(self) -> None:
        """Test layer configuration serialization."""
        layer = RotaryPositionalEncoding2D(
            dim=64, height=16, width=16, theta=5.0, rotate=False, max_freq=32
        )
        
        config = layer.get_config()
        
        assert config['dim'] == 64
        assert config['height'] == 16
        assert config['width'] == 16
        assert config['theta'] == 5.0
        assert config['rotate'] is False
        assert config['max_freq'] == 32

    def test_layer_output_deterministic_when_rotate_false(self) -> None:
        """Test that output is deterministic when rotate=False."""
        layer = RotaryPositionalEncoding2D(dim=64, height=8, width=8, rotate=False)
        
        q = keras.ops.convert_to_tensor(
            np.random.randn(1, 5, 64, 64).astype(np.float32)
        )
        
        output1 = keras.ops.convert_to_numpy(layer(q))
        output2 = keras.ops.convert_to_numpy(layer(q))
        
        np.testing.assert_array_equal(output1, output2)

    def test_layer_output_shape_preservation(self) -> None:
        """Test that output preserves input shape exactly."""
        layer = RotaryPositionalEncoding2D(dim=32, height=4, width=4)
        
        input_shape = (2, 3, 16, 32)
        q = keras.ops.convert_to_tensor(
            np.random.randn(*input_shape).astype(np.float32)
        )
        
        output = layer(q)
        
        assert tuple(keras.ops.shape(output)) == input_shape

    def test_layer_different_heights_widths(self) -> None:
        """Test layer with non-square spatial dimensions."""
        layer = RotaryPositionalEncoding2D(dim=64, height=4, width=8)
        
        q = keras.ops.convert_to_tensor(
            np.random.randn(1, 2, 32, 64).astype(np.float32)  # 4*8=32
        )
        
        output = layer(q)
        assert keras.ops.shape(output) == keras.ops.shape(q)

    @pytest.mark.parametrize("theta", [1.0, 5.0, 10.0, 100.0])
    def test_layer_different_theta_values(self, theta: float) -> None:
        """Test layer with different theta values."""
        layer = RotaryPositionalEncoding2D(dim=64, height=8, width=8, theta=theta)
        
        q = keras.ops.convert_to_tensor(
            np.random.randn(1, 5, 64, 64).astype(np.float32)
        )
        
        output = layer(q)
        assert keras.ops.shape(output) == keras.ops.shape(q)

    def test_layer_rotary_properties(self) -> None:
        """Test that rotary embeddings have expected mathematical properties."""
        layer = RotaryPositionalEncoding2D(dim=4, height=2, width=2, rotate=False)
        
        # Simple input to track transformations
        q = keras.ops.convert_to_tensor(
            np.ones((1, 1, 4, 4), dtype=np.float32)
        )
        
        output = keras.ops.convert_to_numpy(layer(q))
        
        # Output should be different from input (rotation applied)
        assert not np.allclose(output, keras.ops.convert_to_numpy(q))
        
        # Output should have same norm as input for each position/channel pair
        # (rotations preserve magnitude)
        input_norms = np.linalg.norm(
            keras.ops.convert_to_numpy(q).reshape(1, 1, 4, 2, 2), axis=-1
        )
        output_norms = np.linalg.norm(output.reshape(1, 1, 4, 2, 2), axis=-1)
        
        np.testing.assert_allclose(input_norms, output_norms, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__]) 