"""
Tests for Blur2D layer implementations.

This module tests both PyTorch and Keras implementations of the Blur2D layer
to ensure they produce consistent results across different kernel types.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pytest

# Set backend before importing frameworks
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
import torch

from src.DeepLearningUtils.Layers.Blur2D.blur2d_keras import (
    Blur2D as Blur2D_Keras,
    Blur2DConfig as Blur2DConfig_Keras,
    create_blur2d as create_keras_blur2d,
)
from src.DeepLearningUtils.Layers.Blur2D.blur2d_pytorch import (
    Blur2D as Blur2D_PyTorch,
    Blur2DConfig as Blur2DConfig_PyTorch,
    create_blur2d as create_pytorch_blur2d,
)


class TestBlur2DConfig:
    """Test configuration validation for Blur2D layers."""
    
    def test_valid_config_creation(self) -> None:
        """Test that valid configurations can be created."""
        config = Blur2DConfig_PyTorch(kernel_size=3, stride=2, kernel_type="Triangle")
        assert config.kernel_size == 3
        assert config.stride == 2
        assert config.kernel_type == "Triangle"
        
    def test_invalid_kernel_size(self) -> None:
        """Test that invalid kernel sizes raise ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be a positive integer"):
            Blur2DConfig_PyTorch(kernel_size=0)
            
        with pytest.raises(ValueError, match="kernel_size must be a positive integer"):
            Blur2DConfig_PyTorch(kernel_size=-1)
            
    def test_invalid_stride(self) -> None:
        """Test that invalid strides raise ValueError."""
        with pytest.raises(ValueError, match="stride must be a positive integer"):
            Blur2DConfig_PyTorch(stride=0)
            
    def test_triangle_kernel_validation(self) -> None:
        """Test that Triangle kernel validates kernel size."""
        with pytest.raises(ValueError, match="Triangle kernel requires kernel_size > 2"):
            Blur2DConfig_PyTorch(kernel_size=2, kernel_type="Triangle")
            
    def test_binomial_kernel_validation(self) -> None:
        """Test that Binomial kernel validates kernel size."""
        with pytest.raises(ValueError, match="Binomial kernel only supports kernel_size = 5"):
            Blur2DConfig_PyTorch(kernel_size=3, kernel_type="Binomial")


class TestBlur2DImplementations:
    """Test Blur2D layer implementations."""
    
    @pytest.mark.parametrize("kernel_type,kernel_size", [
        ("Rect", 2),
        ("Rect", 5), 
        ("Triangle", 3),
        ("Triangle", 5),
        ("Binomial", 5)
    ])
    def test_blur2d_layers_consistency(
        self, 
        kernel_type: Literal["Rect", "Triangle", "Binomial"], 
        kernel_size: int
    ) -> None:
        """Test that PyTorch and Keras implementations produce consistent results."""
        # Initialize input data
        np.random.seed(42)  # For reproducible tests
        input_data = np.random.rand(1, 10, 10, 3).astype(np.float32)
        
        # Create configurations
        keras_config = Blur2DConfig_Keras(
            kernel_size=kernel_size,
            stride=2,
            kernel_type=kernel_type,
            padding="same"
        )
        pytorch_config = Blur2DConfig_PyTorch(
            kernel_size=kernel_size,
            stride=2,
            kernel_type=kernel_type,
            padding="same"
        )
        
        # Keras Blur2D
        keras_layer = Blur2D_Keras(keras_config)
        keras_input = keras.Input(shape=(10, 10, 3))
        keras_output = keras_layer(keras_input)
        keras_model = keras.Model(inputs=keras_input, outputs=keras_output)
        keras_result = keras_model.predict(input_data, verbose=0)
        
        # PyTorch Blur2D
        pytorch_layer = Blur2D_PyTorch(pytorch_config)
        pytorch_input = torch.tensor(input_data.transpose(0, 3, 1, 2))
        pytorch_result = pytorch_layer(pytorch_input).detach().numpy().transpose(0, 2, 3, 1)
        
        # Compare results
        np.testing.assert_allclose(keras_result, pytorch_result, rtol=1e-5, atol=1e-5)
        
    def test_pytorch_jit_compilation(self) -> None:
        """Test that PyTorch layer can be JIT compiled."""
        config = Blur2DConfig_PyTorch(kernel_size=3, stride=2, kernel_type="Rect")
        pytorch_layer = Blur2D_PyTorch(config)
        
        # Create JIT compiled model
        pytorch_model = torch.jit.script(pytorch_layer)
        
        # Test input
        input_tensor = torch.randn(1, 3, 10, 10)
        
        # Compare regular and JIT results
        regular_result = pytorch_layer(input_tensor).detach().numpy()
        jit_result = pytorch_model(input_tensor).detach().numpy()
        
        np.testing.assert_allclose(regular_result, jit_result, rtol=1e-5, atol=1e-5)
        
    def test_input_validation(self) -> None:
        """Test that layers properly validate input dimensions."""
        config = Blur2DConfig_PyTorch(kernel_size=3, stride=2)
        pytorch_layer = Blur2D_PyTorch(config)
        
        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Input must be 4D tensor"):
            pytorch_layer(torch.randn(1, 3, 10))  # 3D tensor
            
        # Test Keras layer
        keras_config = Blur2DConfig_Keras(kernel_size=3, stride=2)
        keras_layer = Blur2D_Keras(keras_config)
        
        # Build the layer first
        keras_layer.build((1, 10, 10, 3))
        
        with pytest.raises(ValueError, match="Input must be 4D tensor"):
            keras_layer(keras.ops.ones((1, 10, 3)))  # 3D tensor


class TestBackwardCompatibility:
    """Test backward compatibility with convenience functions."""
    
    @pytest.mark.parametrize("kernel_type,kernel_size", [
        ("Rect", 2),
        ("Triangle", 3),
        ("Binomial", 5)
    ])
    def test_convenience_constructors(
        self, 
        kernel_type: Literal["Rect", "Triangle", "Binomial"], 
        kernel_size: int
    ) -> None:
        """Test that convenience constructors work correctly."""
        # Test PyTorch convenience constructor
        pytorch_layer = create_pytorch_blur2d(
            kernel_size=kernel_size,
            stride=2,
            kernel_type=kernel_type,
            padding="same"
        )
        
        # Test Keras convenience constructor
        keras_layer = create_keras_blur2d(
            kernel_size=kernel_size,
            stride=2,
            kernel_type=kernel_type,
            padding="same"
        )
        
        # Verify configurations
        assert pytorch_layer.config.kernel_size == kernel_size
        assert pytorch_layer.config.kernel_type == kernel_type
        assert keras_layer.config.kernel_size == kernel_size
        assert keras_layer.config.kernel_type == kernel_type
        
    def test_keras_serialization(self) -> None:
        """Test that Keras layer can be serialized and deserialized."""
        config = Blur2DConfig_Keras(kernel_size=3, stride=2, kernel_type="Triangle")
        original_layer = Blur2D_Keras(config)
        
        # Get configuration
        layer_config = original_layer.get_config()
        
        # Recreate layer from config
        recreated_layer = Blur2D_Keras.from_config(layer_config)
        
        # Verify configurations match
        assert recreated_layer.config.kernel_size == original_layer.config.kernel_size
        assert recreated_layer.config.stride == original_layer.config.stride
        assert recreated_layer.config.kernel_type == original_layer.config.kernel_type
        assert recreated_layer.config.padding == original_layer.config.padding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])