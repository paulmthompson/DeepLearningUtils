"""
Tests for RelativePositionalEmbedding2D layer implementations.

This module tests both PyTorch and Keras implementations of the RelativePositionalEmbedding2D layer
to ensure they produce consistent results across different configurations.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pytest

# Set backend before importing frameworks
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
import torch

keras.config.set_floatx('float32')
keras.config.set_dtype_policy("float32")

from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_keras import (
    RelativePositionalEmbedding2D as KerasRelativePositionalEmbedding2D,
    RelativePositionalEmbedding2DConfig as KerasConfig,
    create_relative_positional_embedding_2d as create_keras_embedding,
)
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_pytorch import (
    RelativePositionalEmbedding2D as TorchRelativePositionalEmbedding2D,
    RelativePositionalEmbedding2DConfig as TorchConfig,
    create_relative_positional_embedding_2d as create_torch_embedding,
    load_positional_embedding_layer_weights,
)


class TestRelativePositionalEmbedding2DConfig:
    """Test configuration validation for RelativePositionalEmbedding2D layers."""
    
    def test_valid_config_creation(self) -> None:
        """Test that valid configurations can be created."""
        config = TorchConfig(
            query_shape=(4, 16, 16, 64),
            key_shape=(4, 16, 16, 64),
            query_dim=64,
            heads=8
        )
        assert config.query_shape == (4, 16, 16, 64)
        assert config.key_shape == (4, 16, 16, 64)
        assert config.query_dim == 64
        assert config.heads == 8
        assert config.drop_rate == 0.0
        
    def test_invalid_query_shape(self) -> None:
        """Test that invalid query shapes raise ValueError."""
        with pytest.raises(ValueError, match="query_shape must be 4-tuple"):
            TorchConfig(
                query_shape=(4, 16, 16),  # Only 3 elements
                key_shape=(4, 16, 16, 64),
                query_dim=64,
                heads=8
            )
            
        with pytest.raises(ValueError, match="query_shape values must be positive integers"):
            TorchConfig(
                query_shape=(4, 16, -16, 64),  # Negative value
                key_shape=(4, 16, 16, 64),
                query_dim=64,
                heads=8
            )
            
    def test_invalid_key_shape(self) -> None:
        """Test that invalid key shapes raise ValueError."""
        with pytest.raises(ValueError, match="key_shape must be 4-tuple"):
            TorchConfig(
                query_shape=(4, 16, 16, 64),
                key_shape=(4, 16),  # Only 2 elements
                query_dim=64,
                heads=8
            )
            
    def test_invalid_query_dim(self) -> None:
        """Test that invalid query dimensions raise ValueError."""
        with pytest.raises(ValueError, match="query_dim must be positive integer"):
            TorchConfig(
                query_shape=(4, 16, 16, 64),
                key_shape=(4, 16, 16, 64),
                query_dim=0,  # Zero
                heads=8
            )
            
    def test_invalid_heads(self) -> None:
        """Test that invalid head counts raise ValueError."""
        with pytest.raises(ValueError, match="heads must be positive integer"):
            TorchConfig(
                query_shape=(4, 16, 16, 64),
                key_shape=(4, 16, 16, 64),
                query_dim=64,
                heads=-1  # Negative
            )
            
    def test_invalid_drop_rate(self) -> None:
        """Test that invalid dropout rates raise ValueError."""
        with pytest.raises(ValueError, match="drop_rate must be float in \\[0, 1\\]"):
            TorchConfig(
                query_shape=(4, 16, 16, 64),
                key_shape=(4, 16, 16, 64),
                query_dim=64,
                heads=8,
                drop_rate=1.5  # > 1.0
            )


class TestRelativePositionalEmbedding2DImplementations:
    """Test RelativePositionalEmbedding2D layer implementations."""
    
    @pytest.mark.parametrize("query_shape,key_shape,query_dim,heads,drop_rate", [
        ((1, 32, 32, 64), (1, 32, 32, 64), 64, 8, 0.0),
        ((5, 16, 16, 128), (5, 16, 16, 128), 128, 4, 0.0),
        ((2, 8, 8, 32), (2, 8, 8, 32), 32, 2, 0.1),
    ])
    def test_positional_embedding_layer_consistency(
        self, 
        query_shape: Tuple[int, int, int, int], 
        key_shape: Tuple[int, int, int, int], 
        query_dim: int, 
        heads: int, 
        drop_rate: float
    ) -> None:
        """Test that PyTorch and Keras implementations produce consistent results."""
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create configurations
        keras_config = KerasConfig(
            query_shape=query_shape,
            key_shape=key_shape,
            query_dim=query_dim,
            heads=heads,
            drop_rate=drop_rate
        )
        torch_config = TorchConfig(
            query_shape=query_shape,
            key_shape=key_shape,
            query_dim=query_dim,
            heads=heads,
            drop_rate=drop_rate
        )
        
        # Create layers
        keras_layer = KerasRelativePositionalEmbedding2D(keras_config)
        torch_layer = TorchRelativePositionalEmbedding2D(torch_config)
        torch_layer.eval()
        
        # Create test data
        batch_size = 1
        query_seq_len, query_height, query_width, _ = query_shape
        key_seq_len, key_height, key_width, _ = key_shape
        
        keras_input_query = np.random.rand(
            batch_size, heads, query_seq_len * query_height * query_width, query_dim
        ).astype(np.float32)
        keras_input_scores = np.random.rand(
            batch_size, heads, 
            query_seq_len * query_height * query_width, 
            key_seq_len * key_height * key_width
        ).astype(np.float32)
        
        # Get Keras output
        keras_output = keras.ops.convert_to_numpy(
            keras_layer([keras_input_query, keras_input_scores])
        )
        
        # Load weights from Keras to PyTorch
        load_positional_embedding_layer_weights(keras_layer, torch_layer)
        
        # Convert inputs to PyTorch tensors
        torch_input_query = torch.tensor(keras_input_query)
        torch_input_scores = torch.tensor(keras_input_scores)
        
        # Get PyTorch output
        torch_output = torch_layer(torch_input_query, torch_input_scores).detach().numpy()
        
        # Compare outputs
        np.testing.assert_allclose(keras_output, torch_output, rtol=1e-5, atol=1e-3)
        
    def test_pytorch_jit_compilation(self) -> None:
        """Test that PyTorch layer can be JIT compiled."""
        config = TorchConfig(
            query_shape=(2, 8, 8, 32),
            key_shape=(2, 8, 8, 32),
            query_dim=32,
            heads=4
        )
        torch_layer = TorchRelativePositionalEmbedding2D(config)
        torch_layer.eval()
        
        # Create JIT compiled model
        pytorch_model_jit = torch.jit.script(torch_layer)
        
        # Test input
        query = torch.randn(1, 4, 128, 32)  # batch, heads, seq*h*w, dim
        scores = torch.randn(1, 4, 128, 128)  # batch, heads, q_tokens, k_tokens
        
        # Compare regular and JIT results
        regular_result = torch_layer(query, scores).detach().numpy()
        jit_result = pytorch_model_jit(query, scores).detach().numpy()
        
        np.testing.assert_allclose(regular_result, jit_result, rtol=1e-5, atol=1e-5)
        
    def test_input_validation(self) -> None:
        """Test that layers properly validate input dimensions."""
        config = TorchConfig(
            query_shape=(2, 8, 8, 32),
            key_shape=(2, 8, 8, 32),
            query_dim=32,
            heads=4
        )
        torch_layer = TorchRelativePositionalEmbedding2D(config)
        
        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Query must be 4D tensor"):
            torch_layer(torch.randn(1, 4, 32), torch.randn(1, 4, 128, 128))
            
        with pytest.raises(ValueError, match="Scores must be 4D tensor"):
            torch_layer(torch.randn(1, 4, 128, 32), torch.randn(1, 4, 128))
            
        # Test dimension mismatches
        with pytest.raises(ValueError, match="Query tokens mismatch"):
            torch_layer(torch.randn(1, 4, 64, 32), torch.randn(1, 4, 64, 128))  # Wrong q_tokens
            
        with pytest.raises(ValueError, match="Heads mismatch"):
            torch_layer(torch.randn(1, 8, 128, 32), torch.randn(1, 8, 128, 128))  # Wrong heads
            
        # Test Keras layer
        keras_config = KerasConfig(
            query_shape=(2, 8, 8, 32),
            key_shape=(2, 8, 8, 32),
            query_dim=32,
            heads=4
        )
        keras_layer = KerasRelativePositionalEmbedding2D(keras_config)
        
        with pytest.raises(ValueError, match="Expected 2 inputs"):
            keras_layer([keras.ops.ones((1, 4, 128, 32))])  # Only 1 input
            
        with pytest.raises(ValueError, match="Query must be 4D tensor"):
            keras_layer([keras.ops.ones((1, 4, 32)), keras.ops.ones((1, 4, 128, 128))])


class TestBackwardCompatibility:
    """Test backward compatibility with convenience functions."""
    
    @pytest.mark.parametrize("query_shape,key_shape,query_dim,heads,drop_rate", [
        ((1, 32, 32, 64), (1, 32, 32, 64), 64, 8, 0.0),
        ((2, 16, 16, 128), (2, 16, 16, 128), 128, 4, 0.1),
    ])
    def test_convenience_constructors(
        self, 
        query_shape: Tuple[int, int, int, int], 
        key_shape: Tuple[int, int, int, int], 
        query_dim: int, 
        heads: int, 
        drop_rate: float
    ) -> None:
        """Test that convenience constructors work correctly."""
        # Test PyTorch convenience constructor
        torch_layer = create_torch_embedding(
            query_shape=query_shape,
            key_shape=key_shape,
            query_dim=query_dim,
            heads=heads,
            drop_rate=drop_rate
        )
        
        # Test Keras convenience constructor
        keras_layer = create_keras_embedding(
            query_shape=query_shape,
            key_shape=key_shape,
            query_dim=query_dim,
            heads=heads,
            drop_rate=drop_rate
        )
        
        # Verify configurations
        assert torch_layer.config.query_shape == query_shape
        assert torch_layer.config.key_shape == key_shape
        assert torch_layer.config.query_dim == query_dim
        assert torch_layer.config.heads == heads
        assert torch_layer.config.drop_rate == drop_rate
        
        assert keras_layer.config.query_shape == query_shape
        assert keras_layer.config.key_shape == key_shape
        assert keras_layer.config.query_dim == query_dim
        assert keras_layer.config.heads == heads
        assert keras_layer.config.drop_rate == drop_rate
        
    def test_keras_serialization(self) -> None:
        """Test that Keras layer can be serialized and deserialized."""
        config = KerasConfig(
            query_shape=(2, 8, 8, 32),
            key_shape=(2, 8, 8, 32),
            query_dim=32,
            heads=4,
            drop_rate=0.1
        )
        original_layer = KerasRelativePositionalEmbedding2D(config)
        
        # Get configuration
        layer_config = original_layer.get_config()
        
        # Recreate layer from config
        recreated_layer = KerasRelativePositionalEmbedding2D.from_config(layer_config)
        
        # Verify configurations match
        assert recreated_layer.config.query_shape == original_layer.config.query_shape
        assert recreated_layer.config.key_shape == original_layer.config.key_shape
        assert recreated_layer.config.query_dim == original_layer.config.query_dim
        assert recreated_layer.config.heads == original_layer.config.heads
        assert recreated_layer.config.drop_rate == original_layer.config.drop_rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])