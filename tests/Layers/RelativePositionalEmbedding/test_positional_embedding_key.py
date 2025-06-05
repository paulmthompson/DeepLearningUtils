"""
Tests for RelativePositionalEmbedding2DKey layer implementations.

This module tests both PyTorch and Keras implementations of the RelativePositionalEmbedding2DKey layer
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

from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_keras import (
    RelativePositionalEmbedding2DKey as KerasRelativePositionalEmbedding2DKey,
    RelativePositionalEmbedding2DConfig as KerasConfig,
    create_relative_positional_embedding_2d_key as create_keras_embedding_key,
)
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_pytorch import (
    RelativePositionalEmbedding2DKey as TorchRelativePositionalEmbedding2DKey,
    RelativePositionalEmbedding2DConfig as TorchConfig,
    create_relative_positional_embedding_2d_key as create_torch_embedding_key,
    load_positional_embedding_layer_weights,
)


class TestRelativePositionalEmbedding2DKeyImplementations:
    """Test RelativePositionalEmbedding2DKey layer implementations."""
    
    @pytest.mark.parametrize("query_shape,key_shape,query_dim,heads,drop_rate", [
        ((1, 32, 32, 64), (1, 32, 32, 64), 64, 8, 0.0),
        ((5, 16, 16, 128), (5, 16, 16, 128), 128, 4, 0.0),
        ((2, 8, 8, 32), (2, 8, 8, 32), 32, 2, 0.1),
    ])
    def test_positional_embedding_layer_key_consistency(
        self, 
        query_shape: Tuple[int, int, int, int], 
        key_shape: Tuple[int, int, int, int], 
        query_dim: int, 
        heads: int, 
        drop_rate: float
    ) -> None:
        """Test that PyTorch and Keras key-based implementations produce consistent results."""
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
        keras_layer = KerasRelativePositionalEmbedding2DKey(keras_config)
        torch_layer = TorchRelativePositionalEmbedding2DKey(torch_config)
        torch_layer.eval()
        
        # Create test data
        batch_size = 1
        key_seq_len, key_height, key_width, _ = key_shape
        query_seq_len, query_height, query_width, _ = query_shape
        
        keras_input_key = np.random.rand(
            batch_size, heads, key_seq_len * key_height * key_width, query_dim
        ).astype(np.float32)
        keras_input_scores = np.random.rand(
            batch_size, heads, 
            query_seq_len * query_height * query_width, 
            key_seq_len * key_height * key_width
        ).astype(np.float32)
        
        # Create and run Keras model
        keras_input = [
            keras.Input(shape=keras_input_key.shape[1:]),
            keras.Input(shape=keras_input_scores.shape[1:])
        ]
        keras_output = keras_layer(keras_input)
        keras_model = keras.Model(inputs=keras_input, outputs=keras_output)
        
        keras_output = keras_model.predict([keras_input_key, keras_input_scores], verbose=0)
        
        # Load weights from Keras to PyTorch
        load_positional_embedding_layer_weights(keras_layer, torch_layer)
        
        # Convert inputs to PyTorch tensors
        torch_input_key = torch.tensor(keras_input_key)
        torch_input_scores = torch.tensor(keras_input_scores)
        
        # Get PyTorch output
        torch_output = torch_layer(torch_input_key, torch_input_scores).detach().numpy()
        
        # Compare outputs
        np.testing.assert_allclose(keras_output, torch_output, rtol=1e-5, atol=1e-3)
        
    def test_pytorch_jit_compilation_key(self) -> None:
        """Test that PyTorch key-based layer can be JIT compiled."""
        config = TorchConfig(
            query_shape=(2, 8, 8, 32),
            key_shape=(2, 8, 8, 32),
            query_dim=32,
            heads=4
        )
        torch_layer = TorchRelativePositionalEmbedding2DKey(config)
        torch_layer.eval()
        
        # Create JIT compiled model
        pytorch_model_jit = torch.jit.script(torch_layer)
        
        # Test input
        key = torch.randn(1, 4, 128, 32)  # batch, heads, seq*h*w, dim
        scores = torch.randn(1, 4, 128, 128)  # batch, heads, q_tokens, k_tokens
        
        # Compare regular and JIT results
        regular_result = torch_layer(key, scores).detach().numpy()
        jit_result = pytorch_model_jit(key, scores).detach().numpy()
        
        np.testing.assert_allclose(regular_result, jit_result, rtol=1e-5, atol=1e-5)
        
    def test_input_validation_key(self) -> None:
        """Test that key-based layers properly validate input dimensions."""
        config = TorchConfig(
            query_shape=(2, 8, 8, 32),
            key_shape=(2, 8, 8, 32),
            query_dim=32,
            heads=4
        )
        torch_layer = TorchRelativePositionalEmbedding2DKey(config)
        
        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Key must be 4D tensor"):
            torch_layer(torch.randn(1, 4, 32), torch.randn(1, 4, 128, 128))
            
        with pytest.raises(ValueError, match="Scores must be 4D tensor"):
            torch_layer(torch.randn(1, 4, 128, 32), torch.randn(1, 4, 128))
            
        # Test dimension mismatches
        with pytest.raises(ValueError, match="Key tokens mismatch"):
            torch_layer(torch.randn(1, 4, 64, 32), torch.randn(1, 4, 128, 64))  # Wrong k_tokens
            
        with pytest.raises(ValueError, match="Heads mismatch"):
            torch_layer(torch.randn(1, 8, 128, 32), torch.randn(1, 8, 128, 128))  # Wrong heads


class TestBackwardCompatibilityKey:
    """Test backward compatibility with convenience functions for key-based layers."""
    
    @pytest.mark.parametrize("query_shape,key_shape,query_dim,heads,drop_rate", [
        ((1, 32, 32, 64), (1, 32, 32, 64), 64, 8, 0.0),
        ((2, 16, 16, 128), (2, 16, 16, 128), 128, 4, 0.1),
    ])
    def test_convenience_constructors_key(
        self, 
        query_shape: Tuple[int, int, int, int], 
        key_shape: Tuple[int, int, int, int], 
        query_dim: int, 
        heads: int, 
        drop_rate: float
    ) -> None:
        """Test that key-based convenience constructors work correctly."""
        # Test PyTorch convenience constructor
        torch_layer = create_torch_embedding_key(
            query_shape=query_shape,
            key_shape=key_shape,
            query_dim=query_dim,
            heads=heads,
            drop_rate=drop_rate
        )
        
        # Test Keras convenience constructor
        keras_layer = create_keras_embedding_key(
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
        
    def test_keras_serialization_key(self) -> None:
        """Test that Keras key-based layer can be serialized and deserialized."""
        config = KerasConfig(
            query_shape=(2, 8, 8, 32),
            key_shape=(2, 8, 8, 32),
            query_dim=32,
            heads=4,
            drop_rate=0.1
        )
        original_layer = KerasRelativePositionalEmbedding2DKey(config)
        
        # Get configuration
        layer_config = original_layer.get_config()
        
        # Recreate layer from config
        recreated_layer = KerasRelativePositionalEmbedding2DKey.from_config(layer_config)
        
        # Verify configurations match
        assert recreated_layer.config.query_shape == original_layer.config.query_shape
        assert recreated_layer.config.key_shape == original_layer.config.key_shape
        assert recreated_layer.config.query_dim == original_layer.config.query_dim
        assert recreated_layer.config.heads == original_layer.config.heads
        assert recreated_layer.config.drop_rate == original_layer.config.drop_rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])