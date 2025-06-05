from __future__ import annotations

"""
Copyright 2019, Facebook, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

"""
This keras implementation of relative positional encoding from MVit2 was
adapted from the original pytorch implementation:
https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py

Adapted by Paul Thompson 2024
Improved by following company Python guidelines 2024
"""

from dataclasses import dataclass
from typing import Any, List, Tuple

import keras
import numpy as np


@dataclass(frozen=True)
class RelativePositionalEmbedding2DConfig:
    """Configuration for RelativePositionalEmbedding2D layer.
    
    Attributes:
        query_shape: Shape of query tensor (seq_len, height, width, channels).
        key_shape: Shape of key tensor (seq_len, height, width, channels).
        query_dim: Dimension of the query embeddings.
        heads: Number of attention heads.
        drop_rate: Dropout rate applied to embeddings.
    """
    query_shape: Tuple[int, int, int, int]
    key_shape: Tuple[int, int, int, int]
    query_dim: int
    heads: int
    drop_rate: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
            TypeError: If parameters have wrong types.
        """
        # Validate query_shape
        if not isinstance(self.query_shape, tuple) or len(self.query_shape) != 4:
            raise ValueError(f"query_shape must be 4-tuple, got {self.query_shape}")
        if not all(isinstance(x, int) and x > 0 for x in self.query_shape):
            raise ValueError(f"query_shape values must be positive integers, got {self.query_shape}")
            
        # Validate key_shape
        if not isinstance(self.key_shape, tuple) or len(self.key_shape) != 4:
            raise ValueError(f"key_shape must be 4-tuple, got {self.key_shape}")
        if not all(isinstance(x, int) and x > 0 for x in self.key_shape):
            raise ValueError(f"key_shape values must be positive integers, got {self.key_shape}")
            
        # Validate query_dim
        if not isinstance(self.query_dim, int) or self.query_dim <= 0:
            raise ValueError(f"query_dim must be positive integer, got {self.query_dim}")
            
        # Validate heads
        if not isinstance(self.heads, int) or self.heads <= 0:
            raise ValueError(f"heads must be positive integer, got {self.heads}")
            
        # Validate drop_rate
        if not isinstance(self.drop_rate, (int, float)) or not (0.0 <= self.drop_rate <= 1.0):
            raise ValueError(f"drop_rate must be float in [0, 1], got {self.drop_rate}")


class RelativePositionalEmbedding2D(keras.layers.Layer):
    """Relative positional embedding for 2D spatiotemporal attention.
    
    This layer computes relative positional embeddings for query-key attention
    in 2D space with temporal dimension. Based on MViT2 implementation.
    
    Args:
        config: Configuration object containing all layer parameters.
        **kwargs: Additional keyword arguments passed to parent Layer.
        
    Raises:
        ValueError: If configuration parameters are invalid.
        
    Example:
        >>> config = RelativePositionalEmbedding2DConfig(
        ...     query_shape=(4, 16, 16, 64),
        ...     key_shape=(4, 16, 16, 64),
        ...     query_dim=64,
        ...     heads=8
        ... )
        >>> layer = RelativePositionalEmbedding2D(config)
        >>> query = keras.random.uniform((1, 8, 1024, 64))
        >>> scores = keras.random.uniform((1, 8, 1024, 1024))
        >>> output = layer([query, scores])
    """
    
    def __init__(self, config: RelativePositionalEmbedding2DConfig, **kwargs: Any) -> None:
        """Initialize RelativePositionalEmbedding2D layer.
        
        Pre-conditions:
            - config must be a valid RelativePositionalEmbedding2DConfig instance
            
        Post-conditions:
            - Layer is ready for forward pass
            - Embedding weights are properly initialized
        """
        super().__init__(**kwargs)
        
        if not isinstance(config, RelativePositionalEmbedding2DConfig):
            raise TypeError(f"config must be RelativePositionalEmbedding2DConfig, got {type(config)}")
            
        self.config = config
        
        # Store config values
        self.query_seq_len, self.query_height, self.query_width, self.query_channels = config.query_shape
        self.key_seq_len, self.key_height, self.key_width, self.key_channels = config.key_shape
        self.query_dim = config.query_dim
        self.heads = config.heads
        self.drop_rate = config.drop_rate
        
        # Calculate embedding dimensions
        max_height_dist = max(self.query_height, self.key_height) - 1
        max_width_dist = max(self.query_width, self.key_width) - 1
        max_t_dist = max(self.query_seq_len, self.key_seq_len) - 1
        
        # Create embedding weights
        self.height_embeddings = self.add_weight(
            name='height_embeddings',
            shape=(2 * max_height_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        self.width_embeddings = self.add_weight(
            name='width_embeddings',
            shape=(2 * max_width_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        self.time_embeddings = self.add_weight(
            name='time_embeddings',
            shape=(2 * max_t_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        
        # Dropout layers
        self.height_dropout = keras.layers.Dropout(rate=self.drop_rate)
        self.width_dropout = keras.layers.Dropout(rate=self.drop_rate)
        self.time_dropout = keras.layers.Dropout(rate=self.drop_rate)
        
        # Reshape layers
        self.query_reshape_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len, self.query_height, self.query_width, self.query_dim)
        )
        
        self.score_reshape_pre_embedding_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len, self.query_height, self.query_width, 
             self.key_seq_len, self.key_height, self.key_width)
        )
        
        self.score_reshape_after_embedding_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len * self.query_height * self.query_width, 
             self.key_seq_len * self.key_height * self.key_width)
        )

    def _compute_relative_distances(self) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Compute relative distance matrices for height, width, and time.
        
        Returns:
            Tuple of (height_distances, width_distances, time_distances) tensors.
        """
        # Height distances
        q_h_ratio = round(max(self.key_height / self.query_height, 1.0))
        k_h_ratio = round(max(self.query_height / self.key_height, 1.0))
        dist_h = (
            keras.ops.arange(self.query_height)[:, None] * q_h_ratio 
            - keras.ops.arange(self.key_height)[None, :] * k_h_ratio
        )
        dist_h += (self.key_height - 1) * k_h_ratio
        
        # Width distances
        q_w_ratio = round(max(self.key_width / self.query_width, 1.0))
        k_w_ratio = round(max(self.query_width / self.key_width, 1.0))
        dist_w = (
            keras.ops.arange(self.query_width)[:, None] * q_w_ratio 
            - keras.ops.arange(self.key_width)[None, :] * k_w_ratio
        )
        dist_w += (self.key_width - 1) * k_w_ratio
        
        # Time distances
        q_t_ratio = round(max(self.key_seq_len / self.query_seq_len, 1.0))
        k_t_ratio = round(max(self.query_seq_len / self.key_seq_len, 1.0))
        dist_t = (
            keras.ops.arange(self.query_seq_len)[:, None] * q_t_ratio 
            - keras.ops.arange(self.key_seq_len)[None, :] * k_t_ratio
        )
        dist_t += (self.key_seq_len - 1) * k_t_ratio
        
        return dist_h, dist_w, dist_t

    def build(self, input_shape: List[Tuple[int, ...]]) -> None:
        """Build layer by validating input shapes.
        
        Args:
            input_shape: List of shapes for [query, scores] inputs.
            
        Pre-conditions:
            - input_shape must contain exactly 2 shapes
            - Shapes must match expected dimensions
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2 inputs [query, scores], got {len(input_shape)}")
            
        query_shape, scores_shape = input_shape
        
        if len(query_shape) != 4:
            raise ValueError(f"Query must be 4D tensor, got {len(query_shape)}D")
        if len(scores_shape) != 4:
            raise ValueError(f"Scores must be 4D tensor, got {len(scores_shape)}D")
            
        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """Apply relative positional embeddings to attention scores.
        
        Args:
            inputs: List containing [query, scores] tensors.
            
        Returns:
            Updated attention scores with relative positional bias.
            
        Pre-conditions:
            - inputs must contain exactly 2 tensors [query, scores]
            - query must be 4D tensor (batch, heads, seq*h*w, query_dim)
            - scores must be 4D tensor (batch, heads, q_tokens, k_tokens)
            
        Post-conditions:
            - Output has same shape as input scores
            - Relative positional bias is added to scores
        """
        if len(inputs) != 2:
            raise ValueError(f"Expected 2 inputs [query, scores], got {len(inputs)}")
            
        query, scores = inputs
        
        if query.ndim != 4:
            raise ValueError(f"Query must be 4D tensor, got {query.ndim}D")
        if scores.ndim != 4:
            raise ValueError(f"Scores must be 4D tensor, got {scores.ndim}D")
            
        # Validate dimensions
        batch_size, heads, q_tokens, query_dim = query.shape
        _, _, _, k_tokens = scores.shape
        
        expected_q_tokens = self.query_seq_len * self.query_height * self.query_width
        expected_k_tokens = self.key_seq_len * self.key_height * self.key_width
        
        if q_tokens != expected_q_tokens:
            raise ValueError(f"Query tokens mismatch: expected {expected_q_tokens}, got {q_tokens}")
        if k_tokens != expected_k_tokens:
            raise ValueError(f"Key tokens mismatch: expected {expected_k_tokens}, got {k_tokens}")
        if query_dim != self.query_dim:
            raise ValueError(f"Query dim mismatch: expected {self.query_dim}, got {query_dim}")
        if heads != self.heads:
            raise ValueError(f"Heads mismatch: expected {self.heads}, got {heads}")
            
        # Compute relative distances
        dist_h, dist_w, dist_t = self._compute_relative_distances()
        
        # Get embeddings
        Rh = keras.ops.take(self.height_embeddings, dist_h, axis=0)
        Rw = keras.ops.take(self.width_embeddings, dist_w, axis=0)
        Rt = keras.ops.take(self.time_embeddings, dist_t, axis=0)
        
        # Apply dropout
        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)
        
        # Reshape query for computation
        query_reshaped = self.query_reshape_layer(query)
        
        # Compute relative attention terms
        rel_h = keras.ops.einsum("bNthwc,hkc->bNthwk", query_reshaped, Rh)
        rel_w = keras.ops.einsum("bNthwc,wkc->bNthwk", query_reshaped, Rw)
        rel_t = keras.ops.einsum("bNthwc,tkc->bNthwk", query_reshaped, Rt)
        
        # Reshape scores for broadcasting
        scores = self.score_reshape_pre_embedding_layer(scores)
        
        # Add relative positional bias
        scores += rel_h[:, :, :, :, :, None, :, None]
        scores += rel_w[:, :, :, :, :, None, None, :]
        scores += rel_t[:, :, :, :, :, :, None, None]
        
        # Reshape back to original format
        scores = self.score_reshape_after_embedding_layer(scores)
        
        return scores

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            "query_shape": self.config.query_shape,
            "key_shape": self.config.key_shape,
            "query_dim": self.config.query_dim,
            "heads": self.config.heads,
            "drop_rate": self.config.drop_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RelativePositionalEmbedding2D:
        """Create layer from configuration dictionary.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            RelativePositionalEmbedding2D layer instance.
        """
        # Extract embedding-specific config
        embedding_config = RelativePositionalEmbedding2DConfig(
            query_shape=config.pop("query_shape"),
            key_shape=config.pop("key_shape"),
            query_dim=config.pop("query_dim"),
            heads=config.pop("heads"),
            drop_rate=config.pop("drop_rate", 0.0),
        )
        
        # Pass remaining config to parent
        return cls(embedding_config, **config)


class RelativePositionalEmbedding2DKey(keras.layers.Layer):
    """Relative positional embedding for 2D spatiotemporal attention (key-based).
    
    This layer computes relative positional embeddings based on the key tensor
    for query-key attention in 2D space with temporal dimension.
    
    Args:
        config: Configuration object containing all layer parameters.
        **kwargs: Additional keyword arguments passed to parent Layer.
        
    Raises:
        ValueError: If configuration parameters are invalid.
        
    Example:
        >>> config = RelativePositionalEmbedding2DConfig(
        ...     query_shape=(4, 16, 16, 64),
        ...     key_shape=(4, 16, 16, 64),
        ...     query_dim=64,
        ...     heads=8
        ... )
        >>> layer = RelativePositionalEmbedding2DKey(config)
        >>> key = keras.random.uniform((1, 8, 1024, 64))
        >>> scores = keras.random.uniform((1, 8, 1024, 1024))
        >>> output = layer([key, scores])
    """
    
    def __init__(self, config: RelativePositionalEmbedding2DConfig, **kwargs: Any) -> None:
        """Initialize RelativePositionalEmbedding2DKey layer.
        
        Pre-conditions:
            - config must be a valid RelativePositionalEmbedding2DConfig instance
            
        Post-conditions:
            - Layer is ready for forward pass
            - Embedding weights are properly initialized
        """
        super().__init__(**kwargs)
        
        if not isinstance(config, RelativePositionalEmbedding2DConfig):
            raise TypeError(f"config must be RelativePositionalEmbedding2DConfig, got {type(config)}")
            
        self.config = config
        
        # Store config values
        self.query_seq_len, self.query_height, self.query_width, self.query_channels = config.query_shape
        self.key_seq_len, self.key_height, self.key_width, self.key_channels = config.key_shape
        self.query_dim = config.query_dim
        self.heads = config.heads
        self.drop_rate = config.drop_rate
        
        # Calculate embedding dimensions
        max_height_dist = max(self.query_height, self.key_height) - 1
        max_width_dist = max(self.query_width, self.key_width) - 1
        max_time_dist = max(self.query_seq_len, self.key_seq_len) - 1
        
        # Create embedding weights
        self.height_embeddings = self.add_weight(
            name='height_embeddings',
            shape=(2 * max_height_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        self.width_embeddings = self.add_weight(
            name='width_embeddings',
            shape=(2 * max_width_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        self.time_embeddings = self.add_weight(
            name='time_embeddings',
            shape=(2 * max_time_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        
        # Dropout layers
        self.height_dropout = keras.layers.Dropout(rate=self.drop_rate)
        self.width_dropout = keras.layers.Dropout(rate=self.drop_rate)
        self.time_dropout = keras.layers.Dropout(rate=self.drop_rate)
        
        # Reshape layers
        self.key_reshape_layer = keras.layers.Reshape(
            (self.heads, self.key_seq_len, self.key_height, self.key_width, self.query_dim)
        )
        
        self.score_reshape_pre_embedding_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len, self.query_height, self.query_width, 
             self.key_seq_len, self.key_height, self.key_width)
        )
        
        self.score_reshape_after_embedding_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len * self.query_height * self.query_width, 
             self.key_seq_len * self.key_height * self.key_width)
        )

    def _compute_relative_distances(self) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Compute relative distance matrices for height, width, and time (key-based).
        
        Returns:
            Tuple of (height_distances, width_distances, time_distances) tensors.
        """
        # Height distances (key to query)
        q_h_ratio = round(max(self.key_height / self.query_height, 1.0))
        k_h_ratio = round(max(self.query_height / self.key_height, 1.0))
        dist_h = (
            keras.ops.arange(self.key_height)[:, None] * k_h_ratio 
            - keras.ops.arange(self.query_height)[None, :] * q_h_ratio
        )
        dist_h += (self.query_height - 1) * q_h_ratio
        
        # Width distances (key to query)
        q_w_ratio = round(max(self.key_width / self.query_width, 1.0))
        k_w_ratio = round(max(self.query_width / self.key_width, 1.0))
        dist_w = (
            keras.ops.arange(self.key_width)[:, None] * k_w_ratio 
            - keras.ops.arange(self.query_width)[None, :] * q_w_ratio
        )
        dist_w += (self.query_width - 1) * q_w_ratio
        
        # Time distances (key to query)
        q_t_ratio = round(max(self.key_seq_len / self.query_seq_len, 1.0))
        k_t_ratio = round(max(self.query_seq_len / self.key_seq_len, 1.0))
        dist_t = (
            keras.ops.arange(self.key_seq_len)[:, None] * k_t_ratio 
            - keras.ops.arange(self.query_seq_len)[None, :] * q_t_ratio
        )
        dist_t += (self.query_seq_len - 1) * q_t_ratio
        
        return dist_h, dist_w, dist_t

    def build(self, input_shape: List[Tuple[int, ...]]) -> None:
        """Build layer by validating input shapes.
        
        Args:
            input_shape: List of shapes for [key, scores] inputs.
            
        Pre-conditions:
            - input_shape must contain exactly 2 shapes
            - Shapes must match expected dimensions
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2 inputs [key, scores], got {len(input_shape)}")
            
        key_shape, scores_shape = input_shape
        
        if len(key_shape) != 4:
            raise ValueError(f"Key must be 4D tensor, got {len(key_shape)}D")
        if len(scores_shape) != 4:
            raise ValueError(f"Scores must be 4D tensor, got {len(scores_shape)}D")
            
        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """Apply relative positional embeddings to attention scores (key-based).
        
        Args:
            inputs: List containing [key, scores] tensors.
            
        Returns:
            Updated attention scores with relative positional bias.
            
        Pre-conditions:
            - inputs must contain exactly 2 tensors [key, scores]
            - key must be 4D tensor (batch, heads, seq*h*w, query_dim)
            - scores must be 4D tensor (batch, heads, q_tokens, k_tokens)
            
        Post-conditions:
            - Output has same shape as input scores
            - Relative positional bias is added to scores
        """
        if len(inputs) != 2:
            raise ValueError(f"Expected 2 inputs [key, scores], got {len(inputs)}")
            
        key, scores = inputs
        
        if key.ndim != 4:
            raise ValueError(f"Key must be 4D tensor, got {key.ndim}D")
        if scores.ndim != 4:
            raise ValueError(f"Scores must be 4D tensor, got {scores.ndim}D")
            
        # Validate dimensions
        batch_size, heads, k_tokens, key_dim = key.shape
        _, _, q_tokens, _ = scores.shape
        
        expected_q_tokens = self.query_seq_len * self.query_height * self.query_width
        expected_k_tokens = self.key_seq_len * self.key_height * self.key_width
        
        if q_tokens != expected_q_tokens:
            raise ValueError(f"Query tokens mismatch: expected {expected_q_tokens}, got {q_tokens}")
        if k_tokens != expected_k_tokens:
            raise ValueError(f"Key tokens mismatch: expected {expected_k_tokens}, got {k_tokens}")
        if key_dim != self.query_dim:
            raise ValueError(f"Key dim mismatch: expected {self.query_dim}, got {key_dim}")
        if heads != self.heads:
            raise ValueError(f"Heads mismatch: expected {self.heads}, got {heads}")
            
        # Compute relative distances
        dist_h, dist_w, dist_t = self._compute_relative_distances()
        
        # Get embeddings
        Rh = keras.ops.take(self.height_embeddings, dist_h, axis=0)
        Rw = keras.ops.take(self.width_embeddings, dist_w, axis=0)
        Rt = keras.ops.take(self.time_embeddings, dist_t, axis=0)
        
        # Apply dropout
        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)
        
        # Reshape key for computation
        key_reshaped = self.key_reshape_layer(key)
        
        # Compute relative attention terms
        rel_h = keras.ops.einsum("hqc,bNshwc->bNqshw", Rh, key_reshaped)
        rel_w = keras.ops.einsum("wqc,bNshwc->bNqshw", Rw, key_reshaped)
        rel_t = keras.ops.einsum("sqc,bNshwc->bNqshw", Rt, key_reshaped)
        
        # Reshape scores for broadcasting
        scores = self.score_reshape_pre_embedding_layer(scores)
        
        # Add relative positional bias
        scores += rel_h[:, :, None, :, None, :, :, :]
        scores += rel_w[:, :, None, None, :, :, :, :]
        scores += rel_t[:, :, :, None, None, :, :, :]
        
        # Reshape back to original format
        scores = self.score_reshape_after_embedding_layer(scores)
        
        return scores

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            "query_shape": self.config.query_shape,
            "key_shape": self.config.key_shape,
            "query_dim": self.config.query_dim,
            "heads": self.config.heads,
            "drop_rate": self.config.drop_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RelativePositionalEmbedding2DKey:
        """Create layer from configuration dictionary.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            RelativePositionalEmbedding2DKey layer instance.
        """
        # Extract embedding-specific config
        embedding_config = RelativePositionalEmbedding2DConfig(
            query_shape=config.pop("query_shape"),
            key_shape=config.pop("key_shape"),
            query_dim=config.pop("query_dim"),
            heads=config.pop("heads"),
            drop_rate=config.pop("drop_rate", 0.0),
        )
        
        # Pass remaining config to parent
        return cls(embedding_config, **config)


# Convenience constructors for backward compatibility
def create_relative_positional_embedding_2d(
    query_shape: Tuple[int, int, int, int],
    key_shape: Tuple[int, int, int, int],
    query_dim: int,
    heads: int,
    drop_rate: float = 0.0
) -> RelativePositionalEmbedding2D:
    """Create RelativePositionalEmbedding2D layer with specified parameters.
    
    Args:
        query_shape: Shape of query tensor.
        key_shape: Shape of key tensor.
        query_dim: Query embedding dimension.
        heads: Number of attention heads.
        drop_rate: Dropout rate.
        
    Returns:
        Configured RelativePositionalEmbedding2D layer.
    """
    config = RelativePositionalEmbedding2DConfig(
        query_shape=query_shape,
        key_shape=key_shape,
        query_dim=query_dim,
        heads=heads,
        drop_rate=drop_rate
    )
    return RelativePositionalEmbedding2D(config)


def create_relative_positional_embedding_2d_key(
    query_shape: Tuple[int, int, int, int],
    key_shape: Tuple[int, int, int, int],
    query_dim: int,
    heads: int,
    drop_rate: float = 0.0
) -> RelativePositionalEmbedding2DKey:
    """Create RelativePositionalEmbedding2DKey layer with specified parameters.
    
    Args:
        query_shape: Shape of query tensor.
        key_shape: Shape of key tensor.
        query_dim: Query embedding dimension.
        heads: Number of attention heads.
        drop_rate: Dropout rate.
        
    Returns:
        Configured RelativePositionalEmbedding2DKey layer.
    """
    config = RelativePositionalEmbedding2DConfig(
        query_shape=query_shape,
        key_shape=key_shape,
        query_dim=query_dim,
        heads=heads,
        drop_rate=drop_rate
    )
    return RelativePositionalEmbedding2DKey(config)
