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
This pytorch implementation of relative positional encoding from MVit2 was
adapted from the original pytorch implementation:
https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py

Adapted by Paul Thompson 2024
Improved by following company Python guidelines 2024
"""

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn


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


class RelativePositionalEmbedding2D(nn.Module):
    """Relative positional embedding for 2D spatiotemporal attention.
    
    This layer computes relative positional embeddings for query-key attention
    in 2D space with temporal dimension. Based on MViT2 implementation.
    
    Args:
        config: Configuration object containing all layer parameters.
        
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
        >>> query = torch.randn(1, 8, 1024, 64)  # (batch, heads, seq*h*w, dim)
        >>> scores = torch.randn(1, 8, 1024, 1024)  # (batch, heads, q_tokens, k_tokens)
        >>> output = layer(query, scores)
    """
    
    def __init__(self, config: RelativePositionalEmbedding2DConfig) -> None:
        """Initialize RelativePositionalEmbedding2D layer.
        
        Pre-conditions:
            - config must be a valid RelativePositionalEmbedding2DConfig instance
            
        Post-conditions:
            - Layer is ready for forward pass
            - Embedding parameters are properly initialized
        """
        super().__init__()
        
        if not isinstance(config, RelativePositionalEmbedding2DConfig):
            raise TypeError(f"config must be RelativePositionalEmbedding2DConfig, got {type(config)}")
            
        self.config = config
        
        # Store config values for TorchScript compatibility
        self.query_seq_len, self.query_height, self.query_width, self.query_channels = config.query_shape
        self.key_seq_len, self.key_height, self.key_width, self.key_channels = config.key_shape
        self.query_dim = config.query_dim
        self.heads = config.heads
        self.drop_rate = config.drop_rate
        
        # Calculate embedding dimensions
        max_height_dist = max(self.query_height, self.key_height) - 1
        max_width_dist = max(self.query_width, self.key_width) - 1
        max_t_dist = max(self.query_seq_len, self.key_seq_len) - 1
        
        # Create learnable embeddings
        self.height_embeddings = nn.Parameter(
            torch.randn(2 * max_height_dist + 1, self.query_dim)
        )
        self.width_embeddings = nn.Parameter(
            torch.randn(2 * max_width_dist + 1, self.query_dim)
        )
        self.time_embeddings = nn.Parameter(
            torch.randn(2 * max_t_dist + 1, self.query_dim)
        )
        
        # Dropout layers
        self.height_dropout = nn.Dropout(p=self.drop_rate)
        self.width_dropout = nn.Dropout(p=self.drop_rate)
        self.time_dropout = nn.Dropout(p=self.drop_rate)

    def _compute_relative_distances(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute relative distance matrices for height, width, and time.
        
        Returns:
            Tuple of (height_distances, width_distances, time_distances) tensors.
        """
        # Height distances
        q_h_ratio = round(max(self.key_height / self.query_height, 1.0))
        k_h_ratio = round(max(self.query_height / self.key_height, 1.0))
        dist_h = (
            torch.arange(self.query_height).unsqueeze(1) * q_h_ratio 
            - torch.arange(self.key_height).unsqueeze(0) * k_h_ratio
        )
        dist_h += (self.key_height - 1) * k_h_ratio
        
        # Width distances
        q_w_ratio = round(max(self.key_width / self.query_width, 1.0))
        k_w_ratio = round(max(self.query_width / self.key_width, 1.0))
        dist_w = (
            torch.arange(self.query_width).unsqueeze(1) * q_w_ratio 
            - torch.arange(self.key_width).unsqueeze(0) * k_w_ratio
        )
        dist_w += (self.key_width - 1) * k_w_ratio
        
        # Time distances
        q_t_ratio = round(max(self.key_seq_len / self.query_seq_len, 1.0))
        k_t_ratio = round(max(self.query_seq_len / self.key_seq_len, 1.0))
        dist_t = (
            torch.arange(self.query_seq_len).unsqueeze(1) * q_t_ratio 
            - torch.arange(self.key_seq_len).unsqueeze(0) * k_t_ratio
        )
        dist_t += (self.key_seq_len - 1) * k_t_ratio
        
        return dist_h.int(), dist_w.int(), dist_t.int()

    def forward(self, query: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply relative positional embeddings to attention scores.
        
        Args:
            query: Query tensor of shape (batch, heads, seq*h*w, query_dim).
            scores: Attention scores of shape (batch, heads, q_tokens, k_tokens).
            
        Returns:
            Updated attention scores with relative positional bias.
            
        Pre-conditions:
            - query must be 4D tensor with correct dimensions
            - scores must be 4D tensor with correct dimensions
            - query and scores must be on same device
            
        Post-conditions:
            - Output has same shape as input scores
            - Relative positional bias is added to scores
        """
        if query.dim() != 4:
            raise ValueError(f"Query must be 4D tensor, got {query.dim()}D")
        if scores.dim() != 4:
            raise ValueError(f"Scores must be 4D tensor, got {scores.dim()}D")
            
        batch_size, heads, q_tokens, query_dim = query.shape
        _, _, _, k_tokens = scores.shape
        
        # Validate dimensions
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
        Rh = self.height_embeddings[dist_h]
        Rw = self.width_embeddings[dist_w]
        Rt = self.time_embeddings[dist_t]
        
        # Apply dropout
        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)
        
        # Reshape query for computation
        query_reshaped = query.view(
            batch_size, self.heads, self.query_seq_len, 
            self.query_height, self.query_width, self.query_dim
        )
        
        # Compute relative attention terms
        rel_h = torch.einsum("bNthwc,hkc->bNthwk", query_reshaped, Rh)
        rel_w = torch.einsum("bNthwc,wkc->bNthwk", query_reshaped, Rw)
        rel_t = torch.einsum("bNthwc,tkc->bNthwk", query_reshaped, Rt)
        
        # Reshape scores for broadcasting
        scores = scores.view(
            batch_size, self.heads, self.query_seq_len, self.query_height, self.query_width,
            self.key_seq_len, self.key_height, self.key_width
        )
        
        # Add relative positional bias
        scores += rel_h[:, :, :, :, :, None, :, None]
        scores += rel_w[:, :, :, :, :, None, None, :]
        scores += rel_t[:, :, :, :, :, :, None, None]
        
        # Reshape back to original format
        scores = scores.view(batch_size, self.heads, q_tokens, k_tokens)
        
        return scores


class RelativePositionalEmbedding2DKey(nn.Module):
    """Relative positional embedding for 2D spatiotemporal attention (key-based).
    
    This layer computes relative positional embeddings based on the key tensor
    for query-key attention in 2D space with temporal dimension.
    
    Args:
        config: Configuration object containing all layer parameters.
        
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
        >>> key = torch.randn(1, 8, 1024, 64)  # (batch, heads, seq*h*w, dim)
        >>> scores = torch.randn(1, 8, 1024, 1024)  # (batch, heads, q_tokens, k_tokens)
        >>> output = layer(key, scores)
    """
    
    def __init__(self, config: RelativePositionalEmbedding2DConfig) -> None:
        """Initialize RelativePositionalEmbedding2DKey layer.
        
        Pre-conditions:
            - config must be a valid RelativePositionalEmbedding2DConfig instance
            
        Post-conditions:
            - Layer is ready for forward pass
            - Embedding parameters are properly initialized
        """
        super().__init__()
        
        if not isinstance(config, RelativePositionalEmbedding2DConfig):
            raise TypeError(f"config must be RelativePositionalEmbedding2DConfig, got {type(config)}")
            
        self.config = config
        
        # Store config values for TorchScript compatibility
        self.query_seq_len, self.query_height, self.query_width, self.query_channels = config.query_shape
        self.key_seq_len, self.key_height, self.key_width, self.key_channels = config.key_shape
        self.query_dim = config.query_dim
        self.heads = config.heads
        self.drop_rate = config.drop_rate
        
        # Calculate embedding dimensions
        max_height_dist = max(self.query_height, self.key_height) - 1
        max_width_dist = max(self.query_width, self.key_width) - 1
        max_time_dist = max(self.query_seq_len, self.key_seq_len) - 1
        
        # Create learnable embeddings
        self.height_embeddings = nn.Parameter(
            torch.randn(2 * max_height_dist + 1, self.query_dim)
        )
        self.width_embeddings = nn.Parameter(
            torch.randn(2 * max_width_dist + 1, self.query_dim)
        )
        self.time_embeddings = nn.Parameter(
            torch.randn(2 * max_time_dist + 1, self.query_dim)
        )
        
        # Dropout layers
        self.height_dropout = nn.Dropout(p=self.drop_rate)
        self.width_dropout = nn.Dropout(p=self.drop_rate)
        self.time_dropout = nn.Dropout(p=self.drop_rate)

    def _compute_relative_distances(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute relative distance matrices for height, width, and time (key-based).
        
        Returns:
            Tuple of (height_distances, width_distances, time_distances) tensors.
        """
        # Height distances (key to query)
        q_h_ratio = round(max(self.key_height / self.query_height, 1.0))
        k_h_ratio = round(max(self.query_height / self.key_height, 1.0))
        dist_h = (
            torch.arange(self.key_height).unsqueeze(1) * k_h_ratio 
            - torch.arange(self.query_height).unsqueeze(0) * q_h_ratio
        )
        dist_h += (self.query_height - 1) * q_h_ratio
        
        # Width distances (key to query)
        q_w_ratio = round(max(self.key_width / self.query_width, 1.0))
        k_w_ratio = round(max(self.query_width / self.key_width, 1.0))
        dist_w = (
            torch.arange(self.key_width).unsqueeze(1) * k_w_ratio 
            - torch.arange(self.query_width).unsqueeze(0) * q_w_ratio
        )
        dist_w += (self.query_width - 1) * q_w_ratio
        
        # Time distances (key to query)
        q_t_ratio = round(max(self.key_seq_len / self.query_seq_len, 1.0))
        k_t_ratio = round(max(self.query_seq_len / self.key_seq_len, 1.0))
        dist_t = (
            torch.arange(self.key_seq_len).unsqueeze(1) * k_t_ratio 
            - torch.arange(self.query_seq_len).unsqueeze(0) * q_t_ratio
        )
        dist_t += (self.query_seq_len - 1) * q_t_ratio
        
        return dist_h.int(), dist_w.int(), dist_t.int()

    def forward(self, key: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply relative positional embeddings to attention scores (key-based).
        
        Args:
            key: Key tensor of shape (batch, heads, seq*h*w, query_dim).
            scores: Attention scores of shape (batch, heads, q_tokens, k_tokens).
            
        Returns:
            Updated attention scores with relative positional bias.
            
        Pre-conditions:
            - key must be 4D tensor with correct dimensions
            - scores must be 4D tensor with correct dimensions
            - key and scores must be on same device
            
        Post-conditions:
            - Output has same shape as input scores
            - Relative positional bias is added to scores
        """
        if key.dim() != 4:
            raise ValueError(f"Key must be 4D tensor, got {key.dim()}D")
        if scores.dim() != 4:
            raise ValueError(f"Scores must be 4D tensor, got {scores.dim()}D")
            
        batch_size, heads, k_tokens, key_dim = key.shape
        _, _, q_tokens, _ = scores.shape
        
        # Validate dimensions
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
        Rh = self.height_embeddings[dist_h]
        Rw = self.width_embeddings[dist_w]
        Rt = self.time_embeddings[dist_t]
        
        # Apply dropout
        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)
        
        # Reshape key for computation
        key_reshaped = key.view(
            batch_size, self.heads, self.key_seq_len, 
            self.key_height, self.key_width, self.query_dim
        )
        
        # Compute relative attention terms
        rel_h = torch.einsum("hqc,bNshwc->bNqshw", Rh, key_reshaped)
        rel_w = torch.einsum("wqc,bNshwc->bNqshw", Rw, key_reshaped)
        rel_t = torch.einsum("sqc,bNshwc->bNqshw", Rt, key_reshaped)
        
        # Reshape scores for broadcasting
        scores = scores.view(
            batch_size, self.heads, self.query_seq_len, self.query_height, self.query_width,
            self.key_seq_len, self.key_height, self.key_width
        )
        
        # Add relative positional bias
        scores += rel_h[:, :, None, :, None, :, :, :]
        scores += rel_w[:, :, None, None, :, :, :, :]
        scores += rel_t[:, :, :, None, None, :, :, :]
        
        # Reshape back to original format
        scores = scores.view(batch_size, self.heads, q_tokens, k_tokens)
        
        return scores


def load_positional_embedding_layer_weights(keras_layer: Any, pytorch_module: nn.Module) -> None:
    """Load weights from Keras layer to PyTorch module.
    
    Args:
        keras_layer: Source Keras layer with weights.
        pytorch_module: Target PyTorch module to load weights into.
        
    Raises:
        ValueError: If weight shapes don't match.
    """
    print(f"Loading custom layer weights for {keras_layer.name}")
    
    weights = keras_layer.get_weights()
    if len(weights) != 3:
        raise ValueError(f"Expected 3 weight arrays, got {len(weights)}")
        
    # Load weights with shape validation
    height_weights = torch.tensor(weights[0])
    if height_weights.shape != pytorch_module.height_embeddings.shape:
        raise ValueError(f"Height embeddings shape mismatch: expected {pytorch_module.height_embeddings.shape}, got {height_weights.shape}")
    pytorch_module.height_embeddings.data = height_weights
    
    width_weights = torch.tensor(weights[1])
    if width_weights.shape != pytorch_module.width_embeddings.shape:
        raise ValueError(f"Width embeddings shape mismatch: expected {pytorch_module.width_embeddings.shape}, got {width_weights.shape}")
    pytorch_module.width_embeddings.data = width_weights
    
    time_weights = torch.tensor(weights[2])
    if time_weights.shape != pytorch_module.time_embeddings.shape:
        raise ValueError(f"Time embeddings shape mismatch: expected {pytorch_module.time_embeddings.shape}, got {time_weights.shape}")
    pytorch_module.time_embeddings.data = time_weights


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


# Legacy alias for backward compatibility
load_positionaL_embedding_layer_weights = load_positional_embedding_layer_weights
