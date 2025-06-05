"""
PyTorch implementation of RelativePositionalEmbedding2D layers.

This implements relative positional encoding from MVit2 for 2D spatial and temporal attention.

Reference:
Copyright 2019, Facebook, Inc
Licensed under the Apache License, Version 2.0

Original implementation:
https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py

Adapted by Paul Thompson 2024
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

from .positional_embedding_config import RelativePositionalEmbeddingConfig
from .distance_calculations import calculate_all_distances, calculate_key_distances, validate_embedding_shapes


class RelativePositionalEmbedding2D(nn.Module):
    """
    PyTorch implementation of 2D relative positional embedding.
    
    This layer computes relative positional embeddings for spatial and temporal dimensions
    in attention mechanisms. It adds relative position information to attention scores
    based on the spatial and temporal distances between query and key positions.
    
    The layer supports different resolutions for query and key tensors and handles
    both spatial (height/width) and temporal (sequence) dimensions.
    
    Examples:
        >>> # Basic usage
        >>> config = RelativePositionalEmbeddingConfig(
        ...     query_shape=(1, 32, 32, 64),
        ...     key_shape=(1, 32, 32, 64),
        ...     query_dim=64,
        ...     heads=8
        ... )
        >>> layer = RelativePositionalEmbedding2D(config)
        >>> 
        >>> # Direct parameter specification
        >>> layer = RelativePositionalEmbedding2D(
        ...     query_shape=(1, 16, 16, 128),
        ...     key_shape=(1, 16, 16, 128),
        ...     query_dim=128,
        ...     heads=4,
        ...     drop_rate=0.1
        ... )
    
    Attributes:
        config: Configuration dataclass containing all layer parameters.
        
    Pre-conditions:
        - Query tensor must have shape (batch, heads, seq_len*height*width, query_dim)
        - Scores tensor must have shape (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w)
        
    Post-conditions:
        - Output has same shape as input scores tensor
        - Relative positional information is added to attention scores
    """
    
    def __init__(
        self,
        config: Union[RelativePositionalEmbeddingConfig, None] = None,
        *,
        query_shape: Union[Tuple[int, int, int, int], None] = None,
        key_shape: Union[Tuple[int, int, int, int], None] = None,
        query_dim: Union[int, None] = None,
        heads: Union[int, None] = None,
        drop_rate: float = 0.0
    ) -> None:
        """
        Initialize RelativePositionalEmbedding2D layer.
        
        Args:
            config: Configuration dataclass. If provided, other parameters are ignored.
            query_shape: Shape of query tensor (seq_len, height, width, channels). Used only if config is None.
            key_shape: Shape of key tensor (seq_len, height, width, channels). Used only if config is None.
            query_dim: Dimension of query embeddings. Used only if config is None.
            heads: Number of attention heads. Used only if config is None.
            drop_rate: Dropout rate for embeddings. Used only if config is None.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__()
        
        # Create configuration from parameters or use provided config
        if config is None:
            if any(param is None for param in [query_shape, key_shape, query_dim, heads]):
                raise ValueError(
                    "Either config must be provided, or all of query_shape, key_shape, "
                    "query_dim, and heads must be specified"
                )
            
            self.config = RelativePositionalEmbeddingConfig(
                query_shape=query_shape,  # type: ignore
                key_shape=key_shape,      # type: ignore
                query_dim=query_dim,      # type: ignore
                heads=heads,              # type: ignore
                drop_rate=drop_rate
            )
        else:
            self.config = config
        
        # Store individual attributes for JIT compatibility
        self.query_seq_len = self.config.query_seq_len
        self.query_height = self.config.query_height
        self.query_width = self.config.query_width
        self.key_seq_len = self.config.key_seq_len
        self.key_height = self.config.key_height
        self.key_width = self.config.key_width
        self.query_dim = self.config.query_dim
        self.heads = self.config.heads
        self.drop_rate = self.config.drop_rate
        
        # Validate and get embedding shapes
        height_shape, width_shape, time_shape = validate_embedding_shapes(self.config)
        
        # Create embedding parameters
        self.height_embeddings = nn.Parameter(torch.randn(height_shape))
        self.width_embeddings = nn.Parameter(torch.randn(width_shape))
        self.time_embeddings = nn.Parameter(torch.randn(time_shape))
        
        # Create dropout layers
        self.height_dropout = nn.Dropout(p=self.drop_rate)
        self.width_dropout = nn.Dropout(p=self.drop_rate)
        self.time_dropout = nn.Dropout(p=self.drop_rate)

        # Precompute distance matrices for efficiency
        height_dist, width_dist, time_dist = calculate_all_distances(self.config)
        self.register_buffer('height_distances', torch.from_numpy(height_dist))
        self.register_buffer('width_distances', torch.from_numpy(width_dist))
        self.register_buffer('time_distances', torch.from_numpy(time_dist))
    
    def forward(self, query: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply relative positional embedding to attention scores.
        
        Args:
            query: Query tensor with shape (batch, heads, seq_len*height*width, query_dim).
            scores: Attention scores with shape (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w).
            
        Returns:
            Modified attention scores with relative positional information added.
            
        Pre-conditions:
            - query.dim() == 4
            - scores.dim() == 4
            - query and scores have compatible shapes
            
        Post-conditions:
            - Output has same shape as input scores
            - Relative positional information is added to attention scores
            
        Raises:
            RuntimeError: If input tensors have wrong shapes.
        """
        if query.dim() != 4:
            raise RuntimeError(
                f"Expected 4D query tensor (batch, heads, seq_len*height*width, query_dim), "
                f"got {query.dim()}D tensor with shape {query.shape}"
            )
        
        if scores.dim() != 4:
            raise RuntimeError(
                f"Expected 4D scores tensor (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w), "
                f"got {scores.dim()}D tensor with shape {scores.shape}"
            )
        
        # Get embedding vectors using precomputed distances
        Rh = self.height_embeddings[self.height_distances]
        Rw = self.width_embeddings[self.width_distances]
        Rt = self.time_embeddings[self.time_distances]
        
        # Apply dropout
        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)

        # Reshape query for einsum operations
        query_reshaped = query.view(
            -1, self.heads, self.query_seq_len, self.query_height, self.query_width, self.query_dim
        )

        # Compute relative position contributions
        rel_h = torch.einsum("bNthwc,hkc->bNthwk", query_reshaped, Rh)
        rel_w = torch.einsum("bNthwc,wkc->bNthwk", query_reshaped, Rw)
        rel_t = torch.einsum("bNthwc,tkc->bNthwk", query_reshaped, Rt)

        # Reshape scores for position addition
        scores = scores.view(
            -1, self.heads, self.query_seq_len, self.query_height, self.query_width,
            self.key_seq_len, self.key_height, self.key_width
        )
        
        # Add relative position contributions
        scores += rel_h[:, :, :, :, :, None, :, None]
        scores += rel_w[:, :, :, :, :, None, None, :]
        scores += rel_t[:, :, :, :, :, :, None, None]

        # Reshape back to original shape
        scores = scores.view(
            -1, self.heads,
            self.query_seq_len * self.query_height * self.query_width,
            self.key_seq_len * self.key_height * self.key_width
        )

        return scores
    
    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"query_shape={self.config.query_shape}, "
            f"key_shape={self.config.key_shape}, "
            f"query_dim={self.config.query_dim}, "
            f"heads={self.config.heads}, "
            f"drop_rate={self.config.drop_rate}"
        )


class RelativePositionalEmbedding2DKey(nn.Module):
    """
    PyTorch implementation of 2D relative positional embedding for keys.
    
    This variant computes relative positional embeddings from the perspective of keys
    to queries, which provides a different bias pattern in attention mechanisms.
    
    Examples:
        >>> # Basic usage
        >>> config = RelativePositionalEmbeddingConfig(
        ...     query_shape=(1, 32, 32, 64),
        ...     key_shape=(1, 32, 32, 64),
        ...     query_dim=64,
        ...     heads=8
        ... )
        >>> layer = RelativePositionalEmbedding2DKey(config)
    
    Attributes:
        config: Configuration dataclass containing all layer parameters.
        
    Pre-conditions:
        - Key tensor must have shape (batch, heads, seq_len*height*width, query_dim)
        - Scores tensor must have shape (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w)
        
    Post-conditions:
        - Output has same shape as input scores tensor
        - Relative positional information is added to attention scores
    """
    
    def __init__(
        self,
        config: Union[RelativePositionalEmbeddingConfig, None] = None,
        *,
        query_shape: Union[Tuple[int, int, int, int], None] = None,
        key_shape: Union[Tuple[int, int, int, int], None] = None,
        query_dim: Union[int, None] = None,
        heads: Union[int, None] = None,
        drop_rate: float = 0.0
    ) -> None:
        """
        Initialize RelativePositionalEmbedding2DKey layer.
        
        Args:
            config: Configuration dataclass. If provided, other parameters are ignored.
            query_shape: Shape of query tensor. Used only if config is None.
            key_shape: Shape of key tensor. Used only if config is None.
            query_dim: Dimension of query embeddings. Used only if config is None.
            heads: Number of attention heads. Used only if config is None.
            drop_rate: Dropout rate for embeddings. Used only if config is None.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__()
        
        # Create configuration from parameters or use provided config
        if config is None:
            if any(param is None for param in [query_shape, key_shape, query_dim, heads]):
                raise ValueError(
                    "Either config must be provided, or all of query_shape, key_shape, "
                    "query_dim, and heads must be specified"
                )
            
            self.config = RelativePositionalEmbeddingConfig(
                query_shape=query_shape,  # type: ignore
                key_shape=key_shape,      # type: ignore
                query_dim=query_dim,      # type: ignore
                heads=heads,              # type: ignore
                drop_rate=drop_rate
            )
        else:
            self.config = config
        
        # Store individual attributes for JIT compatibility
        self.query_seq_len = self.config.query_seq_len
        self.query_height = self.config.query_height
        self.query_width = self.config.query_width
        self.key_seq_len = self.config.key_seq_len
        self.key_height = self.config.key_height
        self.key_width = self.config.key_width
        self.query_dim = self.config.query_dim
        self.heads = self.config.heads
        self.drop_rate = self.config.drop_rate
        
        # Validate and get embedding shapes
        height_shape, width_shape, time_shape = validate_embedding_shapes(self.config)
        
        # Create embedding parameters
        self.height_embeddings = nn.Parameter(torch.randn(height_shape))
        self.width_embeddings = nn.Parameter(torch.randn(width_shape))
        self.time_embeddings = nn.Parameter(torch.randn(time_shape))
        
        # Create dropout layers
        self.height_dropout = nn.Dropout(p=self.drop_rate)
        self.width_dropout = nn.Dropout(p=self.drop_rate)
        self.time_dropout = nn.Dropout(p=self.drop_rate)

        # Precompute distance matrices for key-based calculation
        height_dist, width_dist, time_dist = calculate_key_distances(self.config)
        self.register_buffer('height_distances', torch.from_numpy(height_dist))
        self.register_buffer('width_distances', torch.from_numpy(width_dist))
        self.register_buffer('time_distances', torch.from_numpy(time_dist))
    
    def forward(self, key: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply key-based relative positional embedding to attention scores.
        
        Args:
            key: Key tensor with shape (batch, heads, seq_len*height*width, query_dim).
            scores: Attention scores with shape (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w).
            
        Returns:
            Modified attention scores with key-based relative positional information added.
            
        Pre-conditions:
            - key.dim() == 4
            - scores.dim() == 4
            - key and scores have compatible shapes
            
        Post-conditions:
            - Output has same shape as input scores
            - Key-based relative positional information is added to attention scores
            
        Raises:
            RuntimeError: If input tensors have wrong shapes.
        """
        if key.dim() != 4:
            raise RuntimeError(
                f"Expected 4D key tensor (batch, heads, seq_len*height*width, query_dim), "
                f"got {key.dim()}D tensor with shape {key.shape}"
            )
        
        if scores.dim() != 4:
            raise RuntimeError(
                f"Expected 4D scores tensor (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w), "
                f"got {scores.dim()}D tensor with shape {scores.shape}"
            )
        
        # Get embedding vectors using precomputed distances
        Rh = self.height_embeddings[self.height_distances]
        Rw = self.width_embeddings[self.width_distances]
        Rt = self.time_embeddings[self.time_distances]
        
        # Apply dropout
        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)

        # Reshape key for einsum operations
        key_reshaped = key.view(
            -1, self.heads, self.key_seq_len, self.key_height, self.key_width, self.query_dim
        )

        # Compute relative position contributions from key perspective
        rel_h = torch.einsum("hqc,bNshwc->bNqshw", Rh, key_reshaped)
        rel_w = torch.einsum("wqc,bNshwc->bNqshw", Rw, key_reshaped)
        rel_t = torch.einsum("sqc,bNshwc->bNqshw", Rt, key_reshaped)

        # Reshape scores for position addition
        scores = scores.view(
            -1, self.heads, self.query_seq_len, self.query_height, self.query_width,
            self.key_seq_len, self.key_height, self.key_width
        )
        
        # Add relative position contributions
        scores += rel_h[:, :, None, :, None, :, :, :]
        scores += rel_w[:, :, None, None, :, :, :, :]
        scores += rel_t[:, :, :, None, None, :, :, :]

        # Reshape back to original shape
        scores = scores.view(
            -1, self.heads,
            self.query_seq_len * self.query_height * self.query_width,
            self.key_seq_len * self.key_height * self.key_width
        )

        return scores

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"query_shape={self.config.query_shape}, "
            f"key_shape={self.config.key_shape}, "
            f"query_dim={self.config.query_dim}, "
            f"heads={self.config.heads}, "
            f"drop_rate={self.config.drop_rate}"
        )


def load_positionaL_embedding_layer_weights(
    keras_layer, 
    pytorch_module
) -> None:
    """
    Load weights from Keras layer to PyTorch module.
    
    Args:
        keras_layer: Source Keras layer with trained weights.
        pytorch_module: Target PyTorch module to load weights into.
        
    Pre-conditions:
        - keras_layer has valid weights
        - pytorch_module has compatible parameter structure
        
    Post-conditions:
        - PyTorch module parameters are updated with Keras weights
    """
    print(f"Loading custom layer weights for {keras_layer.name}")

    weights = keras_layer.get_weights()
    if len(weights) >= 3:
        pytorch_module.height_embeddings.data = torch.tensor(weights[0])
        pytorch_module.width_embeddings.data = torch.tensor(weights[1])
        pytorch_module.time_embeddings.data = torch.tensor(weights[2])
    else:
        raise ValueError(f"Expected at least 3 weight matrices, got {len(weights)}")
