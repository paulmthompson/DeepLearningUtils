"""
Keras implementation of RelativePositionalEmbedding2D layers.

This implements relative positional encoding from MVit2 for 2D spatial and temporal attention.

Reference:
Copyright 2019, Facebook, Inc
Licensed under the Apache License, Version 2.0

Original implementation:
https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py

Adapted by Paul Thompson 2024
"""

from typing import Tuple, Union, Any, List
import keras
import numpy as np

from .positional_embedding_config import RelativePositionalEmbeddingConfig
from .distance_calculations import calculate_all_distances, calculate_key_distances, validate_embedding_shapes


class RelativePositionalEmbedding2D(keras.layers.Layer):
    """
    Keras implementation of 2D relative positional embedding.
    
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
        drop_rate: float = 0.0,
        **kwargs: Any
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
            **kwargs: Additional arguments passed to parent Layer class.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__(**kwargs)
        
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
        
        # Validate and get embedding shapes
        height_shape, width_shape, time_shape = validate_embedding_shapes(self.config)
        
        # Create embedding weights
        self.height_embeddings = self.add_weight(
            name='height_embeddings',
            shape=height_shape,
            initializer='uniform',
            trainable=True,
        )
        self.width_embeddings = self.add_weight(
            name='width_embeddings',
            shape=width_shape,
            initializer='uniform',
            trainable=True,
        )
        self.time_embeddings = self.add_weight(
            name='time_embeddings',
            shape=time_shape,
            initializer='uniform',
            trainable=True,
        )

        # Create dropout layers
        self.height_dropout = keras.layers.Dropout(rate=self.config.drop_rate)
        self.width_dropout = keras.layers.Dropout(rate=self.config.drop_rate)
        self.time_dropout = keras.layers.Dropout(rate=self.config.drop_rate)
        
        # Create reshape layers
        self.query_reshape_layer = keras.layers.Reshape(
            (self.config.heads, self.config.query_seq_len, 
             self.config.query_height, self.config.query_width, self.config.query_dim)
        )

        self.score_reshape_pre_embedding_layer = keras.layers.Reshape(
            (self.config.heads, self.config.query_seq_len, self.config.query_height, self.config.query_width,
             self.config.key_seq_len, self.config.key_height, self.config.key_width)
        )

        self.score_reshape_after_embedding_layer = keras.layers.Reshape(
            (self.config.heads, 
             self.config.query_seq_len * self.config.query_height * self.config.query_width,
             self.config.key_seq_len * self.config.key_height * self.config.key_width)
        )
        
        # Precompute distance matrices for efficiency
        height_dist, width_dist, time_dist = calculate_all_distances(self.config)
        self.height_distances = keras.ops.convert_to_tensor(height_dist, dtype="int32")
        self.width_distances = keras.ops.convert_to_tensor(width_dist, dtype="int32")
        self.time_distances = keras.ops.convert_to_tensor(time_dist, dtype="int32")
    
    def build(self, input_shape: List[Tuple[int, ...]]) -> None:
        """
        Build layer by validating input shapes.
        
        Args:
            input_shape: List of shapes for [query, scores] tensors.
            
        Pre-conditions:
            - input_shape must be a list of 2 tuples
            - Each shape must represent 4D tensor
            
        Post-conditions:
            - Layer is built and ready for forward pass
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                f"Expected input_shape to be a list of 2 shapes [query_shape, scores_shape], "
                f"got {input_shape}"
            )

        query_shape, scores_shape = input_shape

        if len(query_shape) != 4:
            raise ValueError(
                f"Expected 4D query shape (batch, heads, seq_len*height*width, query_dim), "
                f"got {len(query_shape)}D shape: {query_shape}"
            )
        
        if len(scores_shape) != 4:
            raise ValueError(
                f"Expected 4D scores shape (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w), "
                f"got {len(scores_shape)}D shape: {scores_shape}"
            )
        
        super().build(input_shape)
    
    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Apply relative positional embedding to attention scores.
        
        Args:
            inputs: List of [query, scores] tensors.
                query: Query tensor with shape (batch, heads, seq_len*height*width, query_dim).
                scores: Attention scores with shape (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w).
            
        Returns:
            Modified attention scores with relative positional information added.
            
        Pre-conditions:
            - inputs must be a list of 2 tensors
            - Both tensors must be 4D
            - Layer must be built
            
        Post-conditions:
            - Output has same shape as input scores
            - Relative positional information is added to attention scores
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(f"Expected inputs to be a list of 2 tensors, got {inputs}")

        query, scores = inputs

        # Get embedding vectors using precomputed distances
        Rh = keras.ops.take(self.height_embeddings, self.height_distances, axis=0)
        Rw = keras.ops.take(self.width_embeddings, self.width_distances, axis=0)
        Rt = keras.ops.take(self.time_embeddings, self.time_distances, axis=0)
        
        # Apply dropout
        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)

        # Reshape query for einsum operations
        query_reshaped = self.query_reshape_layer(query)

        # Compute relative position contributions
        rel_h = keras.ops.einsum("bNthwc,hkc->bNthwk", query_reshaped, Rh)
        rel_w = keras.ops.einsum("bNthwc,wkc->bNthwk", query_reshaped, Rw)
        rel_t = keras.ops.einsum("bNthwc,tkc->bNthwk", query_reshaped, Rt)
        
        # Reshape scores for position addition
        scores = self.score_reshape_pre_embedding_layer(scores)

        # Add relative position contributions
        scores += rel_h[:, :, :, :, :, None, :, None]
        scores += rel_w[:, :, :, :, :, None, None, :]
        scores += rel_t[:, :, :, :, :, :, None, None]

        # Reshape back to original shape
        scores = self.score_reshape_after_embedding_layer(scores)

        return scores
    
    def get_config(self) -> dict[str, Any]:
        """
        Return layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            'query_shape': self.config.query_shape,
            'key_shape': self.config.key_shape,
            'query_dim': self.config.query_dim,
            'heads': self.config.heads,
            'drop_rate': self.config.drop_rate,
        })
        return config


class RelativePositionalEmbedding2DKey(keras.layers.Layer):
    """
    Keras implementation of 2D relative positional embedding for keys.
    
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
        drop_rate: float = 0.0,
        **kwargs: Any
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
            **kwargs: Additional arguments passed to parent Layer class.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__(**kwargs)
        
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
        
        # Validate and get embedding shapes
        height_shape, width_shape, time_shape = validate_embedding_shapes(self.config)
        
        # Create embedding weights
        self.height_embeddings = self.add_weight(
            name='height_embeddings',
            shape=height_shape,
            initializer='uniform',
            trainable=True,
        )
        self.width_embeddings = self.add_weight(
            name='width_embeddings',
            shape=width_shape,
            initializer='uniform',
            trainable=True,
        )
        self.time_embeddings = self.add_weight(
            name='time_embeddings',
            shape=time_shape,
            initializer='uniform',
            trainable=True,
        )

        # Create dropout layers
        self.height_dropout = keras.layers.Dropout(rate=self.config.drop_rate)
        self.width_dropout = keras.layers.Dropout(rate=self.config.drop_rate)
        self.time_dropout = keras.layers.Dropout(rate=self.config.drop_rate)
        
        # Create reshape layers
        self.key_reshape_layer = keras.layers.Reshape(
            (self.config.heads, self.config.key_seq_len,
             self.config.key_height, self.config.key_width, self.config.query_dim)
        )

        self.score_reshape_pre_embedding_layer = keras.layers.Reshape(
            (self.config.heads, self.config.query_seq_len, self.config.query_height, self.config.query_width,
             self.config.key_seq_len, self.config.key_height, self.config.key_width)
        )

        self.score_reshape_after_embedding_layer = keras.layers.Reshape(
            (self.config.heads,
             self.config.query_seq_len * self.config.query_height * self.config.query_width,
             self.config.key_seq_len * self.config.key_height * self.config.key_width)
        )
        
        # Precompute distance matrices for key-based calculation
        height_dist, width_dist, time_dist = calculate_key_distances(self.config)
        self.height_distances = keras.ops.convert_to_tensor(height_dist, dtype="int32")
        self.width_distances = keras.ops.convert_to_tensor(width_dist, dtype="int32")
        self.time_distances = keras.ops.convert_to_tensor(time_dist, dtype="int32")
    
    def build(self, input_shape: List[Tuple[int, ...]]) -> None:
        """
        Build layer by validating input shapes.
        
        Args:
            input_shape: List of shapes for [key, scores] tensors.
            
        Pre-conditions:
            - input_shape must be a list of 2 tuples
            - Each shape must represent 4D tensor
            
        Post-conditions:
            - Layer is built and ready for forward pass
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                f"Expected input_shape to be a list of 2 shapes [key_shape, scores_shape], "
                f"got {input_shape}"
            )

        key_shape, scores_shape = input_shape

        if len(key_shape) != 4:
            raise ValueError(
                f"Expected 4D key shape (batch, heads, seq_len*height*width, query_dim), "
                f"got {len(key_shape)}D shape: {key_shape}"
            )
        
        if len(scores_shape) != 4:
            raise ValueError(
                f"Expected 4D scores shape (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w), "
                f"got {len(scores_shape)}D shape: {scores_shape}"
            )
        
        super().build(input_shape)
    
    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Apply key-based relative positional embedding to attention scores.
        
        Args:
            inputs: List of [key, scores] tensors.
                key: Key tensor with shape (batch, heads, seq_len*height*width, query_dim).
                scores: Attention scores with shape (batch, heads, q_seq*q_h*q_w, k_seq*k_h*k_w).
            
        Returns:
            Modified attention scores with key-based relative positional information added.
            
        Pre-conditions:
            - inputs must be a list of 2 tensors
            - Both tensors must be 4D
            - Layer must be built
            
        Post-conditions:
            - Output has same shape as input scores
            - Key-based relative positional information is added to attention scores
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(f"Expected inputs to be a list of 2 tensors, got {inputs}")
        
        key, scores = inputs
        
        # Get embedding vectors using precomputed distances
        Rh = keras.ops.take(self.height_embeddings, self.height_distances, axis=0)
        Rw = keras.ops.take(self.width_embeddings, self.width_distances, axis=0)
        Rt = keras.ops.take(self.time_embeddings, self.time_distances, axis=0)
        
        # Apply dropout
        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)

        # Reshape key for einsum operations
        key_reshaped = self.key_reshape_layer(key)

        # Compute relative position contributions from key perspective
        rel_h = keras.ops.einsum("hqc,bNshwc->bNqshw", Rh, key_reshaped)
        rel_w = keras.ops.einsum("wqc,bNshwc->bNqshw", Rw, key_reshaped)
        rel_t = keras.ops.einsum("sqc,bNshwc->bNqshw", Rt, key_reshaped)
        
        # Reshape scores for position addition
        scores = self.score_reshape_pre_embedding_layer(scores)

        # Add relative position contributions
        scores += rel_h[:, :, None, :, None, :, :, :]
        scores += rel_w[:, :, None, None, :, :, :, :]
        scores += rel_t[:, :, :, None, None, :, :, :]

        # Reshape back to original shape
        scores = self.score_reshape_after_embedding_layer(scores)

        return scores
    
    def get_config(self) -> dict[str, Any]:
        """
        Return layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            'query_shape': self.config.query_shape,
            'key_shape': self.config.key_shape,
            'query_dim': self.config.query_dim,
            'heads': self.config.heads,
            'drop_rate': self.config.drop_rate,
        })
        return config
