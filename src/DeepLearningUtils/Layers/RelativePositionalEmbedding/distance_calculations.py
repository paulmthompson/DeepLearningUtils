"""
Distance calculation utilities for RelativePositionalEmbedding layers.

This module provides shared functions for calculating spatial and temporal distances
used in relative positional embeddings. These functions return numpy arrays that
can be converted to framework-specific tensors.
"""

import numpy as np
from typing import Tuple

from .positional_embedding_config import RelativePositionalEmbeddingConfig


def calculate_spatial_distances(
    query_size: int, 
    key_size: int
) -> np.ndarray:
    """
    Calculate spatial distances between query and key positions.
    
    This function computes the relative distances between all pairs of spatial
    positions in query and key tensors, accounting for different resolutions.
    
    Args:
        query_size: Spatial size of query tensor (height or width).
        key_size: Spatial size of key tensor (height or width).
        
    Returns:
        Distance matrix with shape (query_size, key_size).
        
    Pre-conditions:
        - query_size > 0
        - key_size > 0
        
    Post-conditions:
        - Output shape is (query_size, key_size)
        - All distances are non-negative integers
        - Maximum distance is properly offset for indexing
    """
    if query_size <= 0:
        raise ValueError(f"query_size must be positive, got {query_size}")
    if key_size <= 0:
        raise ValueError(f"key_size must be positive, got {key_size}")
    
    # Calculate scaling ratios for different resolutions
    q_ratio = round(max(key_size / query_size, 1.0))
    k_ratio = round(max(query_size / key_size, 1.0))
    
    # Compute relative distances
    query_positions = np.arange(query_size)[:, None] * q_ratio
    key_positions = np.arange(key_size)[None, :] * k_ratio
    
    distances = query_positions - key_positions
    
    # Offset by maximum distance to ensure non-negative indices
    distances += (key_size - 1) * k_ratio
    
    return distances.astype(np.int32)


def calculate_temporal_distances(
    query_seq_len: int,
    key_seq_len: int
) -> np.ndarray:
    """
    Calculate temporal distances between query and key time steps.
    
    Args:
        query_seq_len: Sequence length of query tensor.
        key_seq_len: Sequence length of key tensor.
        
    Returns:
        Distance matrix with shape (query_seq_len, key_seq_len).
        
    Pre-conditions:
        - query_seq_len > 0
        - key_seq_len > 0
        
    Post-conditions:
        - Output shape is (query_seq_len, key_seq_len)
        - All distances are non-negative integers
        - Maximum distance is properly offset for indexing
    """
    if query_seq_len <= 0:
        raise ValueError(f"query_seq_len must be positive, got {query_seq_len}")
    if key_seq_len <= 0:
        raise ValueError(f"key_seq_len must be positive, got {key_seq_len}")
    
    # Calculate scaling ratios for different sequence lengths
    q_ratio = round(max(key_seq_len / query_seq_len, 1.0))
    k_ratio = round(max(query_seq_len / key_seq_len, 1.0))
    
    # Compute relative distances
    query_positions = np.arange(query_seq_len)[:, None] * q_ratio
    key_positions = np.arange(key_seq_len)[None, :] * k_ratio
    
    distances = query_positions - key_positions
    
    # Offset by maximum distance to ensure non-negative indices
    distances += (key_seq_len - 1) * k_ratio
    
    return distances.astype(np.int32)


def calculate_all_distances(
    config: RelativePositionalEmbeddingConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate all distance matrices for a given configuration.
    
    Args:
        config: Configuration containing query and key shapes.
        
    Returns:
        Tuple of (height_distances, width_distances, time_distances).
        
    Pre-conditions:
        - config is valid (validated in RelativePositionalEmbeddingConfig.__post_init__)
        
    Post-conditions:
        - All distance matrices have proper shapes and non-negative values
        - Distance values are suitable for use as indices into embedding matrices
    """
    height_distances = calculate_spatial_distances(
        config.query_height, 
        config.key_height
    )
    
    width_distances = calculate_spatial_distances(
        config.query_width, 
        config.key_width
    )
    
    time_distances = calculate_temporal_distances(
        config.query_seq_len, 
        config.key_seq_len
    )
    
    return height_distances, width_distances, time_distances


def calculate_key_distances(
    config: RelativePositionalEmbeddingConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate distance matrices for key-based relative positioning.
    
    This variant calculates distances from the perspective of keys to queries,
    which is used in RelativePositionalEmbedding2DKey.
    
    Args:
        config: Configuration containing query and key shapes.
        
    Returns:
        Tuple of (height_distances, width_distances, time_distances) from key perspective.
        
    Pre-conditions:
        - config is valid (validated in RelativePositionalEmbeddingConfig.__post_init__)
        
    Post-conditions:
        - All distance matrices have proper shapes and non-negative values
        - Distance values are suitable for use as indices into embedding matrices
    """
    # For key-based distances, we swap the query/key roles
    height_distances = calculate_spatial_distances(
        config.key_height, 
        config.query_height
    )
    
    width_distances = calculate_spatial_distances(
        config.key_width, 
        config.query_width
    )
    
    time_distances = calculate_temporal_distances(
        config.key_seq_len, 
        config.query_seq_len
    )
    
    return height_distances, width_distances, time_distances


def validate_embedding_shapes(
    config: RelativePositionalEmbeddingConfig
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Calculate and validate embedding matrix shapes.
    
    Args:
        config: Configuration containing dimensions.
        
    Returns:
        Tuple of ((height_embed_rows, query_dim), (width_embed_rows, query_dim), (time_embed_rows, query_dim)).
        
    Pre-conditions:
        - config is valid
        
    Post-conditions:
        - All shapes are positive
        - Shapes are consistent with distance calculation requirements
    """
    height_shape = (2 * config.max_height_dist + 1, config.query_dim)
    width_shape = (2 * config.max_width_dist + 1, config.query_dim)
    time_shape = (2 * config.max_time_dist + 1, config.query_dim)
    
    # Validate that shapes are reasonable
    for name, shape in [("height", height_shape), ("width", width_shape), ("time", time_shape)]:
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Invalid {name} embedding shape: {shape}")
    
    return height_shape, width_shape, time_shape 