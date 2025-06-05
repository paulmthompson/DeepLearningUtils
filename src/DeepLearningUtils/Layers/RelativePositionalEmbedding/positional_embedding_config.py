"""
Configuration dataclass for RelativePositionalEmbedding layers.

This module provides a shared configuration structure for both Keras and PyTorch
implementations of the RelativePositionalEmbedding2D layers.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RelativePositionalEmbeddingConfig:
    """
    Configuration for RelativePositionalEmbedding2D layers.
    
    This dataclass encapsulates all parameters needed to configure a RelativePositionalEmbedding
    layer, providing validation and documentation of expected inputs.
    
    Attributes:
        query_shape: Shape of query tensor (seq_len, height, width, channels).
        key_shape: Shape of key/memory tensor (seq_len, height, width, channels).
        query_dim: Dimension of the query embeddings.
        heads: Number of attention heads.
        drop_rate: Dropout rate applied to embeddings.
        
    Pre-conditions:
        - All shape dimensions must be positive
        - query_dim must be positive
        - heads must be positive
        - drop_rate must be between 0.0 and 1.0 (inclusive)
        - Shape tuples must have exactly 4 elements
        
    Post-conditions:
        - Configuration is immutable (frozen=True)
        - All values are validated in __post_init__
    """
    
    query_shape: Tuple[int, int, int, int]
    key_shape: Tuple[int, int, int, int]
    query_dim: int
    heads: int
    drop_rate: float = 0.0
    
    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.
        
        Raises:
            ValueError: If any parameter is invalid.
            TypeError: If parameters have incorrect types.
        """
        # Validate query_shape
        if not isinstance(self.query_shape, tuple):
            raise TypeError(f"query_shape must be a tuple, got {type(self.query_shape)}")
        
        if len(self.query_shape) != 4:
            raise ValueError(
                f"query_shape must have exactly 4 elements (seq_len, height, width, channels), "
                f"got {len(self.query_shape)} elements: {self.query_shape}"
            )
        
        for i, dim in enumerate(self.query_shape):
            if not isinstance(dim, int):
                raise TypeError(
                    f"query_shape[{i}] must be an integer, got {type(dim)}"
                )
            if dim <= 0:
                raise ValueError(
                    f"query_shape[{i}] must be positive, got {dim}"
                )
        
        # Validate key_shape
        if not isinstance(self.key_shape, tuple):
            raise TypeError(f"key_shape must be a tuple, got {type(self.key_shape)}")
        
        if len(self.key_shape) != 4:
            raise ValueError(
                f"key_shape must have exactly 4 elements (seq_len, height, width, channels), "
                f"got {len(self.key_shape)} elements: {self.key_shape}"
            )
        
        for i, dim in enumerate(self.key_shape):
            if not isinstance(dim, int):
                raise TypeError(
                    f"key_shape[{i}] must be an integer, got {type(dim)}"
                )
            if dim <= 0:
                raise ValueError(
                    f"key_shape[{i}] must be positive, got {dim}"
                )
        
        # Validate query_dim
        if not isinstance(self.query_dim, int):
            raise TypeError(f"query_dim must be an integer, got {type(self.query_dim)}")
        
        if self.query_dim <= 0:
            raise ValueError(f"query_dim must be positive, got {self.query_dim}")
        
        # Validate heads
        if not isinstance(self.heads, int):
            raise TypeError(f"heads must be an integer, got {type(self.heads)}")
        
        if self.heads <= 0:
            raise ValueError(f"heads must be positive, got {self.heads}")
        
        # Validate drop_rate
        if not isinstance(self.drop_rate, (int, float)):
            raise TypeError(f"drop_rate must be a number, got {type(self.drop_rate)}")
        
        if not (0.0 <= self.drop_rate <= 1.0):
            raise ValueError(
                f"drop_rate must be between 0.0 and 1.0 (inclusive), got {self.drop_rate}"
            )
    
    @property
    def query_seq_len(self) -> int:
        """Get query sequence length."""
        return self.query_shape[0]
    
    @property
    def query_height(self) -> int:
        """Get query height."""
        return self.query_shape[1]
    
    @property
    def query_width(self) -> int:
        """Get query width."""
        return self.query_shape[2]
    
    @property
    def query_channels(self) -> int:
        """Get query channels."""
        return self.query_shape[3]
    
    @property
    def key_seq_len(self) -> int:
        """Get key sequence length."""
        return self.key_shape[0]
    
    @property
    def key_height(self) -> int:
        """Get key height."""
        return self.key_shape[1]
    
    @property
    def key_width(self) -> int:
        """Get key width."""
        return self.key_shape[2]
    
    @property
    def key_channels(self) -> int:
        """Get key channels."""
        return self.key_shape[3]
    
    @property
    def max_height_dist(self) -> int:
        """Calculate maximum height distance for embeddings."""
        return max(self.query_height, self.key_height) - 1
    
    @property
    def max_width_dist(self) -> int:
        """Calculate maximum width distance for embeddings."""
        return max(self.query_width, self.key_width) - 1
    
    @property
    def max_time_dist(self) -> int:
        """Calculate maximum time distance for embeddings."""
        return max(self.query_seq_len, self.key_seq_len) - 1 