"""
Configuration dataclasses for sequential co-attention layers.

This module provides comprehensive configuration management for sequential co-attention
mechanisms, ensuring type safety, validation, and immutability following established
design guidelines.

The configuration system supports both memory attention modules and full co-attention
modules with temporal sequencing capabilities.
"""

from dataclasses import dataclass
from typing import Tuple, Union, Literal


@dataclass(frozen=True)
class CoMemoryAttentionConfig:
    """
    Configuration for co-memory attention module.
    
    This configuration manages cross-attention between query sequences and memory banks,
    supporting spatial-temporal attention with configurable normalization and positional
    embeddings.
    
    Attributes:
        query_shape: Shape of query tensor (seq_len, height, width, channels).
        memory_shape: Shape of memory tensor (seq_len, height, width, channels).
        key_dim: Dimension for key/query projections.
        value_dim: Dimension for value projections.
        use_norm: Whether to apply layer normalization.
        attention_drop_rate: Dropout rate for attention weights.
        use_positional_embedding: Whether to use query positional embeddings.
        use_key_positional_embedding: Whether to use key positional embeddings.
        attention_heads: Number of attention heads.
        use_qkv_embedding: Whether to use query/key/value projections.
        
    Pre-conditions:
        - All shape dimensions must be positive
        - key_dim and value_dim must be divisible by attention_heads
        - attention_drop_rate must be in [0.0, 1.0]
        
    Post-conditions:
        - Configuration is immutable after creation
        - All parameters are validated and type-safe
        - Multi-head attention configuration can be derived
    """
    
    query_shape: Tuple[int, int, int, int]
    memory_shape: Tuple[int, int, int, int]
    key_dim: int = 128
    value_dim: int = 256
    use_norm: bool = False
    attention_drop_rate: float = 0.0
    use_positional_embedding: bool = True
    use_key_positional_embedding: bool = True
    attention_heads: int = 8
    use_qkv_embedding: bool = False
    
    def __post_init__(self) -> None:
        """
        Validate co-memory attention configuration.
        
        Raises:
            ValueError: If any parameter fails validation.
            TypeError: If parameters have incorrect types.
        """
        # Validate query_shape
        if not isinstance(self.query_shape, tuple) or len(self.query_shape) != 4:
            raise ValueError(
                f"query_shape must be a 4-tuple (seq_len, height, width, channels), "
                f"got {self.query_shape}"
            )
        
        if not all(isinstance(dim, int) and dim > 0 for dim in self.query_shape):
            raise ValueError(
                f"All query_shape dimensions must be positive integers, "
                f"got {self.query_shape}"
            )
        
        # Validate memory_shape
        if not isinstance(self.memory_shape, tuple) or len(self.memory_shape) != 4:
            raise ValueError(
                f"memory_shape must be a 4-tuple (seq_len, height, width, channels), "
                f"got {self.memory_shape}"
            )
        
        if not all(isinstance(dim, int) and dim > 0 for dim in self.memory_shape):
            raise ValueError(
                f"All memory_shape dimensions must be positive integers, "
                f"got {self.memory_shape}"
            )
        
        # Validate dimensions
        if not isinstance(self.key_dim, int) or self.key_dim <= 0:
            raise ValueError(f"key_dim must be a positive integer, got {self.key_dim}")
        
        if not isinstance(self.value_dim, int) or self.value_dim <= 0:
            raise ValueError(f"value_dim must be a positive integer, got {self.value_dim}")
        
        if not isinstance(self.attention_heads, int) or self.attention_heads <= 0:
            raise ValueError(f"attention_heads must be a positive integer, got {self.attention_heads}")
        
        # Validate divisibility
        if self.key_dim % self.attention_heads != 0:
            raise ValueError(
                f"key_dim ({self.key_dim}) must be divisible by attention_heads ({self.attention_heads})"
            )
        
        if self.value_dim % self.attention_heads != 0:
            raise ValueError(
                f"value_dim ({self.value_dim}) must be divisible by attention_heads ({self.attention_heads})"
            )
        
        # Validate dropout
        if not isinstance(self.attention_drop_rate, (int, float)) or not (0.0 <= self.attention_drop_rate <= 1.0):
            raise ValueError(
                f"attention_drop_rate must be in range [0.0, 1.0], got {self.attention_drop_rate}"
            )
        
        # Validate boolean flags
        for flag_name in ["use_norm", "use_positional_embedding", "use_key_positional_embedding", "use_qkv_embedding"]:
            flag_value = getattr(self, flag_name)
            if not isinstance(flag_value, bool):
                raise TypeError(f"{flag_name} must be a boolean, got {type(flag_value)}")
    
    @property
    def query_seq_len(self) -> int:
        """Sequence length of query tensor."""
        return self.query_shape[0]
    
    @property
    def query_height(self) -> int:
        """Height dimension of query tensor."""
        return self.query_shape[1]
    
    @property
    def query_width(self) -> int:
        """Width dimension of query tensor."""
        return self.query_shape[2]
    
    @property
    def query_channels(self) -> int:
        """Channel dimension of query tensor."""
        return self.query_shape[3]
    
    @property
    def memory_seq_len(self) -> int:
        """Sequence length of memory tensor."""
        return self.memory_shape[0]
    
    @property
    def memory_height(self) -> int:
        """Height dimension of memory tensor."""
        return self.memory_shape[1]
    
    @property
    def memory_width(self) -> int:
        """Width dimension of memory tensor."""
        return self.memory_shape[2]
    
    @property
    def memory_channels(self) -> int:
        """Channel dimension of memory tensor."""
        return self.memory_shape[3]
    
    @property
    def query_spatial_size(self) -> int:
        """Total spatial size of query (height * width)."""
        return self.query_height * self.query_width
    
    @property
    def memory_spatial_size(self) -> int:
        """Total spatial size of memory (height * width)."""
        return self.memory_height * self.memory_width
    
    @property
    def query_total_size(self) -> int:
        """Total size of query tensor (seq_len * height * width)."""
        return self.query_seq_len * self.query_spatial_size
    
    @property
    def memory_total_size(self) -> int:
        """Total size of memory tensor (seq_len * height * width)."""
        return self.memory_seq_len * self.memory_spatial_size


@dataclass(frozen=True)
class CoAttentionConfig:
    """
    Configuration for sequential co-attention module.
    
    This configuration manages the full sequential co-attention mechanism including
    temporal processing, MLP components, and normalization layers.
    
    Attributes:
        query_shape: Shape of query tensor (batch, seq_len, height, width, channels).
        memory_shape: Shape of memory tensor (batch, frames, height, width, channels).
        key_dim: Dimension for key projections.
        value_dim: Dimension for value projections.
        hidden_dim: Hidden dimension for MLP layers.
        layer_norm_eps: Epsilon value for layer normalization.
        
    Pre-conditions:
        - All shape dimensions must be positive
        - query_shape and memory_shape must be 5-tuples
        - Spatial dimensions should be compatible
        
    Post-conditions:
        - Configuration is immutable after creation
        - All parameters are validated and type-safe
        - MLP and normalization configurations are derived
    """
    
    query_shape: Tuple[int, int, int, int, int]
    memory_shape: Tuple[int, int, int, int, int]
    key_dim: int = 128
    value_dim: int = 128
    hidden_dim: int = 512
    layer_norm_eps: float = 1e-3
    
    def __post_init__(self) -> None:
        """
        Validate co-attention configuration.
        
        Raises:
            ValueError: If any parameter fails validation.
            TypeError: If parameters have incorrect types.
        """
        # Validate query_shape
        if not isinstance(self.query_shape, tuple) or len(self.query_shape) != 5:
            raise ValueError(
                f"query_shape must be a 5-tuple (batch, seq_len, height, width, channels), "
                f"got {self.query_shape}"
            )
        
        if not all(isinstance(dim, int) and dim > 0 for dim in self.query_shape):
            raise ValueError(
                f"All query_shape dimensions must be positive integers, "
                f"got {self.query_shape}"
            )
        
        # Validate memory_shape
        if not isinstance(self.memory_shape, tuple) or len(self.memory_shape) != 5:
            raise ValueError(
                f"memory_shape must be a 5-tuple (batch, frames, height, width, channels), "
                f"got {self.memory_shape}"
            )
        
        if not all(isinstance(dim, int) and dim > 0 for dim in self.memory_shape):
            raise ValueError(
                f"All memory_shape dimensions must be positive integers, "
                f"got {self.memory_shape}"
            )
        
        # Validate dimensions
        if not isinstance(self.key_dim, int) or self.key_dim <= 0:
            raise ValueError(f"key_dim must be a positive integer, got {self.key_dim}")
        
        if not isinstance(self.value_dim, int) or self.value_dim <= 0:
            raise ValueError(f"value_dim must be a positive integer, got {self.value_dim}")
        
        if not isinstance(self.hidden_dim, int) or self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {self.hidden_dim}")
        
        # Validate epsilon
        if not isinstance(self.layer_norm_eps, float) or self.layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be a positive float, got {self.layer_norm_eps}")
    
    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self.query_shape[0]
    
    @property
    def query_seq_len(self) -> int:
        """Sequence length of query."""
        return self.query_shape[1]
    
    @property
    def query_height(self) -> int:
        """Height dimension of query."""
        return self.query_shape[2]
    
    @property
    def query_width(self) -> int:
        """Width dimension of query."""
        return self.query_shape[3]
    
    @property
    def query_channels(self) -> int:
        """Channel dimension of query."""
        return self.query_shape[4]
    
    @property
    def memory_frames(self) -> int:
        """Number of memory frames."""
        return self.memory_shape[1]
    
    @property
    def memory_height(self) -> int:
        """Height dimension of memory."""
        return self.memory_shape[2]
    
    @property
    def memory_width(self) -> int:
        """Width dimension of memory."""
        return self.memory_shape[3]
    
    @property
    def memory_channels(self) -> int:
        """Channel dimension of memory."""
        return self.memory_shape[4] 