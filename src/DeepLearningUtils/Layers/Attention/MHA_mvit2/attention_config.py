"""
Configuration dataclasses for MVit2 attention layers.

This module provides configuration objects for multi-head attention implementations
that ensure early error detection, type safety, and comprehensive validation.

The configuration supports different query and key tensor shapes, various attention
head configurations, and dropout settings for robust attention mechanisms.
"""

from dataclasses import dataclass
from typing import Tuple, Literal

AttentionType = Literal["softmax", "linear"]


@dataclass(frozen=True)
class DotProductAttentionConfig:
    """
    Configuration for dot product attention mechanism.
    
    This dataclass encapsulates all parameters needed for dot product attention,
    providing comprehensive validation and type safety. The attention mechanism
    computes attention scores between queries and keys, optionally applies
    positional embeddings, and produces weighted outputs.
    
    Attributes:
        query_shape: Shape of query tensor (seq_len, height, width, channels).
        key_shape: Shape of key tensor (seq_len, height, width, channels).
        query_dim: Dimension of query embeddings per attention head.
        heads: Number of attention heads for parallel processing.
        use_scale: Whether to scale attention scores by sqrt(query_dim).
        drop_rate: Dropout rate applied to attention weights (0.0 to 1.0).
        use_positional_embedding: Whether to add query-based positional bias.
        use_key_positional_embedding: Whether to add key-based positional bias.
        attention_type: Type of attention mechanism ("softmax" or "linear").
        name: Optional name identifier for the attention module.
        
    Pre-conditions:
        - All shape dimensions must be positive integers
        - query_dim must be positive and divisible by heads
        - drop_rate must be in range [0.0, 1.0]
        - Query and key shapes must have compatible spatial dimensions
        
    Post-conditions:
        - Configuration is immutable after creation
        - All parameters are validated and type-safe
        - Computed properties are available for tensor reshaping
    """
    
    query_shape: Tuple[int, int, int, int]
    key_shape: Tuple[int, int, int, int] 
    query_dim: int
    heads: int
    use_scale: bool = True
    drop_rate: float = 0.0
    use_positional_embedding: bool = True
    use_key_positional_embedding: bool = True
    attention_type: AttentionType = "softmax"
    name: str = ""
    
    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.
        
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
        
        # Validate key_shape
        if not isinstance(self.key_shape, tuple) or len(self.key_shape) != 4:
            raise ValueError(
                f"key_shape must be a 4-tuple (seq_len, height, width, channels), "
                f"got {self.key_shape}"
            )
        
        if not all(isinstance(dim, int) and dim > 0 for dim in self.key_shape):
            raise ValueError(
                f"All key_shape dimensions must be positive integers, "
                f"got {self.key_shape}"
            )
        
        # Validate query_dim
        if not isinstance(self.query_dim, int) or self.query_dim <= 0:
            raise ValueError(
                f"query_dim must be a positive integer, got {self.query_dim}"
            )
        
        # Validate heads
        if not isinstance(self.heads, int) or self.heads <= 0:
            raise ValueError(
                f"heads must be a positive integer, got {self.heads}"
            )
        
        if self.query_dim % self.heads != 0:
            raise ValueError(
                f"query_dim ({self.query_dim}) must be divisible by heads ({self.heads})"
            )
        
        # Validate drop_rate
        if not isinstance(self.drop_rate, (int, float)) or not (0.0 <= self.drop_rate <= 1.0):
            raise ValueError(
                f"drop_rate must be a float in range [0.0, 1.0], got {self.drop_rate}"
            )
        
        # Validate boolean flags
        if not isinstance(self.use_scale, bool):
            raise TypeError(f"use_scale must be a boolean, got {type(self.use_scale)}")
        
        if not isinstance(self.use_positional_embedding, bool):
            raise TypeError(f"use_positional_embedding must be a boolean, got {type(self.use_positional_embedding)}")
        
        if not isinstance(self.use_key_positional_embedding, bool):
            raise TypeError(f"use_key_positional_embedding must be a boolean, got {type(self.use_key_positional_embedding)}")
        
        # Validate attention_type
        if self.attention_type not in ["softmax", "linear"]:
            raise ValueError(
                f"attention_type must be 'softmax' or 'linear', got '{self.attention_type}'"
            )
        
        # Validate name
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a string, got {type(self.name)}")
    
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
    def key_seq_len(self) -> int:
        """Sequence length of key tensor."""
        return self.key_shape[0]
    
    @property
    def key_height(self) -> int:
        """Height dimension of key tensor."""
        return self.key_shape[1]
    
    @property
    def key_width(self) -> int:
        """Width dimension of key tensor."""
        return self.key_shape[2]
    
    @property
    def key_channels(self) -> int:
        """Channel dimension of key tensor."""
        return self.key_shape[3]
    
    @property
    def query_spatial_size(self) -> int:
        """Total spatial size of query (height * width)."""
        return self.query_height * self.query_width
    
    @property
    def key_spatial_size(self) -> int:
        """Total spatial size of key (height * width)."""
        return self.key_height * self.key_width
    
    @property
    def query_total_size(self) -> int:
        """Total size of query tensor (seq_len * height * width)."""
        return self.query_seq_len * self.query_spatial_size
    
    @property
    def key_total_size(self) -> int:
        """Total size of key tensor (seq_len * height * width)."""
        return self.key_seq_len * self.key_spatial_size
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.query_dim // self.heads


@dataclass(frozen=True)
class MultiHeadAttentionConfig:
    """
    Configuration for multi-head attention layer.
    
    This configuration manages the complete multi-head attention setup including
    query/key/value projections, attention computation, and output projection.
    
    Attributes:
        query_shape: Shape of input query tensor (seq_len, height, width, channels).
        key_shape: Shape of input key tensor (seq_len, height, width, channels).
        heads: Number of attention heads.
        value_dim: Dimension of value projections (total across all heads).
        key_dim: Dimension of key/query projections (total across all heads).
        attention_drop_rate: Dropout rate for attention weights.
        use_positional_embedding: Whether to use query-based positional embeddings.
        use_key_positional_embedding: Whether to use key-based positional embeddings.
        attention_type: Type of attention mechanism.
        output_activation: Optional activation function for output.
        use_query_embedding: Whether to project input queries.
        use_key_embedding: Whether to project input keys.
        use_value_embedding: Whether to project input values.
        
    Pre-conditions:
        - value_dim and key_dim must be divisible by heads
        - All shape dimensions must be positive
        - attention_drop_rate must be in [0.0, 1.0]
        
    Post-conditions:
        - All derived attention configurations are valid
        - Embedding dimensions are properly computed
    """
    
    query_shape: Tuple[int, int, int, int]
    key_shape: Tuple[int, int, int, int]
    heads: int = 8
    value_dim: int = 128
    key_dim: int = 128
    attention_drop_rate: float = 0.0
    use_positional_embedding: bool = True
    use_key_positional_embedding: bool = True
    attention_type: AttentionType = "softmax"
    output_activation: str = None
    use_query_embedding: bool = True
    use_key_embedding: bool = True
    use_value_embedding: bool = True
    
    def __post_init__(self) -> None:
        """
        Validate multi-head attention configuration.
        
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
        
        # Validate key_shape
        if not isinstance(self.key_shape, tuple) or len(self.key_shape) != 4:
            raise ValueError(
                f"key_shape must be a 4-tuple (seq_len, height, width, channels), "
                f"got {self.key_shape}"
            )
        
        if not all(isinstance(dim, int) and dim > 0 for dim in self.key_shape):
            raise ValueError(
                f"All key_shape dimensions must be positive integers, "
                f"got {self.key_shape}"
            )
        
        # Validate heads
        if not isinstance(self.heads, int) or self.heads <= 0:
            raise ValueError(f"heads must be a positive integer, got {self.heads}")
        
        # Validate dimensions
        if not isinstance(self.value_dim, int) or self.value_dim <= 0:
            raise ValueError(f"value_dim must be a positive integer, got {self.value_dim}")
        
        if not isinstance(self.key_dim, int) or self.key_dim <= 0:
            raise ValueError(f"key_dim must be a positive integer, got {self.key_dim}")
        
        if self.value_dim % self.heads != 0:
            raise ValueError(
                f"value_dim ({self.value_dim}) must be divisible by heads ({self.heads})"
            )
        
        if self.key_dim % self.heads != 0:
            raise ValueError(
                f"key_dim ({self.key_dim}) must be divisible by heads ({self.heads})"
            )
        
        # Validate dropout
        if not isinstance(self.attention_drop_rate, (int, float)) or not (0.0 <= self.attention_drop_rate <= 1.0):
            raise ValueError(
                f"attention_drop_rate must be in range [0.0, 1.0], got {self.attention_drop_rate}"
            )
        
        # Validate boolean flags
        for flag_name in ["use_positional_embedding", "use_key_positional_embedding", 
                         "use_query_embedding", "use_key_embedding", "use_value_embedding"]:
            flag_value = getattr(self, flag_name)
            if not isinstance(flag_value, bool):
                raise TypeError(f"{flag_name} must be a boolean, got {type(flag_value)}")
        
        # Validate attention_type
        if self.attention_type not in ["softmax", "linear"]:
            raise ValueError(
                f"attention_type must be 'softmax' or 'linear', got '{self.attention_type}'"
            )
    
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
    def key_seq_len(self) -> int:
        """Sequence length of key tensor."""
        return self.key_shape[0]
    
    @property
    def key_height(self) -> int:
        """Height dimension of key tensor."""
        return self.key_shape[1]
    
    @property
    def key_width(self) -> int:
        """Width dimension of key tensor."""
        return self.key_shape[2]
    
    @property
    def key_channels(self) -> int:
        """Channel dimension of key tensor."""
        return self.key_shape[3]
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head for value projection."""
        return self.value_dim // self.heads
    
    @property
    def key_head_dim(self) -> int:
        """Dimension per attention head for key/query projections."""
        return self.key_dim // self.heads
    
    def get_attention_config(self) -> DotProductAttentionConfig:
        """
        Create DotProductAttentionConfig for the attention mechanism.
        
        Returns:
            Configuration for the underlying dot product attention.
        """
        return DotProductAttentionConfig(
            query_shape=self.query_shape,
            key_shape=self.key_shape,
            query_dim=self.key_head_dim,  # Use key_head_dim for attention computation
            heads=self.heads,
            use_scale=True,
            drop_rate=self.attention_drop_rate,
            use_positional_embedding=self.use_positional_embedding,
            use_key_positional_embedding=self.use_key_positional_embedding,
            attention_type=self.attention_type,
            name="attention"
        )
