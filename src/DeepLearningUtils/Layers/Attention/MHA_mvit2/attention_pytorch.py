"""
PyTorch implementation of MVit2 multi-head attention layers.

This implements multi-head dot product attention with support for relative positional
embeddings, following the MVit2 architecture design. The implementation includes
both standard softmax attention and linear attention variants.

Reference:
Copyright 2019, Facebook, Inc
Licensed under the Apache License, Version 2.0

Original implementation:
https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py

Adapted by Paul Thompson 2024
"""

from typing import Union, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_config import DotProductAttentionConfig, MultiHeadAttentionConfig
from ...RelativePositionalEmbedding.positional_embedding_pytorch import (
    RelativePositionalEmbedding2D,
    RelativePositionalEmbedding2DKey,
    load_positionaL_embedding_layer_weights
)


class DotProductAttention(nn.Module):
    """
    PyTorch implementation of dot product attention with optional positional embeddings.
    
    This layer computes attention scores using dot product between queries and keys,
    optionally applies scaling and positional embeddings, and produces weighted
    combinations of values. Supports both softmax and linear attention mechanisms.
    
    The layer handles spatial-temporal inputs with shapes corresponding to video
    sequences or multi-scale feature maps, making it suitable for MVit2 architectures.
    
    Examples:
        >>> # Basic softmax attention
        >>> config = DotProductAttentionConfig(
        ...     query_shape=(1, 8, 8, 64),
        ...     key_shape=(1, 8, 8, 64),
        ...     query_dim=16,
        ...     heads=8
        ... )
        >>> attention = DotProductAttention(config)
        >>> 
        >>> # Linear attention without positional embeddings
        >>> attention = DotProductAttention(
        ...     query_shape=(2, 16, 16, 128),
        ...     key_shape=(2, 8, 8, 128),
        ...     query_dim=32,
        ...     heads=4,
        ...     attention_type="linear",
        ...     use_positional_embedding=False
        ... )
    
    Attributes:
        config: Configuration dataclass containing all layer parameters.
        
    Pre-conditions:
        - Query tensor must have shape (batch, heads, seq_len*height*width, query_dim)
        - Key tensor must have shape (batch, heads, seq_len*height*width, query_dim)
        - Value tensor must have shape (batch, heads, seq_len*height*width, query_dim)
        - All tensors must be on same device
        
    Post-conditions:
        - Output has same shape as query tensor
        - Attention weights sum to 1 along key dimension (for softmax attention)
        - Residual connection is applied to query
    """
    
    def __init__(
        self,
        config: Union[DotProductAttentionConfig, None] = None,
        *,
        query_shape: Union[Tuple[int, int, int, int], None] = None,
        key_shape: Union[Tuple[int, int, int, int], None] = None,
        query_dim: Union[int, None] = None,
        heads: Union[int, None] = None,
        use_scale: bool = True,
        drop_rate: float = 0.0,
        use_positional_embedding: bool = True,
        use_key_positional_embedding: bool = True,
        attention_type: str = "softmax",
        name: str = ""
    ) -> None:
        """
        Initialize DotProductAttention layer.
        
        Args:
            config: Configuration dataclass. If provided, other parameters are ignored.
            query_shape: Shape of query tensor. Used only if config is None.
            key_shape: Shape of key tensor. Used only if config is None.
            query_dim: Dimension of query embeddings per head. Used only if config is None.
            heads: Number of attention heads. Used only if config is None.
            use_scale: Whether to scale attention scores. Used only if config is None.
            drop_rate: Dropout rate for attention weights. Used only if config is None.
            use_positional_embedding: Whether to use query positional embeddings. Used only if config is None.
            use_key_positional_embedding: Whether to use key positional embeddings. Used only if config is None.
            attention_type: Type of attention mechanism. Used only if config is None.
            name: Name identifier for the layer. Used only if config is None.
            
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
            
            self.config = DotProductAttentionConfig(
                query_shape=query_shape,  # type: ignore
                key_shape=key_shape,      # type: ignore
                query_dim=query_dim,      # type: ignore
                heads=heads,              # type: ignore
                use_scale=use_scale,
                drop_rate=drop_rate,
                use_positional_embedding=use_positional_embedding,
                use_key_positional_embedding=use_key_positional_embedding,
                attention_type=attention_type,  # type: ignore
                name=name
            )
        else:
            self.config = config
        
        # Store individual attributes for JIT compatibility
        self.query_dim = self.config.query_dim
        self.heads = self.config.heads
        self.use_scale = self.config.use_scale
        self.drop_rate = self.config.drop_rate
        self.use_positional_embedding = self.config.use_positional_embedding
        self.use_key_positional_embedding = self.config.use_key_positional_embedding
        self.attention_type = self.config.attention_type
        self.name = self.config.name
        
        # Create positional embedding modules if needed
        if self.use_positional_embedding:
            self.query_positional_embedding = RelativePositionalEmbedding2D(
                query_shape=self.config.query_shape,
                key_shape=self.config.key_shape,
                query_dim=self.query_dim,
                heads=self.heads,
                drop_rate=self.drop_rate
            )
        else:
            self.query_positional_embedding = None

        if self.use_key_positional_embedding:
            self.key_positional_embedding = RelativePositionalEmbedding2DKey(
                query_shape=self.config.query_shape,
                key_shape=self.config.key_shape,
                query_dim=self.query_dim,
                heads=self.heads,
                drop_rate=self.drop_rate
            )
        else:
            self.key_positional_embedding = None

        # Create dropout layer
        self.drop = nn.Dropout(p=self.drop_rate)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key_input: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply dot product attention to input tensors.
        
        Args:
            query: Query tensor with shape (batch, heads, seq_len*height*width, query_dim).
            key_input: Key tensor with shape (batch, heads, seq_len*height*width, query_dim).
            value: Value tensor with shape (batch, heads, seq_len*height*width, query_dim).
            mask: Optional attention mask with shape (batch, heads, query_len, key_len).
            
        Returns:
            Attention output with same shape as query tensor.
            
        Pre-conditions:
            - All input tensors must be 4D
            - query.dim() == key_input.dim() == value.dim() == 4
            - All tensors must be on same device
            - mask must be broadcastable to attention score shape if provided
            
        Post-conditions:
            - Output has same shape as query
            - Residual connection is applied to query
            - Attention weights are normalized (for softmax attention)
            
        Raises:
            RuntimeError: If input tensors have incompatible shapes.
        """
        if query.dim() != 4:
            raise RuntimeError(
                f"Expected 4D query tensor (batch, heads, seq_len, query_dim), "
                f"got {query.dim()}D tensor with shape {query.shape}"
            )
        
        if key_input.dim() != 4:
            raise RuntimeError(
                f"Expected 4D key tensor (batch, heads, seq_len, query_dim), "
                f"got {key_input.dim()}D tensor with shape {key_input.shape}"
            )
        
        if value.dim() != 4:
            raise RuntimeError(
                f"Expected 4D value tensor (batch, heads, seq_len, query_dim), "
                f"got {value.dim()}D tensor with shape {value.shape}"
            )
        
        if self.attention_type == "linear":
            return self._linear_attention(query, key_input, value, mask)
        else:
            return self._softmax_attention(query, key_input, value, mask)
    
    def _linear_attention(
        self, 
        query: torch.Tensor, 
        key_input: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply linear attention mechanism."""
        # Apply ReLU activation for linear attention
        query_activated = F.relu(query)
        key_activated = F.relu(key_input)
        
        # Transpose key for matrix multiplication
        key = key_activated.transpose(-2, -1)
        
        # Compute attention scores
        score = torch.matmul(query_activated, key)
        
        # Normalize by sum of scores
        score = score / torch.sum(score, dim=1, keepdim=True)
        
        # Apply positional embeddings if enabled
        if self.use_positional_embedding:
            positional_embedding = self.query_positional_embedding
            score = positional_embedding(query, score)
        
        if self.use_key_positional_embedding:
            positional_embedding = self.key_positional_embedding
            score = positional_embedding(key_input, score)
        
        # Apply dropout
        score = self.drop(score)
        
        # Apply mask if provided
        if mask is not None:
            score *= mask
        
        # Compute final output
        score = torch.matmul(score, value)
        
        # Add residual connection
        score += query
        
        return score
    
    def _softmax_attention(
        self, 
        query: torch.Tensor, 
        key_input: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply softmax attention mechanism."""
        # Transpose key for matrix multiplication
        key = key_input.transpose(-2, -1)
        
        # Compute attention scores
        score = torch.matmul(query, key)
        
        # Apply scaling if enabled
        if self.use_scale:
            scale = torch.sqrt(torch.tensor(self.query_dim, dtype=torch.float32, device=query.device))
            score = score / scale
        
        # Apply positional embeddings if enabled
        if self.use_positional_embedding:
            positional_embedding = self.query_positional_embedding
            score = positional_embedding(query, score)
        
        if self.use_key_positional_embedding:
            positional_embedding = self.key_positional_embedding
            score = positional_embedding(key_input, score)
        
        # Apply mask if provided
        if mask is not None:
            score = score - 1e9 * (1 - mask)
        
        # Apply softmax normalization
        score = F.softmax(score, dim=-1)
        
        # Apply dropout
        score = self.drop(score)
        
        # Compute final output
        score = torch.matmul(score, value)
        
        # Add residual connection (MVit2 residual connection)
        score += query
        
        return score
    
    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"query_shape={self.config.query_shape}, "
            f"key_shape={self.config.key_shape}, "
            f"query_dim={self.config.query_dim}, "
            f"heads={self.config.heads}, "
            f"attention_type='{self.config.attention_type}', "
            f"use_scale={self.config.use_scale}, "
            f"drop_rate={self.config.drop_rate}"
        )


class MultiHeadAttention(nn.Module):
    """
    PyTorch implementation of multi-head attention with optional positional embeddings.
    
    This layer implements the complete multi-head attention mechanism including
    query/key/value projections, parallel attention computation across multiple heads,
    and output projection. Supports MVit2-style residual connections and both
    softmax and linear attention variants.
    
    The implementation handles spatial-temporal inputs efficiently through tensor
    reshaping and supports different resolutions for query and key inputs.
    
    Examples:
        >>> # Basic multi-head attention
        >>> config = MultiHeadAttentionConfig(
        ...     query_shape=(1, 8, 8, 64),
        ...     key_shape=(1, 8, 8, 64),
        ...     heads=8,
        ...     value_dim=128,
        ...     key_dim=128
        ... )
        >>> mha = MultiHeadAttention(config)
        >>> 
        >>> # Cross-attention with different shapes
        >>> mha = MultiHeadAttention(
        ...     query_shape=(2, 16, 16, 128),
        ...     key_shape=(2, 8, 8, 128),
        ...     heads=4,
        ...     value_dim=256,
        ...     use_positional_embedding=False
        ... )
    
    Attributes:
        config: Configuration dataclass containing all layer parameters.
        
    Pre-conditions:
        - Query tensor must have shape (batch, seq_len*height*width, channels)
        - Key tensor must have shape (batch, seq_len*height*width, channels)
        - Value tensor must have shape (batch, seq_len*height*width, channels)
        - All tensors must be on same device
        
    Post-conditions:
        - Output has same shape as query tensor
        - Output channels match input query channels
        - Attention computation respects head parallelization
    """
    
    def __init__(
        self,
        config: Union[MultiHeadAttentionConfig, None] = None,
        *,
        query_shape: Union[Tuple[int, int, int, int], None] = None,
        key_shape: Union[Tuple[int, int, int, int], None] = None,
        heads: int = 8,
        value_dim: int = 128,
        key_dim: int = 128,
        attention_drop_rate: float = 0.0,
        use_positional_embedding: bool = True,
        use_key_positional_embedding: bool = True,
        attention_type: str = "softmax",
        use_query_embedding: bool = True,
        use_key_embedding: bool = True,
        use_value_embedding: bool = True
    ) -> None:
        """
        Initialize MultiHeadAttention layer.
        
        Args:
            config: Configuration dataclass. If provided, other parameters are ignored.
            query_shape: Shape of query tensor. Used only if config is None.
            key_shape: Shape of key tensor. Used only if config is None.
            heads: Number of attention heads. Used only if config is None.
            value_dim: Total dimension for value projections. Used only if config is None.
            key_dim: Total dimension for key/query projections. Used only if config is None.
            attention_drop_rate: Dropout rate for attention. Used only if config is None.
            use_positional_embedding: Whether to use query positional embeddings. Used only if config is None.
            use_key_positional_embedding: Whether to use key positional embeddings. Used only if config is None.
            attention_type: Type of attention mechanism. Used only if config is None.
            use_query_embedding: Whether to project input queries. Used only if config is None.
            use_key_embedding: Whether to project input keys. Used only if config is None.
            use_value_embedding: Whether to project input values. Used only if config is None.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__()
        
        # Create configuration from parameters or use provided config
        if config is None:
            if any(param is None for param in [query_shape, key_shape]):
                raise ValueError(
                    "Either config must be provided, or both query_shape and key_shape "
                    "must be specified"
                )
            
            self.config = MultiHeadAttentionConfig(
                query_shape=query_shape,  # type: ignore
                key_shape=key_shape,      # type: ignore
                heads=heads,
                value_dim=value_dim,
                key_dim=key_dim,
                attention_drop_rate=attention_drop_rate,
                use_positional_embedding=use_positional_embedding,
                use_key_positional_embedding=use_key_positional_embedding,
                attention_type=attention_type,  # type: ignore
                use_query_embedding=use_query_embedding,
                use_key_embedding=use_key_embedding,
                use_value_embedding=use_value_embedding
            )
        else:
            self.config = config
        
        # Store individual attributes for JIT compatibility
        self.heads = self.config.heads
        self.value_dim = self.config.value_dim
        self.key_dim = self.config.key_dim
        self.attention_drop_rate = self.config.attention_drop_rate
        self.use_positional_embedding = self.config.use_positional_embedding
        self.use_key_positional_embedding = self.config.use_key_positional_embedding
        self.use_query_embedding = self.config.use_query_embedding
        self.use_key_embedding = self.config.use_key_embedding
        self.use_value_embedding = self.config.use_value_embedding
        self.query_channels = self.config.query_channels
        self.key_channels = self.config.key_channels
        self.head_dim = self.config.head_dim
        self.key_head_dim = self.config.key_head_dim
        
        # Calculate dimensions
        head_dim = self.config.head_dim
        key_head_dim = self.config.key_head_dim
        
        # Create embedding layers
        if self.use_query_embedding:
            self.query_dense = nn.Linear(self.query_channels, self.key_dim, bias=False)
        else:
            self.query_dense = nn.Identity()
        
        if self.use_key_embedding:
            self.key_dense = nn.Linear(self.key_channels, self.key_dim, bias=False)
        else:
            self.key_dense = nn.Identity()
        
        if self.use_value_embedding:
            self.value_dense = nn.Linear(self.key_channels, self.value_dim, bias=False)
        else:
            self.value_dense = nn.Identity()
        
        # Create attention mechanism
        attention_config = self.config.get_attention_config()
        self.attention = DotProductAttention(config=attention_config)
        
        # Create output projection
        self.dense_out = nn.Linear(
            self.value_dim,
            self.query_channels,
            bias=True
        )
        
        # Create output dropout
        self.output_dropout = nn.Dropout(self.attention_drop_rate)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention to input tensors.
        
        Args:
            query: Query tensor with shape (batch, seq_len*height*width, channels).
            key: Key tensor with shape (batch, seq_len*height*width, channels).
            value: Value tensor with shape (batch, seq_len*height*width, channels).
            mask: Optional attention mask.
            
        Returns:
            Attention output with same shape as query tensor.
            
        Pre-conditions:
            - All input tensors must be 3D: (batch, sequence_length, channels)
            - All tensors must be on same device
            - Sequence lengths must match expected configuration
            
        Post-conditions:
            - Output has same shape as input query
            - Output channels match input query channels
            - Residual connections and normalization are applied as configured
            
        Raises:
            RuntimeError: If input tensors have incompatible shapes.
        """
        if query.dim() != 3:
            raise RuntimeError(
                f"Expected 3D query tensor (batch, seq_len, channels), "
                f"got {query.dim()}D tensor with shape {query.shape}"
            )
        
        if key.dim() != 3:
            raise RuntimeError(
                f"Expected 3D key tensor (batch, seq_len, channels), "
                f"got {key.dim()}D tensor with shape {key.shape}"
            )
        
        if value.dim() != 3:
            raise RuntimeError(
                f"Expected 3D value tensor (batch, seq_len, channels), "
                f"got {value.dim()}D tensor with shape {value.shape}"
            )
        
        batch_size = query.size(0)
        
        # Apply input projections
        q = self.query_dense(query)
        k = self.key_dense(key)
        v = self.value_dense(value)
        
        # Reshape for multi-head attention
        # From (batch, seq_len, dim) to (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.heads, self.key_head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.key_head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        attention_output = self.attention(q, k, v, mask=mask)
        
        # Reshape back to original format
        # From (batch, heads, seq_len, head_dim) to (batch, seq_len, value_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.value_dim
        )
        
        # Apply output projection
        output = self.dense_out(attention_output)
        
        # Apply output dropout
        output = self.output_dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"heads={self.heads}, "
            f"value_dim={self.value_dim}, "
            f"key_dim={self.key_dim}, "
            f"attention_drop_rate={self.attention_drop_rate}"
        )


def load_mha_positional_layer_weights(
    keras_layer, 
    pytorch_module
) -> None:
    """
    Load weights from Keras multi-head attention layer to PyTorch module.
    
    This function transfers trained weights from a Keras MultiHeadAttention layer
    to the corresponding PyTorch MultiHeadAttention module, handling the different
    weight layouts and parameter structures between frameworks.
    
    Args:
        keras_layer: Source Keras MultiHeadAttention layer with trained weights.
        pytorch_module: Target PyTorch MultiHeadAttention module to load weights into.
        
    Pre-conditions:
        - keras_layer must be a trained Keras MultiHeadAttention instance
        - pytorch_module must be a compatible PyTorch MultiHeadAttention instance
        - Both layers must have same architectural configuration
        
    Post-conditions:
        - PyTorch module parameters are updated with Keras weights
        - Weight tensors are properly transposed for PyTorch format
        - Positional embedding weights are transferred if enabled
        
    Raises:
        AttributeError: If layer structures are incompatible.
        ValueError: If weight dimensions don't match.
    """
    print(f"Loading custom layer weights for {keras_layer.name}")

    # Load query weights
    if keras_layer.query_embedding:
        pytorch_module.query_dense.weight.data = torch.tensor(
            keras_layer.query_dense.get_weights()[0].T
        )

    # Load key weights
    if keras_layer.key_embedding:
        pytorch_module.key_dense.weight.data = torch.tensor(
            keras_layer.key.get_weights()[0].T
        )

    # Load value weights
    if keras_layer.value_embedding:
        pytorch_module.value_dense.weight.data = torch.tensor(
            keras_layer.value.get_weights()[0].T
        )

    # Load positional embedding weights
    if keras_layer.use_positional_embedding:
        load_positionaL_embedding_layer_weights(
            keras_layer.att.query_positional_embedding,
            pytorch_module.attention.query_positional_embedding
        )

    if keras_layer.use_key_positional_embedding:
        load_positionaL_embedding_layer_weights(
            keras_layer.att.key_positional_embedding,
            pytorch_module.attention.key_positional_embedding
        )
    
    # Load output dense weights and bias
    pytorch_module.dense_out.weight.data = torch.tensor(
        keras_layer.out.get_weights()[0].T
    )
    pytorch_module.dense_out.bias.data = torch.tensor(
        keras_layer.out.get_weights()[1]
    )
