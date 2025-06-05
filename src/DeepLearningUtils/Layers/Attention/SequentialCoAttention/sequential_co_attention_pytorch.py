"""
PyTorch implementation of sequential co-attention layers.

This implements sequential co-attention mechanisms for memory-based attention,
allowing queries to attend to multiple memory frames in temporal sequence.
The implementation supports configurable normalization, positional embeddings,
and multi-head attention patterns.

Adapted and modernized following design guidelines with comprehensive validation,
type safety, and configuration management.
"""

from typing import Union, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequential_co_attention_config import CoMemoryAttentionConfig, CoAttentionConfig
from ..MHA_mvit2.attention_pytorch import MultiHeadAttention, load_mha_positional_layer_weights
from ....utils.model_conversion_helpers import load_layer_norm_weights, load_linear_weights


class CoMemoryAttentionModule(nn.Module):
    """
    PyTorch implementation of co-memory attention module.
    
    This module implements cross-attention between query sequences and memory banks,
    supporting spatial-temporal attention with configurable normalization and
    positional embeddings. The module handles tensor reshaping for multi-head
    attention and provides memory masking capabilities.
    
    Examples:
        >>> # Basic usage with default configuration
        >>> config = CoMemoryAttentionConfig(
        ...     query_shape=(1, 16, 16, 128),
        ...     memory_shape=(1, 16, 16, 128)
        ... )
        >>> module = CoMemoryAttentionModule(config)
        >>> 
        >>> # Custom configuration with normalization
        >>> module = CoMemoryAttentionModule(
        ...     query_shape=(1, 8, 8, 64),
        ...     memory_shape=(2, 8, 8, 64),
        ...     key_dim=64,
        ...     value_dim=128,
        ...     use_norm=True,
        ...     attention_heads=4
        ... )
    
    Attributes:
        config: Configuration dataclass containing all module parameters.
        
    Pre-conditions:
        - Encoder output must be 5D: (batch, seq_len, height, width, channels)
        - Memory tensors must be 5D: (batch, seq_len, height, width, channels)
        - Memory mask must be 2D: (batch, seq_len)
        
    Post-conditions:
        - Output has same shape as encoder input
        - Attention weights respect memory masking
        - Spatial-temporal attention patterns are preserved
    """
    
    def __init__(
        self,
        config: Union[CoMemoryAttentionConfig, None] = None,
        *,
        query_shape: Union[Tuple[int, int, int, int], None] = None,
        memory_shape: Union[Tuple[int, int, int, int], None] = None,
        key_dim: int = 128,
        value_dim: int = 256,
        use_norm: bool = False,
        attention_drop_rate: float = 0.0,
        use_positional_embedding: bool = True,
        use_key_positional_embedding: bool = True,
        attention_heads: int = 8,
        use_qkv_embedding: bool = False
    ) -> None:
        """
        Initialize CoMemoryAttentionModule.
        
        Args:
            config: Configuration dataclass. If provided, other parameters are ignored.
            query_shape: Shape of query tensor. Used only if config is None.
            memory_shape: Shape of memory tensor. Used only if config is None.
            key_dim: Dimension for key projections. Used only if config is None.
            value_dim: Dimension for value projections. Used only if config is None.
            use_norm: Whether to apply layer normalization. Used only if config is None.
            attention_drop_rate: Dropout rate for attention. Used only if config is None.
            use_positional_embedding: Whether to use positional embeddings. Used only if config is None.
            use_key_positional_embedding: Whether to use key positional embeddings. Used only if config is None.
            attention_heads: Number of attention heads. Used only if config is None.
            use_qkv_embedding: Whether to use QKV projections. Used only if config is None.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__()
        
        # Create configuration from parameters or use provided config
        if config is None:
            if any(param is None for param in [query_shape, memory_shape]):
                raise ValueError(
                    "Either config must be provided, or both query_shape and memory_shape "
                    "must be specified"
                )
            
            self.config = CoMemoryAttentionConfig(
                query_shape=query_shape,  # type: ignore
                memory_shape=memory_shape,  # type: ignore
                key_dim=key_dim,
                value_dim=value_dim,
                use_norm=use_norm,
                attention_drop_rate=attention_drop_rate,
                use_positional_embedding=use_positional_embedding,
                use_key_positional_embedding=use_key_positional_embedding,
                attention_heads=attention_heads,
                use_qkv_embedding=use_qkv_embedding
            )
        else:
            self.config = config
        
        # Store individual attributes for JIT compatibility
        self.query_height = self.config.query_height
        self.query_width = self.config.query_width
        self.memory_height = self.config.memory_height
        self.memory_width = self.config.memory_width
        self.query_seq_len = self.config.query_seq_len
        self.memory_seq_len = self.config.memory_seq_len
        self.key_dim = self.config.key_dim
        self.value_dim = self.config.value_dim
        self.use_norm = self.config.use_norm
        self.attention_drop_rate = self.config.attention_drop_rate
        self.use_positional_embedding = self.config.use_positional_embedding
        self.use_key_positional_embedding = self.config.use_key_positional_embedding
        self.attention_heads = self.config.attention_heads
        self.use_qkv_embedding = self.config.use_qkv_embedding
        
        # Create normalization layers
        if self.use_norm:
            self.query_norm = nn.LayerNorm(self.config.query_channels, eps=1e-3)
            self.memory_norm = nn.LayerNorm(self.config.memory_channels, eps=1e-3)
        else:
            self.query_norm = nn.Identity()
            self.memory_norm = nn.Identity()

        # Create multi-head attention module
        self.att = MultiHeadAttention(
            query_shape=self.config.query_shape,
            key_shape=self.config.memory_shape,
            heads=self.attention_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim,
            attention_drop_rate=self.attention_drop_rate,
            use_positional_embedding=self.use_positional_embedding,
            use_key_positional_embedding=self.use_key_positional_embedding,
            use_query_embedding=self.use_qkv_embedding,
            use_key_embedding=self.use_qkv_embedding,
            use_value_embedding=self.use_qkv_embedding
        )

    def forward(
        self, 
        encoder_output: torch.Tensor, 
        memory_key: torch.Tensor, 
        memory_value: torch.Tensor, 
        memory_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply co-memory attention to input tensors.
        
        Args:
            encoder_output: Query tensor with shape (batch, seq_len, height, width, channels).
            memory_key: Key tensor with shape (batch, seq_len, height, width, channels).
            memory_value: Value tensor with shape (batch, seq_len, height, width, channels).
            memory_mask: Attention mask with shape (batch, seq_len).
            
        Returns:
            Attention output with same shape as encoder_output.
            
        Pre-conditions:
            - All input tensors must be on same device
            - encoder_output.dim() == memory_key.dim() == memory_value.dim() == 5
            - memory_mask.dim() == 2
            - Spatial dimensions must match configuration
            
        Post-conditions:
            - Output has same shape as encoder_output
            - Attention weights respect memory masking
            - Gradients flow properly through attention mechanism
            
        Raises:
            RuntimeError: If input tensors have incompatible shapes.
        """
        if encoder_output.dim() != 5:
            raise RuntimeError(
                f"Expected 5D encoder_output tensor (batch, seq_len, height, width, channels), "
                f"got {encoder_output.dim()}D tensor with shape {encoder_output.shape}"
            )
        
        if memory_key.dim() != 5:
            raise RuntimeError(
                f"Expected 5D memory_key tensor (batch, seq_len, height, width, channels), "
                f"got {memory_key.dim()}D tensor with shape {memory_key.shape}"
            )
        
        if memory_value.dim() != 5:
            raise RuntimeError(
                f"Expected 5D memory_value tensor (batch, seq_len, height, width, channels), "
                f"got {memory_value.dim()}D tensor with shape {memory_value.shape}"
            )

        query_encoder = encoder_output

        # Apply normalization
        query_encoder = self.query_norm(query_encoder)
        memory_key = self.memory_norm(memory_key)
        memory_value = self.memory_norm(memory_value)

        # Create attention mask
        mh_mask = create_encoder_memory_mask(
            memory_mask,
            self.query_height * self.query_width * self.query_seq_len,
            self.memory_height * self.memory_width * self.memory_seq_len,
            self.query_seq_len,
            self.memory_seq_len,
            self.attention_heads
        )

        # Reshape tensors for attention computation
        query_encoder = query_encoder.view(
            -1,
            self.query_seq_len * self.query_height * self.query_width,
            query_encoder.shape[-1])
        memory_key = memory_key.view(
            -1,
            self.memory_height * self.memory_width * self.memory_seq_len,
            memory_key.shape[-1])
        memory_value = memory_value.view(
            -1,
            self.memory_height * self.memory_width * self.memory_seq_len,
            memory_value.shape[-1])

        # Apply attention
        att_out = self.att(query_encoder, memory_key, memory_value, mask=mh_mask)

        # Reshape back to original format
        att_out = att_out.view(
            -1,
            self.query_seq_len,
            self.query_height,
            self.query_width,
            att_out.shape[-1])

        return att_out
    
    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return (
            f"query_shape={self.config.query_shape}, "
            f"memory_shape={self.config.memory_shape}, "
            f"key_dim={self.config.key_dim}, "
            f"value_dim={self.config.value_dim}, "
            f"attention_heads={self.config.attention_heads}"
        )


def create_encoder_memory_mask(
        mask,
        encoder_seq_length: int,
        key_seq_length: int,
        query_frame_num: int,
        key_frame_num: int,
        num_heads: int) -> torch.Tensor:

    # Assuming mask is a tensor of shape [batch_size, seq_len]
    mask = mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1,  seq_len, 1]
    mask = torch.tile(mask, [1, encoder_seq_length // query_frame_num, 1,
                                 key_seq_length // key_frame_num])  # [batch_size, encoder_seq_length, seq_len]
    mask = mask.view(mask.shape[0], encoder_seq_length, key_seq_length)  # [batch_size, encoder_seq_length, key_seq_length]

    mask = mask.unsqueeze(1)  # [batch_size, 1,  encoder_seq_length, key_seq_length]
    mask = torch.tile(mask, [1, num_heads, 1, 1])  # [batch_size, num_heads, encoder_seq_length, key_seq_length]
    return mask


def load_coattention_module_weights(keras_coattention_module, pytorch_coattention_module):
    # Load weights from Keras to PyTorch
   # Load query_norm
    if keras_coattention_module.use_norm:
        pytorch_coattention_module.query_norm.weight.data = keras_coattention_module.query_norm.get_weights()[0]
        pytorch_coattention_module.query_norm.bias.data = keras_coattention_module.query_norm.get_weights()[1]
    # load memory norm
    if keras_coattention_module.use_norm:
        pytorch_coattention_module.memory_norm.weight.data = keras_coattention_module.memory_norm.get_weights()[0]
        pytorch_coattention_module.memory_norm.bias.data = keras_coattention_module.memory_norm.get_weights()[1]

    #Load attention
    load_mha_positional_layer_weights(keras_coattention_module.att, pytorch_coattention_module.att)


def load_coattention_weights(keras_coattention, pytorch_coattention):

    load_coattention_module_weights(
        keras_coattention.memory_attention_module,
        pytorch_coattention.memory_attention_module)

    load_layer_norm_weights(keras_coattention.layer_norm1, pytorch_coattention.layer_norm1)
    for i in range(len(keras_coattention.query_norms)):

        load_layer_norm_weights(keras_coattention.query_norms[i], pytorch_coattention.query_norms[i])
        load_linear_weights(keras_coattention.query_denses[i], pytorch_coattention.query_denses[i])

    load_linear_weights(keras_coattention.key_dense, pytorch_coattention.key_dense)
    load_linear_weights(keras_coattention.value_dense, pytorch_coattention.value_dense)
    load_linear_weights(keras_coattention.out_dense, pytorch_coattention.out_dense)

    load_layer_norm_weights(keras_coattention.layer_norm2, pytorch_coattention.layer_norm2)

    load_linear_weights(keras_coattention.mlp1, pytorch_coattention.mlp1)
    load_linear_weights(keras_coattention.mlp2, pytorch_coattention.mlp2)

class CoAttentionModule(nn.Module):
    """
    PyTorch implementation of sequential co-attention module.
    
    This module implements sequential co-attention by processing memory frames
    sequentially, applying attention between queries and each memory frame
    independently. The module includes MLP processing and residual connections
    following the MVit2 pattern.
    
    Examples:
        >>> # Basic usage with memory attention module
        >>> memory_module = CoMemoryAttentionModule(
        ...     query_shape=(1, 16, 16, 128),
        ...     memory_shape=(1, 16, 16, 128)
        ... )
        >>> config = CoAttentionConfig(
        ...     query_shape=(1, 1, 16, 16, 128),
        ...     memory_shape=(1, 5, 16, 16, 128)
        ... )
        >>> module = CoAttentionModule(memory_module, config)
        >>> 
        >>> # Direct parameter specification
        >>> module = CoAttentionModule(
        ...     memory_module,
        ...     query_shape=(1, 1, 8, 8, 64),
        ...     memory_shape=(1, 10, 8, 8, 64),
        ...     key_dim=64,
        ...     hidden_dim=256
        ... )
    
    Attributes:
        config: Configuration dataclass containing all module parameters.
        memory_attention_module: Attention module for memory processing.
        
    Pre-conditions:
        - Query sequence must be 5D: (batch, seq_len, height, width, channels)
        - Memory bank must be 5D: (batch, frames, height, width, channels)
        - Mask must be 2D: (batch, frames)
        
    Post-conditions:
        - Output has same shape as query sequence
        - Sequential attention processing preserves temporal order
        - Residual connections and normalization are applied
    """
    
    def __init__(
        self,
        memory_attention_module: CoMemoryAttentionModule,
        config: Union[CoAttentionConfig, None] = None,
        *,
        query_shape: Union[Tuple[int, int, int, int, int], None] = None,
        memory_shape: Union[Tuple[int, int, int, int, int], None] = None,
        key_dim: int = 128,
        value_dim: int = 128,
        hidden_dim: int = 512,
        layer_norm_eps: float = 1e-3
    ) -> None:
        """
        Initialize CoAttentionModule.
        
        Args:
            memory_attention_module: Attention module for processing memory frames.
            config: Configuration dataclass. If provided, other parameters are ignored.
            query_shape: Shape of query tensor. Used only if config is None.
            memory_shape: Shape of memory tensor. Used only if config is None.
            key_dim: Dimension for key projections. Used only if config is None.
            value_dim: Dimension for value projections. Used only if config is None.
            hidden_dim: Hidden dimension for MLP. Used only if config is None.
            layer_norm_eps: Epsilon for layer normalization. Used only if config is None.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__()
        
        self.memory_attention_module = memory_attention_module
        
        # Create configuration from parameters or use provided config
        if config is None:
            if any(param is None for param in [query_shape, memory_shape]):
                raise ValueError(
                    "Either config must be provided, or both query_shape and memory_shape "
                    "must be specified"
                )
            
            self.config = CoAttentionConfig(
                query_shape=query_shape,  # type: ignore
                memory_shape=memory_shape,  # type: ignore
                key_dim=key_dim,
                value_dim=value_dim,
                hidden_dim=hidden_dim,
                layer_norm_eps=layer_norm_eps
            )
        else:
            self.config = config
        
        # Store individual attributes for JIT compatibility
        self.memory_frames = self.config.memory_frames
        self.key_dim = self.config.key_dim
        self.value_dim = self.config.value_dim
        self.hidden_dim = self.config.hidden_dim
        self.layer_norm_eps = self.config.layer_norm_eps
        self.query_channels = self.config.query_channels
        self.memory_channels = self.config.memory_channels
        
        # Create normalization layers
        self.layer_norm1 = nn.LayerNorm(self.memory_channels, eps=self.layer_norm_eps)

        # Create query processing layers for each frame
        self.query_norms = nn.ModuleList()
        self.query_denses = nn.ModuleList()

        # First frame uses original query channels
        self.query_norms.append(nn.LayerNorm(self.query_channels, eps=self.layer_norm_eps))
        self.query_denses.append(nn.Linear(self.query_channels, self.key_dim, bias=False))
        
        # Subsequent frames use key_dim
        for i in range(1, self.memory_frames):
            self.query_norms.append(nn.LayerNorm(self.key_dim, eps=self.layer_norm_eps))
            self.query_denses.append(nn.Linear(self.key_dim, self.key_dim, bias=False))

        # Create memory processing layers
        self.key_dense = nn.Linear(self.memory_channels, self.key_dim, bias=False)
        self.value_dense = nn.Linear(self.memory_channels, self.value_dim, bias=False)

        # Create output layers
        self.out_dense = nn.Linear(self.key_dim, self.query_channels, bias=False)
        self.layer_norm2 = nn.LayerNorm(self.query_channels, eps=self.layer_norm_eps)

        # Create MLP layers
        self.mlp1 = nn.Linear(self.query_channels, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, self.query_channels)

    def forward(
        self, 
        query_sequence: torch.Tensor, 
        memory_bank_sequence: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply sequential co-attention to input tensors.
        
        Args:
            query_sequence: Query tensor with shape (batch, seq_len, height, width, channels).
            memory_bank_sequence: Memory tensor with shape (batch, frames, height, width, channels).
            mask: Frame mask with shape (batch, frames).
            
        Returns:
            Processed query sequence with same shape as input.
            
        Pre-conditions:
            - All input tensors must be on same device
            - query_sequence.dim() == memory_bank_sequence.dim() == 5
            - mask.dim() == 2
            - Number of frames must match configuration
            
        Post-conditions:
            - Output has same shape as query_sequence
            - Sequential processing preserves temporal order
            - Residual connections and MLP are applied
            
        Raises:
            RuntimeError: If input tensors have incompatible shapes.
        """
        if query_sequence.dim() != 5:
            raise RuntimeError(
                f"Expected 5D query_sequence tensor (batch, seq_len, height, width, channels), "
                f"got {query_sequence.dim()}D tensor with shape {query_sequence.shape}"
            )
        
        if memory_bank_sequence.dim() != 5:
            raise RuntimeError(
                f"Expected 5D memory_bank_sequence tensor (batch, frames, height, width, channels), "
                f"got {memory_bank_sequence.dim()}D tensor with shape {memory_bank_sequence.shape}"
            )
        
        if mask.dim() != 2:
            raise RuntimeError(
                f"Expected 2D mask tensor (batch, frames), "
                f"got {mask.dim()}D tensor with shape {mask.shape}"
            )
        
        # Apply normalization to memory bank
        memory_bank_sequence = self.layer_norm1(memory_bank_sequence)

        # Split memory bank into individual frames
        memory_bank_frames = torch.split(memory_bank_sequence, 1, dim=1)

        # Process each frame sequentially
        attention_results = []
        att_input = query_sequence
        
        for i, (query_norm, query_dense) in enumerate(zip(self.query_norms, self.query_denses)):
            # Process memory frame
            key = self.key_dense(memory_bank_frames[i])
            value = self.value_dense(memory_bank_frames[i])
            
            # Process query
            att_input = query_norm(att_input)
            att_input = query_dense(att_input)
            
            # Apply attention
            attention_result = self.memory_attention_module(
                att_input, key, value, mask[:, i:i+1]
            )
            
            # Apply masking and residual connection
            attention_result *= mask[:, i:i+1, None, None, None]
            attention_result = attention_result + att_input
            attention_results.append(attention_result)
            att_input = attention_result

        # Aggregate results (average over frames)
        att_out = torch.stack(attention_results, dim=1).mean(dim=1)
        
        # Apply output projection and residual connection
        att_out = self.out_dense(att_out)
        att_out = att_out + query_sequence

        # Apply normalization and MLP
        att_out = self.layer_norm2(att_out)
        att_out = F.hardswish(self.mlp1(att_out))
        att_out = self.mlp2(att_out)

        return att_out
    
    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return (
            f"query_shape={self.config.query_shape}, "
            f"memory_shape={self.config.memory_shape}, "
            f"key_dim={self.config.key_dim}, "
            f"value_dim={self.config.value_dim}, "
            f"hidden_dim={self.config.hidden_dim}"
        )