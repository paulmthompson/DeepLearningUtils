"""
Keras implementation of 2D Rotary Positional Encoding layer.

This implements 2D rotary positional encoding for spatial attention mechanisms,
extending the standard rotary positional encoding to handle 2D spatial coordinates.

The layer computes and applies rotary positional embeddings to query tensors based on
their 2D spatial positions (height and width coordinates).

Reference implementations and theory can be found in:
- RoFormer: Enhanced Transformer with Rotary Position Embedding
- Various 2D extensions of rotary positional encoding

Adapted to follow company coding guidelines by Paul Thompson 2024
"""

from typing import Tuple, Union, Any, Optional
import keras
import math
import numpy as np
import tensorflow as tf

from .rotary_config import RotaryPositionalEncodingConfig


class RotaryPositionalEncoding2D(keras.layers.Layer):
    """
    Keras implementation of 2D Rotary Positional Encoding.
    
    This layer computes and applies 2D rotary positional embeddings to query tensors.
    It extends standard rotary positional encoding to handle 2D spatial dimensions
    by computing separate frequency components for height and width coordinates.
    
    The layer is designed to work with input tensors representing 2D spatial data
    where rotary embeddings provide position-aware representations.
    
    Examples:
        >>> # Basic usage with config
        >>> config = RotaryPositionalEncodingConfig(
        ...     dim=64, height=32, width=32
        ... )
        >>> layer = RotaryPositionalEncoding2D(config)
        >>> 
        >>> # Direct parameter specification
        >>> layer = RotaryPositionalEncoding2D(
        ...     dim=128, height=16, width=16, theta=10.0
        ... )
    
    Attributes:
        config: Configuration dataclass containing all layer parameters.
        
    Pre-conditions:
        - Input tensor q must have shape (batch_size, sequence_len, height*width, dimension)
        - dimension must match config.dim
        - height*width must match config.spatial_size
        
    Post-conditions:
        - Output tensor has same shape as input tensor
        - Rotary positional information is applied to the tensor
    """
    
    def __init__(
        self,
        config: Optional[RotaryPositionalEncodingConfig] = None,
        *,
        dim: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        theta: float = 10.0,
        rotate: bool = True,
        max_freq: int = 64,
        **kwargs: Any
    ) -> None:
        """
        Initialize RotaryPositionalEncoding2D layer.
        
        Args:
            config: Configuration dataclass. If provided, other parameters are ignored.
            dim: Dimensionality of embeddings. Used only if config is None.
            height: Height of 2D input. Used only if config is None.
            width: Width of 2D input. Used only if config is None.
            theta: Theta value for frequency calculation. Used only if config is None.
            rotate: Whether to use random rotations. Used only if config is None.
            max_freq: Maximum frequency. Used only if config is None.
            **kwargs: Additional arguments passed to parent Layer class.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__(**kwargs)
        
        # Create configuration from parameters or use provided config
        if config is None:
            if any(param is None for param in [dim, height, width]):
                raise ValueError(
                    "Either config must be provided, or all of dim, height, and width "
                    "must be specified"
                )
            
            self.config = RotaryPositionalEncodingConfig(
                dim=dim,           # type: ignore
                height=height,     # type: ignore
                width=width,       # type: ignore
                theta=theta,
                rotate=rotate,
                max_freq=max_freq
            )
        else:
            self.config = config
        
        # Store individual attributes for easier access
        self.dim = self.config.dim
        self.height = self.config.height
        self.width = self.config.width
        self.theta = self.config.theta
        self.rotate = self.config.rotate
        self.max_freq = self.config.max_freq
        
        # Initialize base frequency weights (matching PyTorch's approach)
        self.base_freqs = self.add_weight(
            name="base_freqs",
            shape=(self.dim // 4,),
            initializer=self._init_2d_freqs,
            trainable=True,
        )

    def _init_2d_freqs(self, shape: Tuple[int, ...], dtype: str = "float32") -> keras.KerasTensor:
        """
        Initialize base frequency components for rotary embeddings.
        
        This method computes the base frequencies that will be used to construct
        the full axial frequency pattern matching PyTorch's approach.
        
        Args:
            shape: Shape of the frequency tensor (dim//4,) for base frequencies.
            dtype: Data type for the frequencies.
            
        Returns:
            Base frequency tensor with shape (dim//4,) where frequencies are
            computed using the same formula as standard rotary embeddings.
            
        Pre-conditions:
            - shape[0] must equal self.dim//4
            - dtype must be a valid float type
            
        Post-conditions:
            - Output tensor has the specified shape and dtype
            - Frequencies are properly scaled according to theta parameter
        """
        base_freq_dim = shape[0]
        
        # Compute base frequencies using theta scaling - matching PyTorch exactly
        # This creates frequencies for dim//4 components, which get repeated/concatenated
        base_freqs = 1.0 / (self.theta ** (keras.ops.arange(0, base_freq_dim, dtype=dtype) * 2.0 / (base_freq_dim * 2)))
        
        return keras.ops.cast(base_freqs, dtype)

    def _init_spatial_coordinates(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Initialize spatial coordinate arrays for the 2D grid.
        
        Returns:
            Tuple of (t_x, t_y) coordinate tensors where:
            - t_x: x-coordinates for each spatial position
            - t_y: y-coordinates for each spatial position
            
        Pre-conditions:
            - self.height and self.width must be positive
            
        Post-conditions:
            - t_x has shape (height*width,) with x-coordinates
            - t_y has shape (height*width,) with y-coordinates
            - Coordinates are in range [0, width-1] and [0, height-1] respectively
        """
        # Create linear sequence for flattened spatial positions
        t = keras.ops.arange(self.config.spatial_size, dtype=keras.backend.floatx())
        
        # Convert to 2D coordinates
        t_x = keras.ops.cast(t % self.width, keras.backend.floatx())  # x coordinate
        t_y = keras.ops.cast(keras.ops.floor(t / self.width), keras.backend.floatx())  # y coordinate
        
        return t_x, t_y



    def _apply_rotary_embedding(
        self, 
        q: keras.KerasTensor, 
        t_x: keras.KerasTensor,
        t_y: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply 2D rotary embeddings using concatenated axial frequencies.
        
        This method exactly matches PyTorch's approach:
        - Creates concatenated axial frequencies for each spatial position
        - Applies rotary embedding to the full tensor using these frequencies
        
        Args:
            q: Query tensor with shape (batch_size, seq_len, height*width, dim).
            t_x: X-coordinates tensor with shape (height*width,).
            t_y: Y-coordinates tensor with shape (height*width,).
            
        Returns:
            Rotated query tensor with same shape as input.
            
        Pre-conditions:
            - q must be 4D tensor with last dimension equal to self.dim
            - t_x and t_y must have shape (height*width,)
            
        Post-conditions:
            - Output has same shape as input q
            - 2D rotary positional encoding has been applied using concatenated axial approach
        """
        batch_size, seq_len, hw, dim = keras.ops.shape(q)
        
        # Ensure float32 for complex operations
        q = keras.ops.cast(q, "float32")
        
        # Create concatenated axial frequencies matching PyTorch exactly
        # Each base frequency is repeated twice to form pairs
        base_freqs_repeated = keras.ops.repeat(self.base_freqs, 2)  # Shape: (dim//2,)
        
        # Create axial frequency matrix for all spatial positions
        # x-frequencies: coordinate * base_freqs for first half
        freqs_x = keras.ops.expand_dims(t_x, -1) @ keras.ops.expand_dims(base_freqs_repeated, 0)  # (hw, dim//2)
        
        # y-frequencies: coordinate * base_freqs for second half  
        freqs_y = keras.ops.expand_dims(t_y, -1) @ keras.ops.expand_dims(base_freqs_repeated, 0)  # (hw, dim//2)
        
        # Concatenate to form full frequency matrix (y-freqs first to match PyTorch)
        freqs_combined = keras.ops.concatenate([freqs_y, freqs_x], axis=-1)  # (hw, dim)
        freqs_combined = keras.ops.cast(freqs_combined, "float32")
        
        # Apply rotary embedding using the combined frequencies
        # Reshape q to separate real/imaginary pairs for rotation
        q_reshaped = keras.ops.reshape(q, (batch_size, seq_len, hw, dim // 2, 2))
        
        # Create complex representation  
        q_complex = tf.complex(q_reshaped[..., 0], q_reshaped[..., 1])
        
        # Convert frequencies to complex exponentials
        # freqs_combined has shape (hw, dim) where each value is a frequency
        # For rotary embeddings, consecutive pairs of features are rotated by the same frequency
        freqs_for_complex = freqs_combined[:, ::2]  # Take every other frequency: (hw, dim//2)
        freqs_complex = tf.complex(freqs_for_complex, keras.ops.zeros_like(freqs_for_complex))
        rotation_matrix = keras.ops.exp(1.0j * freqs_complex)
        
        # Apply rotation
        q_rotated = q_complex * rotation_matrix[None, None, :, :]
        
        # Extract real and imaginary parts
        q_rotated_real = keras.ops.real(q_rotated)
        q_rotated_imag = keras.ops.imag(q_rotated)
        
        # Recombine into original format
        q_rotated = keras.ops.stack([q_rotated_real, q_rotated_imag], axis=-1)
        q_rotated = keras.ops.reshape(q_rotated, (batch_size, seq_len, hw, dim))
        
        return q_rotated

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build layer by validating input shape.
        
        Args:
            input_shape: Shape of the input tensor.
            
        Pre-conditions:
            - input_shape must represent 4D tensor
            - Last dimension must match self.config.dim
            - Third dimension must match self.config.spatial_size
            
        Post-conditions:
            - Layer is built and ready for forward pass
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch_size, seq_len, height*width, dim), "
                f"got {len(input_shape)}D shape: {input_shape}"
            )
        
        if input_shape[-1] != self.config.dim:
            raise ValueError(
                f"Expected last dimension to be {self.config.dim}, "
                f"got {input_shape[-1]}"
            )
        
        if input_shape[-2] != self.config.spatial_size:
            raise ValueError(
                f"Expected spatial dimension to be {self.config.spatial_size} "
                f"(height {self.config.height} * width {self.config.width}), "
                f"got {input_shape[-2]}"
            )
        
        super().build(input_shape)

    def call(self, q: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply 2D rotary positional encoding to query tensor.
        
        Args:
            q: Query tensor with shape (batch_size, seq_len, height*width, dim).
            
        Returns:
            Query tensor with rotary positional encoding applied.
            
        Pre-conditions:
            - q must be 4D tensor
            - q.shape[-1] must equal self.config.dim
            - q.shape[-2] must equal self.config.spatial_size
            - Layer must be built
            
        Post-conditions:
            - Output has same shape as input
            - Rotary positional encoding has been applied based on 2D spatial coordinates
        """
        if keras.ops.ndim(q) != 4:
            raise ValueError(
                f"Expected 4D query tensor (batch_size, seq_len, height*width, dim), "
                f"got {keras.ops.ndim(q)}D tensor with shape {keras.ops.shape(q)}"
            )
        
        # Initialize spatial coordinates
        t_x, t_y = self._init_spatial_coordinates()
        
        # Apply 2D rotary embeddings
        q_rotated = self._apply_rotary_embedding(q, t_x, t_y)
        
        return q_rotated
    
    def get_config(self) -> dict[str, Any]:
        """
        Return layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            'dim': self.config.dim,
            'height': self.config.height,
            'width': self.config.width,
            'theta': self.config.theta,
            'rotate': self.config.rotate,
            'max_freq': self.config.max_freq,
        })
        return config
