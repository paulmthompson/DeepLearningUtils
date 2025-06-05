"""
This implements a blur layer to be used with max pooling or convolutions for anti-aliasing.

Reference:
Zhang, R., 2019. Making Convolutional Networks Shift-Invariant Again. 
https://doi.org/10.48550/arXiv.1904.11486

Github repository for original implementation:
https://github.com/adobe/antialiased-cnns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

import keras
import numpy as np


@dataclass(frozen=True)
class Blur2DConfig:
    """Configuration for Blur2D layer.
    
    Attributes:
        kernel_size: Size of the blur kernel. Must be > 0.
        stride: Stride for the convolution. Must be > 0.
        kernel_type: Type of kernel to use for blurring.
        padding: Padding mode - either 'same' or 'valid'.
    """
    kernel_size: int = 5
    stride: int = 2
    kernel_type: Literal["Rect", "Triangle", "Binomial"] = "Rect"
    padding: str = "same"
    
    def __post_init__(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
            TypeError: If parameters have wrong types.
        """
        if not isinstance(self.kernel_size, int) or self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive integer, got {self.kernel_size}")
            
        if not isinstance(self.stride, int) or self.stride <= 0:
            raise ValueError(f"stride must be a positive integer, got {self.stride}")
            
        if self.kernel_type == "Triangle" and self.kernel_size <= 2:
            raise ValueError("Triangle kernel requires kernel_size > 2")
            
        if self.kernel_type == "Binomial" and self.kernel_size != 5:
            raise ValueError("Binomial kernel only supports kernel_size = 5")
            
        if self.padding not in ["same", "valid"]:
            raise ValueError(f"padding must be 'same' or 'valid', got '{self.padding}'")


class Blur2D(keras.layers.Layer):
    """Anti-aliasing blur layer for Keras.
    
    This layer applies blur filtering to reduce aliasing artifacts when downsampling.
    Supports rectangular, triangular, and binomial kernels.
    
    Args:
        config: Configuration object containing all layer parameters.
        **kwargs: Additional keyword arguments passed to parent Layer.
        
    Raises:
        ValueError: If configuration parameters are invalid.
        
    Example:
        >>> config = Blur2DConfig(kernel_size=3, stride=2, kernel_type="Triangle")
        >>> blur_layer = Blur2D(config)
        >>> x = keras.random.uniform((1, 224, 224, 64))
        >>> y = blur_layer(x)  # Shape: (1, 112, 112, 64)
    """
    
    def __init__(self, config: Blur2DConfig, **kwargs: Any) -> None:
        """Initialize Blur2D layer.
        
        Pre-conditions:
            - config must be a valid Blur2DConfig instance
            
        Post-conditions:
            - Layer is ready for forward pass
            - Kernel template is properly normalized
        """
        super().__init__(**kwargs)
        
        if not isinstance(config, Blur2DConfig):
            raise TypeError(f"config must be Blur2DConfig, got {type(config)}")
            
        self.config = config
        self.kernel2d = self._create_kernel_template()
        
    def _create_kernel_template(self) -> keras.KerasTensor:
        """Create and normalize the blur kernel template.
        
        Returns:
            Normalized 2D kernel tensor with shape (kernel_size, kernel_size, 1, 1).
            
        Raises:
            ValueError: If kernel_type is unsupported.
        """
        if self.config.kernel_type == "Rect":
            kernel = keras.ops.ones((self.config.kernel_size, self.config.kernel_size, 1, 1))
            
        elif self.config.kernel_type == "Triangle":
            kernel = self._create_triangle_kernel()
            
        elif self.config.kernel_type == "Binomial":
            kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
            kernel_2d = np.outer(kernel_1d, kernel_1d)
            kernel = keras.ops.expand_dims(
                keras.ops.convert_to_tensor(kernel_2d), axis=(2, 3)
            )
            
        else:
            raise ValueError(f"Unsupported kernel_type: {self.config.kernel_type}")
            
        # Normalize kernel
        kernel = keras.ops.cast(kernel, keras.backend.floatx())
        return keras.ops.divide(kernel, keras.ops.sum(kernel))
    
    def _create_triangle_kernel(self) -> keras.KerasTensor:
        """Create triangular kernel.
        
        Returns:
            Triangular kernel tensor with shape (kernel_size, kernel_size, 1, 1).
        """
        if self.config.kernel_size % 2 == 0:
            kernel_base = np.arange(1, (self.config.kernel_size + 1) // 2 + 1)
            kernel_base = np.concatenate([kernel_base, np.flip(kernel_base)])
        else:
            kernel_base = np.arange(1, (self.config.kernel_size + 1) // 2 + 1)
            kernel_base = np.concatenate([kernel_base, np.flip(kernel_base[:-1])])
            
        kernel_2d = np.outer(kernel_base, kernel_base).astype(np.float32)
        return keras.ops.expand_dims(
            keras.ops.convert_to_tensor(kernel_2d), axis=(2, 3)
        )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build layer by creating the full kernel for all input channels.
        
        Args:
            input_shape: Shape of input tensor (batch, height, width, channels).
            
        Pre-conditions:
            - input_shape must be 4D (batch, height, width, channels)
            
        Post-conditions:
            - self.kernel is created with proper shape for depthwise convolution
        """
        super().build(input_shape)
        
        if len(input_shape) != 4:
            raise ValueError(f"Input must be 4D tensor, got {len(input_shape)}D")
            
        # Create kernel for all input channels
        channels = input_shape[-1]
        self.kernel = keras.ops.tile(self.kernel2d, [1, 1, channels, 1])

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply blur filtering to input tensor.
        
        Args:
            inputs: Input tensor of shape (batch, height, width, channels).
            
        Returns:
            Blurred tensor with potentially reduced spatial dimensions.
            
        Pre-conditions:
            - inputs must be 4D tensor (batch, height, width, channels)
            - Layer must be built
            
        Post-conditions:
            - Output has same batch size and channels as input
            - Spatial dimensions may be reduced based on stride
        """
        if inputs.ndim != 4:
            raise ValueError(f"Input must be 4D tensor, got {inputs.ndim}D")
            
        # Preserve input dtype
        input_dtype = inputs.dtype
        inputs = keras.ops.cast(inputs, keras.backend.floatx())
        
        # Apply depthwise convolution for blurring
        blurred = keras.ops.depthwise_conv(
            inputs,
            self.kernel,
            strides=(self.config.stride, self.config.stride),
            padding=self.config.padding,
        )
        
        # Restore original dtype
        blurred = keras.ops.cast(blurred, input_dtype)
        return blurred

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.config.kernel_size,
            "stride": self.config.stride,
            "kernel_type": self.config.kernel_type,
            "padding": self.config.padding,
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Blur2D:
        """Create layer from configuration dictionary.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Blur2D layer instance.
        """
        # Extract blur-specific config
        blur_config = Blur2DConfig(
            kernel_size=config.pop("kernel_size", 5),
            stride=config.pop("stride", 2),
            kernel_type=config.pop("kernel_type", "Rect"),
            padding=config.pop("padding", "same"),
        )
        
        # Pass remaining config to parent
        return cls(blur_config, **config)


# Convenience constructor for backward compatibility
def create_blur2d(kernel_size: int = 5, 
                  stride: int = 2, 
                  kernel_type: Literal["Rect", "Triangle", "Binomial"] = "Rect",
                  padding: str = "same") -> Blur2D:
    """Create Blur2D layer with specified parameters.
    
    Args:
        kernel_size: Size of the blur kernel.
        stride: Stride for the convolution.
        kernel_type: Type of kernel to use.
        padding: Padding mode.
        
    Returns:
        Configured Blur2D layer.
    """
    config = Blur2DConfig(
        kernel_size=kernel_size,
        stride=stride, 
        kernel_type=kernel_type,
        padding=padding
    )
    return Blur2D(config)
