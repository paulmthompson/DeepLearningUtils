"""
Keras implementation of Blur2D anti-aliasing layer.

This implements a blur layer to be used with max pooling or convolutions for anti-aliasing.

Reference:
Zhang, R., 2019. Making Convolutional Networks Shift-Invariant Again. 
https://doi.org/10.48550/arXiv.1904.11486

Github repository for original implementation:
https://github.com/adobe/antialiased-cnns
"""

from typing import Union, Tuple, Any
import keras
import numpy as np

from .blur2d_config import Blur2DConfig
from .kernel_generators import generate_blur_kernel


class Blur2D(keras.layers.Layer):
    """
    Keras implementation of 2D blur layer for anti-aliasing.
    
    This layer applies a blur kernel followed by downsampling to reduce aliasing
    artifacts in convolutional neural networks. The blur operation is implemented
    as a depthwise convolution.
    
    The layer supports three kernel types:
    - Rectangular: Uniform averaging kernel
    - Triangle: Linear interpolation kernel with pyramid shape
    - Binomial: Approximates Gaussian using binomial coefficients [1,4,6,4,1]
    
    Examples:
        >>> # Basic usage with default rectangular kernel
        >>> blur = Blur2D()
        >>> 
        >>> # Custom configuration
        >>> config = Blur2DConfig(kernel_size=3, stride=2, kernel_type="Triangle")
        >>> blur = Blur2D(config)
        >>> 
        >>> # Direct parameter specification
        >>> blur = Blur2D(kernel_size=5, kernel_type="Binomial")
    
    Attributes:
        config: Configuration dataclass containing all layer parameters.
        kernel: Convolution kernel used for blurring (set during build).
        
    Pre-conditions:
        - Input tensor must be 4D: (batch_size, height, width, channels)
        
    Post-conditions:
        - Output tensor shape: (batch_size, new_height, new_width, channels)
        - new_height and new_width depend on stride and padding configuration
    """
    
    def __init__(
        self,
        config: Union[Blur2DConfig, None] = None,
        *,
        kernel_size: int = 2,
        stride: int = 2,
        kernel_type: str = "Rect",
        padding: Union[str, int] = "same",
        **kwargs: Any
    ) -> None:
        """
        Initialize Blur2D layer.
        
        Args:
            config: Configuration dataclass. If provided, other parameters are ignored.
            kernel_size: Size of blur kernel. Used only if config is None.
            stride: Stride for downsampling. Used only if config is None.
            kernel_type: Type of blur kernel. Used only if config is None.
            padding: Padding strategy. Used only if config is None.
            **kwargs: Additional arguments passed to parent Layer class.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__(**kwargs)
        
        # Create configuration from parameters or use provided config
        if config is None:
            self.config = Blur2DConfig(
                kernel_size=kernel_size,
                stride=stride,
                kernel_type=kernel_type,  # type: ignore
                padding=padding
            )
        else:
            self.config = config
        
        # Generate the base kernel (will be expanded during build)
        kernel_np = generate_blur_kernel(self.config)
        
        # Convert to Keras tensor format (height, width, in_channels, out_channels)
        # For depthwise conv, we need shape (kernel_h, kernel_w, 1, 1)
        self.kernel2d = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(kernel_np, dtype=keras.backend.floatx()),
            axis=(2, 3)
        )
        
        # Convert padding to string format expected by Keras
        if isinstance(self.config.padding, int):
            if self.config.padding == 0:
                self.padding_str = "valid"
            else:
                # Keras doesn't support arbitrary integer padding in depthwise_conv
                # Fall back to "same" and let user handle custom padding externally
                self.padding_str = "same"
        else:
            self.padding_str = str(self.config.padding)
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build layer by expanding kernel to match input channels.
        
        Args:
            input_shape: Shape of input tensor (batch_size, height, width, channels).
            
        Pre-conditions:
            - input_shape must represent 4D tensor
            - Last dimension (channels) must be known
            
        Post-conditions:
            - self.kernel is created with proper shape for depthwise convolution
        """
        super().build(input_shape)
        
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D shape: {input_shape}"
            )
        
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Number of input channels must be known at build time")
        
        # Expand kernel to match number of input channels for depthwise convolution
        # Shape: (kernel_h, kernel_w, channels, 1)
        self.kernel = keras.ops.tile(self.kernel2d, [1, 1, channels, 1])
    
    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply blur and downsampling to input tensor.
        
        Args:
            inputs: Input tensor with shape (batch_size, height, width, channels).
            
        Returns:
            Blurred and downsampled tensor.
            
        Pre-conditions:
            - inputs must be 4D tensor
            - Layer must be built (self.kernel exists)
            
        Post-conditions:
            - Output has same number of dimensions as input
            - Output spatial dimensions are reduced by stride factor
        """
        # Store original dtype and convert to float for computation
        input_dtype = inputs.dtype
        inputs_float = keras.ops.cast(inputs, keras.backend.floatx())
        
        # Apply depthwise convolution for blurring
        blurred = keras.ops.depthwise_conv(
            inputs_float,
            self.kernel,
            strides=(self.config.stride, self.config.stride),
            padding=self.padding_str,
        )
        
        # Convert back to original dtype
        blurred = keras.ops.cast(blurred, input_dtype)
        return blurred
    
    def get_config(self) -> dict[str, Any]:
        """
        Return layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            'kernel_size': self.config.kernel_size,
            'stride': self.config.stride,
            'kernel_type': self.config.kernel_type,
            'padding': self.config.padding,
        })
        return config
    
    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute output shape given input shape.
        
        Args:
            input_shape: Shape of input tensor.
            
        Returns:
            Shape of output tensor.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")
        
        batch_size, height, width, channels = input_shape
        
        if self.padding_str == "same":
            # For "same" padding, output size is ceil(input_size / stride)
            if height is not None:
                new_height = (height + self.config.stride - 1) // self.config.stride
            else:
                new_height = None
                
            if width is not None:
                new_width = (width + self.config.stride - 1) // self.config.stride
            else:
                new_width = None
        else:
            # For "valid" padding, no padding is added
            if height is not None:
                new_height = (height - self.config.kernel_size + 1 + 
                             self.config.stride - 1) // self.config.stride
            else:
                new_height = None
                
            if width is not None:
                new_width = (width - self.config.kernel_size + 1 + 
                            self.config.stride - 1) // self.config.stride
            else:
                new_width = None
        
        return (batch_size, new_height, new_width, channels)
