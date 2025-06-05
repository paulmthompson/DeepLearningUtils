"""
PyTorch implementation of Blur2D anti-aliasing layer.

This implements a blur layer to be used with max pooling or convolutions for anti-aliasing.

Reference:
Zhang, R., 2019. Making Convolutional Networks Shift-Invariant Again. 
https://doi.org/10.48550/arXiv.1904.11486

Github repository for original implementation:
https://github.com/adobe/antialiased-cnns
"""

import math
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blur2d_config import Blur2DConfig
from .kernel_generators import generate_blur_kernel


class Blur2D(nn.Module):
    """
    PyTorch implementation of 2D blur layer for anti-aliasing.
    
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
        
    Pre-conditions:
        - Input tensor must be 4D: (batch_size, channels, height, width)
        - Input tensor must be on same device as layer
        
    Post-conditions:
        - Output tensor shape: (batch_size, channels, new_height, new_width)
        - new_height and new_width depend on stride and padding configuration
    """
    
    def __init__(
        self,
        config: Union[Blur2DConfig, None] = None,
        *,
        kernel_size: int = 5,
        stride: int = 2,
        kernel_type: str = "Rect",
        padding: Union[str, int] = "same"
    ) -> None:
        """
        Initialize Blur2D layer.
        
        Args:
            config: Configuration dataclass. If provided, other parameters are ignored.
            kernel_size: Size of blur kernel. Used only if config is None.
            stride: Stride for downsampling. Used only if config is None.
            kernel_type: Type of blur kernel. Used only if config is None.
            padding: Padding strategy. Used only if config is None.
            
        Raises:
            ValueError: If configuration parameters are invalid.
            TypeError: If parameters have incorrect types.
        """
        super().__init__()
        
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
        
        # Store configuration values as individual attributes for JIT compatibility
        self.kernel_size = self.config.kernel_size
        self.stride = self.config.stride
        self.kernel_type = self.config.kernel_type
        self.padding_mode = self.config.padding
        
        # Generate and register the kernel
        kernel_np = generate_blur_kernel(self.config)
        
        # Convert to PyTorch tensor and add batch/channel dimensions
        kernel_tensor = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel2d', kernel_tensor)
        
        # Cache padding mode for forward pass
        self.padding_same = (self.config.padding == "same")
        self.explicit_padding = 0 if self.padding_same else int(self.config.padding)
    
    def _calc_same_pad(self, i: int, k: int, s: int) -> int:
        """Calculate padding for 'same' mode. JIT-compatible version."""
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply blur and downsampling to input tensor.
        
        Args:
            x: Input tensor with shape (batch_size, channels, height, width).
            
        Returns:
            Blurred and downsampled tensor.
            
        Pre-conditions:
            - x must be 4D tensor
            - x must be on same device as layer
            
        Post-conditions:
            - Output has same number of dimensions as input
            - Output spatial dimensions are reduced by stride factor
            
        Raises:
            RuntimeError: If input tensor has wrong number of dimensions.
        """
        if x.dim() != 4:
            raise RuntimeError(
                f"Expected 4D input tensor (batch, channels, height, width), "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )
        
        batch_size, channels, height, width = x.size()
        
        # Expand kernel to match number of input channels for depthwise convolution
        kernel = self.kernel2d.repeat(channels, 1, 1, 1).to(x.device)

        # Handle padding
        if self.padding_same:
            pad_h = self._calc_same_pad(height, self.kernel_size, self.stride)
            pad_w = self._calc_same_pad(width, self.kernel_size, self.stride)

            if pad_h > 0 or pad_w > 0:
                x = F.pad(
                    x, 
                    [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                )
            padding = 0
        else:
            padding = self.explicit_padding

        # Apply depthwise convolution for blurring
        blurred = F.conv2d(
            x, 
            kernel, 
            stride=self.stride, 
            padding=padding, 
            groups=channels
        )
        
        return blurred
    
    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"kernel_type='{self.kernel_type}', "
            f"padding={self.padding_mode}"
        )