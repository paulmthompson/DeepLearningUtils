"""
This implements a blur layer to be used with max pooling or convolutions for anti-aliasing.

Reference:
Zhang, R., 2019. Making Convolutional Networks Shift-Invariant Again. 
https://doi.org/10.48550/arXiv.1904.11486

Github repository for original implementation:
https://github.com/adobe/antialiased-cnns
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Blur2DConfig:
    """Configuration for Blur2D layer.
    
    Attributes:
        kernel_size: Size of the blur kernel. Must be > 0.
        stride: Stride for the convolution. Must be > 0.
        kernel_type: Type of kernel to use for blurring.
        padding: Padding mode - either 'same' or integer value.
    """
    kernel_size: int = 5
    stride: int = 2
    kernel_type: Literal["Rect", "Triangle", "Binomial"] = "Rect"
    padding: Union[str, int] = 'same'
    
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
            
        if isinstance(self.padding, str) and self.padding != 'same':
            raise ValueError(f"String padding must be 'same', got '{self.padding}'")
            
        if isinstance(self.padding, int) and self.padding < 0:
            raise ValueError(f"Integer padding must be >= 0, got {self.padding}")


class Blur2D(nn.Module):
    """Anti-aliasing blur layer for PyTorch.
    
    This layer applies blur filtering to reduce aliasing artifacts when downsampling.
    Supports rectangular, triangular, and binomial kernels.
    
    Args:
        config: Configuration object containing all layer parameters.
        
    Raises:
        ValueError: If configuration parameters are invalid.
        
    Example:
        >>> config = Blur2DConfig(kernel_size=3, stride=2, kernel_type="Triangle")
        >>> blur_layer = Blur2D(config)
        >>> x = torch.randn(1, 64, 224, 224)
        >>> y = blur_layer(x)  # Shape: (1, 64, 112, 112)
    """
    
    def __init__(self, config: Blur2DConfig) -> None:
        """Initialize Blur2D layer.
        
        Pre-conditions:
            - config must be a valid Blur2DConfig instance
            
        Post-conditions:
            - Layer is ready for forward pass
            - Kernel is properly normalized
        """
        super().__init__()
        
        if not isinstance(config, Blur2DConfig):
            raise TypeError(f"config must be Blur2DConfig, got {type(config)}")
            
        self.config = config
        # Store config values as individual attributes for TorchScript compatibility
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.kernel_type = config.kernel_type
        self.padding_same = config.padding == 'same'
        # Store padding as integer for TorchScript compatibility
        self.padding_int = 0 if config.padding == 'same' else int(config.padding)
        
        self.kernel2d = self._create_kernel()
        self.kernel2d = self.kernel2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
    def _create_kernel(self) -> torch.Tensor:
        """Create and normalize the blur kernel.
        
        Returns:
            Normalized 2D kernel tensor.
            
        Raises:
            ValueError: If kernel_type is unsupported.
        """
        if self.kernel_type == "Rect":
            kernel = torch.ones((self.kernel_size, self.kernel_size))
            
        elif self.kernel_type == "Triangle":
            kernel = self._create_triangle_kernel()
            
        elif self.kernel_type == "Binomial":
            kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
            kernel = torch.tensor(np.outer(kernel_1d, kernel_1d))
            
        else:
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")
            
        # Normalize kernel
        return kernel / kernel.sum()
    
    def _create_triangle_kernel(self) -> torch.Tensor:
        """Create triangular kernel.
        
        Returns:
            Triangular kernel tensor.
        """
        if self.kernel_size % 2 == 0:
            kernel_base = np.arange(1, (self.kernel_size + 1) // 2 + 1)
            kernel_base = np.concatenate([kernel_base, np.flip(kernel_base)])
        else:
            kernel_base = np.arange(1, (self.kernel_size + 1) // 2 + 1)
            kernel_base = np.concatenate([kernel_base, np.flip(kernel_base[:-1])])
            
        kernel_2d = np.outer(kernel_base, kernel_base).astype(np.float32)
        return torch.tensor(kernel_2d)

    def _calc_same_pad(self, input_size: int, kernel_size: int, stride: int) -> int:
        """Calculate padding for 'same' padding mode.
        
        Args:
            input_size: Size of input dimension.
            kernel_size: Size of kernel dimension.
            stride: Stride value.
            
        Returns:
            Required padding amount.
        """
        return max((math.ceil(input_size / stride) - 1) * stride + (kernel_size - 1) + 1 - input_size, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply blur filtering to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            
        Returns:
            Blurred tensor with potentially reduced spatial dimensions.
            
        Pre-conditions:
            - x must be 4D tensor (batch, channels, height, width)
            - x must be on same device as layer
            
        Post-conditions:
            - Output has same batch size and channels as input
            - Spatial dimensions may be reduced based on stride
        """
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D tensor, got {x.dim()}D")
            
        batch_size, channels, height, width = x.size()
        kernel = self.kernel2d.repeat(channels, 1, 1, 1).to(x.device)

        if self.padding_same:
            pad_h = self._calc_same_pad(height, self.kernel2d.shape[2], self.stride)
            pad_w = self._calc_same_pad(width, self.kernel2d.shape[3], self.stride)

            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            padding = 0
        else:
            padding = self.padding_int

        # Apply depthwise convolution for blurring
        blurred = F.conv2d(x, kernel, stride=self.stride, padding=padding, groups=channels)
        return blurred


# Convenience constructor for backward compatibility
def create_blur2d(kernel_size: int = 5, 
                  stride: int = 2, 
                  kernel_type: Literal["Rect", "Triangle", "Binomial"] = "Rect",
                  padding: Union[str, int] = 'same') -> Blur2D:
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