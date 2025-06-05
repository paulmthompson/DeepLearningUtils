"""
Configuration dataclass for Blur2D layers.

This module provides a shared configuration structure for both Keras and PyTorch
implementations of the Blur2D anti-aliasing layer.
"""

from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True)
class Blur2DConfig:
    """
    Configuration for Blur2D anti-aliasing layer.
    
    This dataclass encapsulates all parameters needed to configure a Blur2D layer,
    providing validation and documentation of expected inputs.
    
    Attributes:
        kernel_size: Size of the blur kernel. Must be positive integer.
                    For Triangle kernels, must be > 2.
                    For Binomial kernels, must be exactly 5.
        stride: Stride for the blur operation. Must be positive integer.
        kernel_type: Type of blur kernel to use.
        padding: Padding strategy. Either "same" or integer value.
        
    Pre-conditions:
        - kernel_size must be positive
        - stride must be positive  
        - For Triangle kernels: kernel_size > 2
        - For Binomial kernels: kernel_size == 5
        
    Post-conditions:
        - Configuration is immutable (frozen=True)
        - All values are validated in __post_init__
    """
    
    kernel_size: int = 5
    stride: int = 2
    kernel_type: Literal["Rect", "Triangle", "Binomial"] = "Rect"
    padding: Union[str, int] = "same"
    
    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.
        
        Raises:
            ValueError: If any parameter is invalid.
            TypeError: If parameters have incorrect types.
        """
        # Validate kernel_size
        if not isinstance(self.kernel_size, int):
            raise TypeError(f"kernel_size must be an integer, got {type(self.kernel_size)}")
        
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        
        # Validate stride
        if not isinstance(self.stride, int):
            raise TypeError(f"stride must be an integer, got {type(self.stride)}")
        
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        
        # Validate kernel_type specific constraints
        if self.kernel_type == "Triangle" and self.kernel_size <= 2:
            raise ValueError(
                f"Triangle kernel requires kernel_size > 2, got {self.kernel_size}"
            )
        
        if self.kernel_type == "Binomial" and self.kernel_size != 5:
            raise ValueError(
                f"Binomial kernel requires kernel_size == 5, got {self.kernel_size}"
            )
        
        # Validate padding
        if isinstance(self.padding, str):
            if self.padding not in ["same", "valid"]:
                raise ValueError(
                    f"String padding must be 'same' or 'valid', got '{self.padding}'"
                )
        elif isinstance(self.padding, int):
            if self.padding < 0:
                raise ValueError(f"Integer padding must be non-negative, got {self.padding}")
        else:
            raise TypeError(
                f"padding must be str or int, got {type(self.padding)}"
            ) 