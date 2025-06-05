"""
Configuration dataclass for RotaryPositionalEncoding2D layer.

This module provides a configuration structure for the Keras implementation
of the RotaryPositionalEncoding2D layer, following company coding guidelines.
"""

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass(frozen=True)
class RotaryPositionalEncodingConfig:
    """
    Configuration for RotaryPositionalEncoding2D layer.
    
    This dataclass encapsulates all parameters needed to configure a RotaryPositionalEncoding2D
    layer, providing validation and documentation of expected inputs.
    
    Attributes:
        dim: The dimensionality of the embeddings (must be even).
        height: The height of the 2D input.
        width: The width of the 2D input.
        theta: The theta value for frequency calculation.
        rotate: Whether to use random rotations.
        max_freq: Maximum frequency for calculations.
        
    Pre-conditions:
        - dim must be positive and even
        - height must be positive
        - width must be positive
        - theta must be positive
        - max_freq must be positive
        
    Post-conditions:
        - Configuration is immutable (frozen=True)
        - All values are validated in __post_init__
    """
    
    dim: int
    height: int
    width: int
    theta: float = 10.0
    rotate: bool = True
    max_freq: int = 64
    
    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.
        
        Raises:
            ValueError: If any parameter is invalid.
            TypeError: If parameters have incorrect types.
        """
        # Validate dim
        if not isinstance(self.dim, int):
            raise TypeError(f"dim must be an integer, got {type(self.dim)}")
        
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        
        if self.dim % 2 != 0:
            raise ValueError(f"dim must be even for rotary embeddings, got {self.dim}")
        
        # Validate height
        if not isinstance(self.height, int):
            raise TypeError(f"height must be an integer, got {type(self.height)}")
        
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        
        # Validate width
        if not isinstance(self.width, int):
            raise TypeError(f"width must be an integer, got {type(self.width)}")
        
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        
        # Validate theta
        if not isinstance(self.theta, (int, float)):
            raise TypeError(f"theta must be a number, got {type(self.theta)}")
        
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        
        # Validate rotate
        if not isinstance(self.rotate, bool):
            raise TypeError(f"rotate must be a boolean, got {type(self.rotate)}")
        
        # Validate max_freq
        if not isinstance(self.max_freq, int):
            raise TypeError(f"max_freq must be an integer, got {type(self.max_freq)}")
        
        if self.max_freq <= 0:
            raise ValueError(f"max_freq must be positive, got {self.max_freq}")
    
    @property
    def spatial_size(self) -> int:
        """Get total spatial size (height * width)."""
        return self.height * self.width
    
    @property
    def freq_dim(self) -> int:
        """Get frequency dimension (dim // 2)."""
        return self.dim // 2 