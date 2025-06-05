"""
Kernel generation functions for Blur2D layers.

This module provides functions to generate different types of blur kernels
used in anti-aliasing operations. These functions return numpy arrays that
can be converted to framework-specific tensors.
"""

import math
import numpy as np
from typing import Tuple

from .blur2d_config import Blur2DConfig


def create_rectangular_kernel(kernel_size: int) -> np.ndarray:
    """
    Create a rectangular (uniform) blur kernel.
    
    Args:
        kernel_size: Size of the square kernel.
        
    Returns:
        Normalized rectangular kernel as numpy array.
        
    Post-conditions:
        - Kernel is square with shape (kernel_size, kernel_size)
        - All elements are equal
        - Kernel sums to 1.0
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    return kernel / kernel.sum()


def create_triangle_kernel(kernel_size: int) -> np.ndarray:
    """
    Create a triangular blur kernel using linear interpolation.
    
    The kernel creates a pyramid-like shape with maximum value at center
    and linearly decreasing values toward edges.
    
    Args:
        kernel_size: Size of the square kernel. Must be > 2.
        
    Returns:
        Normalized triangular kernel as numpy array.
        
    Pre-conditions:
        - kernel_size > 2
        
    Post-conditions:
        - Kernel is square with shape (kernel_size, kernel_size)
        - Maximum value is at center
        - Values decrease linearly toward edges
        - Kernel sums to 1.0
        
    Raises:
        ValueError: If kernel_size <= 2.
    """
    if kernel_size <= 2:
        raise ValueError(f"Triangle kernel requires kernel_size > 2, got {kernel_size}")
    
    if kernel_size % 2 == 0:
        # Even kernel size
        kernel_base = np.arange(1, (kernel_size + 1) // 2 + 1, dtype=np.float32)
        kernel_base = np.concatenate([kernel_base, np.flip(kernel_base)])
    else:
        # Odd kernel size
        kernel_base = np.arange(1, (kernel_size + 1) // 2 + 1, dtype=np.float32)
        kernel_base = np.concatenate([kernel_base, np.flip(kernel_base[:-1])])
    
    kernel = np.outer(kernel_base, kernel_base)
    return kernel / kernel.sum()


def create_binomial_kernel(kernel_size: int) -> np.ndarray:
    """
    Create a binomial blur kernel.
    
    Uses the binomial coefficients [1, 4, 6, 4, 1] to create a 5x5 kernel
    that approximates a Gaussian distribution.
    
    Args:
        kernel_size: Size of the square kernel. Must be exactly 5.
        
    Returns:
        Normalized binomial kernel as numpy array.
        
    Pre-conditions:
        - kernel_size == 5
        
    Post-conditions:
        - Kernel is 5x5
        - Kernel approximates Gaussian distribution
        - Kernel sums to 1.0
        
    Raises:
        ValueError: If kernel_size != 5.
    """
    if kernel_size != 5:
        raise ValueError(f"Binomial kernel requires kernel_size == 5, got {kernel_size}")
    
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel / kernel.sum()


def generate_blur_kernel(config: Blur2DConfig) -> np.ndarray:
    """
    Generate a blur kernel based on the provided configuration.
    
    Args:
        config: Configuration specifying kernel type and parameters.
        
    Returns:
        Normalized blur kernel as numpy array.
        
    Pre-conditions:
        - config is valid (validated in Blur2DConfig.__post_init__)
        
    Post-conditions:
        - Kernel is square with shape (kernel_size, kernel_size)
        - Kernel sums to 1.0
        - Kernel type matches config.kernel_type
        
    Raises:
        ValueError: If kernel_type is not supported.
    """
    kernel_type = config.kernel_type
    kernel_size = config.kernel_size
    
    if kernel_type == "Rect":
        return create_rectangular_kernel(kernel_size)
    elif kernel_type == "Triangle":
        return create_triangle_kernel(kernel_size)
    elif kernel_type == "Binomial":
        return create_binomial_kernel(kernel_size)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")


def calculate_same_padding(input_size: int, kernel_size: int, stride: int) -> int:
    """
    Calculate padding needed for 'same' padding mode.
    
    This implements the same padding calculation used by most deep learning
    frameworks to maintain spatial dimensions after convolution.
    
    Args:
        input_size: Size of input dimension.
        kernel_size: Size of convolution kernel.
        stride: Stride of convolution.
        
    Returns:
        Total padding needed for this dimension.
        
    Pre-conditions:
        - All inputs are positive integers
        
    Post-conditions:
        - Returned padding is non-negative
    """
    return max((math.ceil(input_size / stride) - 1) * stride + (kernel_size - 1) + 1 - input_size, 0) 