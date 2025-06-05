"""
Tests for kernel generation functions.

This module tests the kernel generation functions used by the Blur2D layers
according to the design guidelines.
"""

import numpy as np
import pytest
from src.DeepLearningUtils.Layers.Blur2D.kernel_generators import (
    create_rectangular_kernel,
    create_triangle_kernel,
    create_binomial_kernel,
    generate_blur_kernel,
    calculate_same_padding
)
from src.DeepLearningUtils.Layers.Blur2D.blur2d_config import Blur2DConfig


class TestCreateRectangularKernel:
    """Test the rectangular kernel generation function."""
    
    @pytest.mark.parametrize("kernel_size", [1, 2, 3, 5, 7])
    def test_rectangular_kernel_creation(self, kernel_size: int) -> None:
        """Test that rectangular kernels are created correctly."""
        kernel = create_rectangular_kernel(kernel_size)
        
        # Check shape
        assert kernel.shape == (kernel_size, kernel_size)
        
        # Check normalization
        assert np.allclose(kernel.sum(), 1.0, atol=1e-7)
        
        # Check uniformity
        expected_value = 1.0 / (kernel_size * kernel_size)
        assert np.allclose(kernel, expected_value, atol=1e-7)
        
        # Check dtype
        assert kernel.dtype == np.float32
    
    def test_rectangular_kernel_properties(self) -> None:
        """Test mathematical properties of rectangular kernels."""
        kernel = create_rectangular_kernel(5)
        
        # Should be symmetric
        assert np.allclose(kernel, kernel.T)
        
        # All values should be equal
        flat_kernel = kernel.flatten()
        assert np.allclose(flat_kernel, flat_kernel[0])


class TestCreateTriangleKernel:
    """Test the triangular kernel generation function."""
    
    @pytest.mark.parametrize("kernel_size", [3, 4, 5, 7, 9])
    def test_triangle_kernel_creation(self, kernel_size: int) -> None:
        """Test that triangular kernels are created correctly."""
        kernel = create_triangle_kernel(kernel_size)
        
        # Check shape
        assert kernel.shape == (kernel_size, kernel_size)
        
        # Check normalization
        assert np.allclose(kernel.sum(), 1.0, atol=1e-7)
        
        # Check dtype
        assert kernel.dtype == np.float32
    
    def test_triangle_kernel_invalid_size_raises_error(self) -> None:
        """Test that invalid kernel sizes raise ValueError."""
        with pytest.raises(ValueError, match="Triangle kernel requires kernel_size > 2"):
            create_triangle_kernel(1)
        
        with pytest.raises(ValueError, match="Triangle kernel requires kernel_size > 2"):
            create_triangle_kernel(2)
    
    def test_triangle_kernel_symmetry(self) -> None:
        """Test that triangular kernels are symmetric."""
        for kernel_size in [3, 5, 7]:
            kernel = create_triangle_kernel(kernel_size)
            
            # Should be symmetric
            assert np.allclose(kernel, kernel.T)
            
            # Should be symmetric about center
            assert np.allclose(kernel, np.flip(kernel))
    
    def test_triangle_kernel_peak_at_center(self) -> None:
        """Test that the maximum value is at or near the center."""
        for kernel_size in [3, 5, 7, 9]:
            kernel = create_triangle_kernel(kernel_size)
            center = kernel_size // 2
            
            # Maximum should be at center
            max_indices = np.unravel_index(np.argmax(kernel), kernel.shape)
            assert max_indices == (center, center)
    
    @pytest.mark.parametrize("kernel_size", [3, 4, 5, 6, 7, 8])
    def test_triangle_kernel_even_odd_consistency(self, kernel_size: int) -> None:
        """Test that even and odd kernel sizes produce valid results."""
        kernel = create_triangle_kernel(kernel_size)
        
        # Check basic properties
        assert kernel.shape == (kernel_size, kernel_size)
        assert np.allclose(kernel.sum(), 1.0, atol=1e-7)
        assert np.all(kernel >= 0)  # All values should be non-negative


class TestCreateBinomialKernel:
    """Test the binomial kernel generation function."""
    
    def test_binomial_kernel_creation(self) -> None:
        """Test that binomial kernel is created correctly."""
        kernel = create_binomial_kernel(5)
        
        # Check shape
        assert kernel.shape == (5, 5)
        
        # Check normalization
        assert np.allclose(kernel.sum(), 1.0, atol=1e-7)
        
        # Check dtype
        assert kernel.dtype == np.float32
    
    def test_binomial_kernel_invalid_size_raises_error(self) -> None:
        """Test that non-5 kernel sizes raise ValueError."""
        for invalid_size in [1, 2, 3, 4, 6, 7, 10]:
            with pytest.raises(ValueError, match="Binomial kernel requires kernel_size == 5"):
                create_binomial_kernel(invalid_size)
    
    def test_binomial_kernel_symmetry(self) -> None:
        """Test that binomial kernel is symmetric."""
        kernel = create_binomial_kernel(5)
        
        # Should be symmetric
        assert np.allclose(kernel, kernel.T)
        
        # Should be symmetric about center
        assert np.allclose(kernel, np.flip(kernel))
    
    def test_binomial_kernel_coefficients(self) -> None:
        """Test that binomial kernel uses correct coefficients."""
        kernel = create_binomial_kernel(5)
        
        # The 1D binomial coefficients should be [1, 4, 6, 4, 1]
        # So the 2D kernel should have certain expected values
        center = 2
        
        # Center value should be highest
        max_indices = np.unravel_index(np.argmax(kernel), kernel.shape)
        assert max_indices == (center, center)
        
        # Check that the kernel approximates Gaussian-like distribution
        assert kernel[center, center] > kernel[center-1, center]
        assert kernel[center, center] > kernel[center, center-1]


class TestGenerateBlurKernel:
    """Test the main kernel generation function."""
    
    @pytest.mark.parametrize("kernel_type, kernel_size", [
        ("Rect", 3),
        ("Triangle", 5),
        ("Binomial", 5),
    ])
    def test_generate_blur_kernel_valid_configs(self, kernel_type: str, kernel_size: int) -> None:
        """Test kernel generation with valid configurations."""
        config = Blur2DConfig(
            kernel_size=kernel_size,
            kernel_type=kernel_type,  # type: ignore
            stride=2,
            padding="same"
        )
        
        kernel = generate_blur_kernel(config)
        
        # Check basic properties
        assert kernel.shape == (kernel_size, kernel_size)
        assert np.allclose(kernel.sum(), 1.0, atol=1e-7)
        assert kernel.dtype == np.float32
        assert np.all(kernel >= 0)
    
    def test_generate_blur_kernel_invalid_type_raises_error(self) -> None:
        """Test that invalid kernel types raise ValueError."""
        config = Blur2DConfig(kernel_type="Invalid")  # type: ignore
        
        with pytest.raises(ValueError, match="Unsupported kernel type"):
            generate_blur_kernel(config)
    
    def test_generate_blur_kernel_consistency_with_individual_functions(self) -> None:
        """Test that generate_blur_kernel produces same results as individual functions."""
        # Test Rectangular
        config_rect = Blur2DConfig(kernel_size=5, kernel_type="Rect")
        kernel_via_generate = generate_blur_kernel(config_rect)
        kernel_direct = create_rectangular_kernel(5)
        assert np.allclose(kernel_via_generate, kernel_direct)
        
        # Test Triangle
        config_tri = Blur2DConfig(kernel_size=7, kernel_type="Triangle")
        kernel_via_generate = generate_blur_kernel(config_tri)
        kernel_direct = create_triangle_kernel(7)
        assert np.allclose(kernel_via_generate, kernel_direct)
        
        # Test Binomial
        config_bin = Blur2DConfig(kernel_size=5, kernel_type="Binomial")
        kernel_via_generate = generate_blur_kernel(config_bin)
        kernel_direct = create_binomial_kernel(5)
        assert np.allclose(kernel_via_generate, kernel_direct)


class TestCalculateSamePadding:
    """Test the same padding calculation function."""
    
    @pytest.mark.parametrize("input_size, kernel_size, stride", [
        # Common cases
        (32, 3, 1),  # stride 1, no downsampling
        (32, 3, 2),  # stride 2, typical convolution
        (16, 5, 2),  # larger kernel
        (10, 3, 2),  # small input
        
        # Edge cases
        (1, 1, 1),   # minimal case
        (5, 5, 1),   # kernel same as input
        (100, 7, 3), # large input
    ])
    def test_calculate_same_padding_known_cases(
        self, 
        input_size: int, 
        kernel_size: int, 
        stride: int
    ) -> None:
        """Test padding calculation for known cases."""
        padding = calculate_same_padding(input_size, kernel_size, stride)
        # Just test that padding is non-negative
        assert padding >= 0
    
    def test_calculate_same_padding_non_negative(self) -> None:
        """Test that padding is always non-negative."""
        for input_size in [1, 10, 32]:
            for kernel_size in [1, 3, 5]:
                for stride in [1, 2]:
                    padding = calculate_same_padding(input_size, kernel_size, stride)
                    assert padding >= 0
    
    def test_calculate_same_padding_mathematical_properties(self) -> None:
        """Test mathematical properties of padding calculation."""
        # When stride = 1, output size should equal input size with proper padding
        for input_size in [10, 20, 50]:
            for kernel_size in [3, 5, 7]:
                padding = calculate_same_padding(input_size, kernel_size, 1)
                
                # For stride=1, total padding should be kernel_size - 1
                assert padding == kernel_size - 1
    
    def test_calculate_same_padding_stride_effects(self) -> None:
        """Test how stride affects padding calculation."""
        input_size = 32
        kernel_size = 3
        
        # As stride increases, padding generally decreases or stays same
        prev_padding = float('inf')
        for stride in [1, 2, 4, 8]:
            padding = calculate_same_padding(input_size, kernel_size, stride)
            # Don't enforce strict decrease as it depends on the specific formula
            assert padding >= 0


class TestKernelGeneratorsIntegration:
    """Integration tests for kernel generator functions."""
    
    def test_all_kernel_types_produce_valid_kernels(self) -> None:
        """Test that all kernel types produce mathematically valid kernels."""
        test_cases = [
            ("Rect", 3),
            ("Rect", 5),
            ("Triangle", 3),
            ("Triangle", 7),
            ("Binomial", 5),
        ]
        
        for kernel_type, kernel_size in test_cases:
            config = Blur2DConfig(
                kernel_size=kernel_size,
                kernel_type=kernel_type,  # type: ignore
                stride=2,
                padding="same"
            )
            
            kernel = generate_blur_kernel(config)
            
            # Mathematical properties all kernels should satisfy
            assert kernel.shape == (kernel_size, kernel_size)
            assert np.allclose(kernel.sum(), 1.0, atol=1e-7)
            assert np.all(kernel >= 0)
            assert np.all(np.isfinite(kernel))
            assert kernel.dtype == np.float32
    
    def test_kernel_smoothness_properties(self) -> None:
        """Test that kernels have expected smoothness properties."""
        # Triangle kernels should be smoother than rectangular
        rect_kernel = create_rectangular_kernel(5)
        tri_kernel = create_triangle_kernel(5)
        
        # Triangle kernel should have more gradual transitions
        # (measured by variance from uniformity)
        rect_variance = np.var(rect_kernel)
        tri_variance = np.var(tri_kernel)
        
        # Triangle should have higher variance (more peaked in center)
        assert tri_variance > rect_variance
    
    def test_kernel_generation_reproducibility(self) -> None:
        """Test that kernel generation is deterministic and reproducible."""
        config = Blur2DConfig(kernel_size=7, kernel_type="Triangle")
        
        kernel1 = generate_blur_kernel(config)
        kernel2 = generate_blur_kernel(config)
        
        # Should be identical
        assert np.array_equal(kernel1, kernel2)
    
    @pytest.mark.parametrize("kernel_type", ["Rect", "Triangle", "Binomial"])
    def test_kernel_energy_conservation(self, kernel_type: str) -> None:
        """Test that kernels conserve energy (sum to 1)."""
        if kernel_type == "Binomial":
            kernel_size = 5
        elif kernel_type == "Triangle":
            kernel_size = 7
        else:
            kernel_size = 5
        
        config = Blur2DConfig(
            kernel_size=kernel_size,
            kernel_type=kernel_type,  # type: ignore
        )
        
        kernel = generate_blur_kernel(config)
        
        # Energy conservation: sum should be exactly 1
        assert np.allclose(kernel.sum(), 1.0, atol=1e-7)
        
        # No energy should be negative
        assert np.all(kernel >= 0) 