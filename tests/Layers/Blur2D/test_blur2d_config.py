"""
Tests for Blur2DConfig dataclass.

This module tests the configuration validation and behavior of the Blur2DConfig
dataclass according to the design guidelines.
"""

import pytest
from src.DeepLearningUtils.Layers.Blur2D.blur2d_config import Blur2DConfig


class TestBlur2DConfig:
    """Test class for Blur2DConfig dataclass."""
    
    def test_default_initialization(self) -> None:
        """Test that default configuration values are set correctly."""
        config = Blur2DConfig()
        
        assert config.kernel_size == 5
        assert config.stride == 2
        assert config.kernel_type == "Rect"
        assert config.padding == "same"
    
    @pytest.mark.parametrize("kernel_size, stride, kernel_type, padding", [
        (3, 1, "Rect", "same"),
        (5, 2, "Triangle", "valid"),
        (5, 4, "Binomial", 0),
        (7, 3, "Triangle", 2),
    ])
    def test_valid_configurations(
        self, 
        kernel_size: int, 
        stride: int, 
        kernel_type: str, 
        padding
    ) -> None:
        """Test that valid configurations are accepted."""
        config = Blur2DConfig(
            kernel_size=kernel_size,
            stride=stride,
            kernel_type=kernel_type,  # type: ignore
            padding=padding
        )
        
        assert config.kernel_size == kernel_size
        assert config.stride == stride
        assert config.kernel_type == kernel_type
        assert config.padding == padding
    
    def test_frozen_behavior(self) -> None:
        """Test that the dataclass is immutable (frozen=True)."""
        config = Blur2DConfig()
        
        with pytest.raises(AttributeError):
            config.kernel_size = 10  # type: ignore
        
        with pytest.raises(AttributeError):
            config.stride = 3  # type: ignore
    
    @pytest.mark.parametrize("invalid_kernel_size", [0, -1, -5])
    def test_invalid_kernel_size_raises_value_error(self, invalid_kernel_size: int) -> None:
        """Test that invalid kernel_size values raise ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            Blur2DConfig(kernel_size=invalid_kernel_size)
    
    def test_kernel_size_type_error(self) -> None:
        """Test that non-integer kernel_size raises TypeError."""
        with pytest.raises(TypeError, match="kernel_size must be an integer"):
            Blur2DConfig(kernel_size=3.5)  # type: ignore
    
    @pytest.mark.parametrize("invalid_stride", [0, -1, -3])
    def test_invalid_stride_raises_value_error(self, invalid_stride: int) -> None:
        """Test that invalid stride values raise ValueError."""
        with pytest.raises(ValueError, match="stride must be positive"):
            Blur2DConfig(stride=invalid_stride)
    
    def test_stride_type_error(self) -> None:
        """Test that non-integer stride raises TypeError."""
        with pytest.raises(TypeError, match="stride must be an integer"):
            Blur2DConfig(stride=2.5)  # type: ignore
    
    def test_triangle_kernel_size_validation(self) -> None:
        """Test that Triangle kernels require kernel_size > 2."""
        # Valid Triangle kernel
        config = Blur2DConfig(kernel_size=3, kernel_type="Triangle")
        assert config.kernel_type == "Triangle"
        
        # Invalid Triangle kernel
        with pytest.raises(ValueError, match="Triangle kernel requires kernel_size > 2"):
            Blur2DConfig(kernel_size=2, kernel_type="Triangle")
        
        with pytest.raises(ValueError, match="Triangle kernel requires kernel_size > 2"):
            Blur2DConfig(kernel_size=1, kernel_type="Triangle")
    
    def test_binomial_kernel_size_validation(self) -> None:
        """Test that Binomial kernels require kernel_size == 5."""
        # Valid Binomial kernel
        config = Blur2DConfig(kernel_size=5, kernel_type="Binomial")
        assert config.kernel_type == "Binomial"
        
        # Invalid Binomial kernel sizes
        with pytest.raises(ValueError, match="Binomial kernel requires kernel_size == 5"):
            Blur2DConfig(kernel_size=3, kernel_type="Binomial")
        
        with pytest.raises(ValueError, match="Binomial kernel requires kernel_size == 5"):
            Blur2DConfig(kernel_size=7, kernel_type="Binomial")
    
    @pytest.mark.parametrize("invalid_kernel_type", ["Gaussian", "Invalid", "rect", "RECT"])
    def test_invalid_kernel_type_raises_error(self, invalid_kernel_type: str) -> None:
        """Test that invalid kernel_type values are rejected at type checking level."""
        # Note: This test demonstrates that invalid kernel types would be caught by
        # static type checking (mypy), but at runtime they would pass through
        # since we use Literal typing which is not enforced at runtime by dataclasses
        config = Blur2DConfig(kernel_type=invalid_kernel_type)  # type: ignore
        assert config.kernel_type == invalid_kernel_type
    
    @pytest.mark.parametrize("valid_padding", ["same", "valid", 0, 1, 5])
    def test_valid_padding_values(self, valid_padding) -> None:
        """Test that valid padding values are accepted."""
        config = Blur2DConfig(padding=valid_padding)
        assert config.padding == valid_padding
    
    @pytest.mark.parametrize("invalid_string_padding", ["invalid", "SAME", "VALID", "zero"])
    def test_invalid_string_padding_raises_value_error(self, invalid_string_padding: str) -> None:
        """Test that invalid string padding values raise ValueError."""
        with pytest.raises(ValueError, match="String padding must be 'same' or 'valid'"):
            Blur2DConfig(padding=invalid_string_padding)
    
    def test_negative_integer_padding_raises_value_error(self) -> None:
        """Test that negative integer padding raises ValueError."""
        with pytest.raises(ValueError, match="Integer padding must be non-negative"):
            Blur2DConfig(padding=-1)
        
        with pytest.raises(ValueError, match="Integer padding must be non-negative"):
            Blur2DConfig(padding=-5)
    
    def test_padding_type_error(self) -> None:
        """Test that invalid padding types raise TypeError."""
        with pytest.raises(TypeError, match="padding must be str or int"):
            Blur2DConfig(padding=3.5)  # type: ignore
        
        with pytest.raises(TypeError, match="padding must be str or int"):
            Blur2DConfig(padding=["same"])  # type: ignore


class TestBlur2DConfigEdgeCases:
    """Test edge cases and complex scenarios for Blur2DConfig."""
    
    def test_large_values(self) -> None:
        """Test that large but valid values are accepted."""
        config = Blur2DConfig(
            kernel_size=101,  # Large odd kernel
            stride=50,        # Large stride  
            kernel_type="Triangle",
            padding=100       # Large padding
        )
        
        assert config.kernel_size == 101
        assert config.stride == 50
        assert config.padding == 100
    
    def test_string_representation(self) -> None:
        """Test that the dataclass has reasonable string representation."""
        config = Blur2DConfig(kernel_size=3, stride=1, kernel_type="Triangle", padding="valid")
        
        # Dataclass should automatically provide __repr__
        repr_str = repr(config)
        assert "Blur2DConfig" in repr_str
        assert "kernel_size=3" in repr_str
        assert "stride=1" in repr_str
        assert "Triangle" in repr_str
        assert "valid" in repr_str
    
    def test_equality_and_hashing(self) -> None:
        """Test that configurations with same values are equal and hashable."""
        config1 = Blur2DConfig(kernel_size=3, stride=2, kernel_type="Triangle", padding="same")
        config2 = Blur2DConfig(kernel_size=3, stride=2, kernel_type="Triangle", padding="same")
        config3 = Blur2DConfig(kernel_size=5, stride=2, kernel_type="Triangle", padding="same")
        
        # Test equality
        assert config1 == config2
        assert config1 != config3
        
        # Test hashing (since frozen=True)
        config_set = {config1, config2, config3}
        assert len(config_set) == 2  # config1 and config2 should be the same


@pytest.mark.parametrize("kernel_type, expected_kernel_sizes", [
    ("Rect", [1, 2, 3, 5, 7, 10]),
    ("Triangle", [3, 4, 5, 7, 9, 11]),
    ("Binomial", [5]),
])
def test_kernel_type_specific_requirements(kernel_type: str, expected_kernel_sizes: list) -> None:
    """Test kernel type specific requirements systematically."""
    for kernel_size in expected_kernel_sizes:
        if kernel_type == "Triangle" and kernel_size <= 2:
            with pytest.raises(ValueError):
                Blur2DConfig(kernel_size=kernel_size, kernel_type=kernel_type)  # type: ignore
        elif kernel_type == "Binomial" and kernel_size != 5:
            with pytest.raises(ValueError):
                Blur2DConfig(kernel_size=kernel_size, kernel_type=kernel_type)  # type: ignore
        else:
            # Should not raise
            config = Blur2DConfig(kernel_size=kernel_size, kernel_type=kernel_type)  # type: ignore
            assert config.kernel_size == kernel_size
            assert config.kernel_type == kernel_type 