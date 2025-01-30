import pytest
import importlib

def test_import_torch():
    try:
        torch = importlib.import_module('torch')
        assert torch is not None
    except ImportError:
        pytest.fail("PyTorch is not installed")

def test_import_efficientvit_pytorch():
    try:
        efficientvit_pytorch = importlib.import_module('src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_pytorch')
        assert efficientvit_pytorch is not None
    except ImportError:
        pytest.fail("EfficientViT PyTorch implementation is not installed")

if __name__ == "__main__":
    pytest.main([__file__])