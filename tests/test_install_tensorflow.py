import pytest
import importlib

def test_import_tensorflow():
    try:
        tensorflow = importlib.import_module('tensorflow')
        assert tensorflow is not None
    except ImportError:
        pytest.fail("TensorFlow is not installed")

def test_import_efficientvit_keras():
    try:
        efficientvit_keras = importlib.import_module('src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_keras')
        assert efficientvit_keras is not None
    except ImportError:
        pytest.fail("EfficientViT Keras implementation is not installed")

if __name__ == "__main__":
    pytest.main([__file__])