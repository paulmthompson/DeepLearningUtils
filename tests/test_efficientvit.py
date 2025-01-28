import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
import keras
from Layers.Backbones.Efficientvit.efficientvit_keras import EfficientViT_B as KerasEfficientViT
from Layers.Backbones.Efficientvit.efficientvit_pytorch import EfficientViT_B as PyTorchEfficientViT
from utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

"""
@pytest.fixture
def keras_model():
    return KerasEfficientViT()

@pytest.fixture
def pytorch_model():
    return PyTorchEfficientViT().cpu()
"""


def test_model_conversion(

):
    device = torch.device('cpu')

    #block_types=["conv", "conv", "conv", "conv"]
    use_norm = True
    block_types = ["conv", "conv", "transform", "transform"],

    keras_model = KerasEfficientViT(
        block_types=block_types,
        use_norm=use_norm,
    ).to(device)

    pytorch_model = PyTorchEfficientViT(
        block_types=block_types,
        use_norm=use_norm,
    ).to(device)
    # Generate random input data
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    input_tensor = torch.tensor(input_data).permute(0, 3, 1, 2).to(device)  # Convert to PyTorch format

    # Get Keras model output
    keras_output = keras_model.predict(input_data)

    # Load weights from Keras model to PyTorch model
    load_keras_weights_to_pytorch_by_name(keras_model, pytorch_model)

    # Get PyTorch model output
    # Put in eval mode
    pytorch_model.eval()
    pytorch_output = pytorch_model(input_tensor)[0].cpu().detach().numpy().transpose(0, 2, 3, 1)

    # Compare outputs
    np.testing.assert_allclose(keras_output, pytorch_output, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])