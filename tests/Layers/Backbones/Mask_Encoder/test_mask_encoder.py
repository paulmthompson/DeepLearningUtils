import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import keras
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_keras import create_mask_encoder
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_pytorch import MaskEncoder
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_pytorch import load_mask_encoder_weights

from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

@pytest.mark.parametrize("input_shape, output_channels, anti_aliasing", [
    ((128, 128, 3), 256, True),
    ((128, 128, 3), 256, False)
])
def test_mask_encoder(input_shape, output_channels, anti_aliasing):
    # Create Keras model
    keras_model = create_mask_encoder(input_shape, output_channels, anti_aliasing)
    keras_input = np.random.rand(1, *input_shape).astype(np.float32)
    keras_output = keras_model.predict(keras_input)

    # Create PyTorch model
    pytorch_model = MaskEncoder(input_shape[-1], output_channels, anti_aliasing)
    pytorch_model.eval()

    # Load weights from Keras to PyTorch
    #load_mask_encoder_weights(keras_model, pytorch_model)
    load_keras_weights_to_pytorch_by_name(keras_model, pytorch_model, custom_loaders=None)

    # Convert inputs to PyTorch tensors
    pytorch_input = torch.tensor(keras_input).permute(0, 3, 1, 2)  # Change to (batch, channels, height, width)

    # Get PyTorch output
    pytorch_output = pytorch_model(pytorch_input).detach().numpy()
    pytorch_output = pytorch_output.transpose(0, 2, 3, 1)
    # Compare outputs
    np.testing.assert_allclose(keras_output, pytorch_output, rtol=1e-5, atol=1e-8)