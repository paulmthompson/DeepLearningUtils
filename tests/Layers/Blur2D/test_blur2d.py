
import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import keras

from src.DeepLearningUtils.Layers.Blur2D.blur2d_keras import Blur2D as Blur2D_Keras
from src.DeepLearningUtils.Layers.Blur2D.blur2d_pytorch import Blur2D as Blur2D_PyTorch


def test_blur2d_layers():
    # Initialize input data
    input_data = np.random.rand(1, 10, 10, 3).astype(np.float32)  # Shape: (batch, height, width, channels)
    
    # Keras Blur2D
    keras_layer = Blur2D_Keras(kernel_size=5, stride=2, kernel_type="Rect", padding="same")
    keras_input = keras.Input(shape=(10, 10, 3))
    keras_output = keras_layer(keras_input)
    keras_model = keras.Model(inputs=keras_input, outputs=keras_output)
    keras_result = keras_model.predict(input_data)

    # PyTorch Blur2D
    pytorch_layer = Blur2D_PyTorch(kernel_size=5, stride=2, kernel_type="Rect", padding="same")
    pytorch_input = torch.tensor(input_data.transpose(0, 3, 1, 2))  # Convert to (batch, channels, height, width)
    pytorch_result = pytorch_layer(pytorch_input).detach().numpy().transpose(0, 2, 3, 1)  # Convert back to (batch, height, width, channels)

    # Compare results
    np.testing.assert_allclose(keras_result, pytorch_result, rtol=1e-5, atol=1e-5)


    # Create JIT compiled PyTorch model
    pytorch_model = torch.jit.script(pytorch_layer)
    pytorch_jit_result = pytorch_model(pytorch_input).detach().numpy().transpose(0, 2, 3, 1)

    # Compare JIT results
    np.testing.assert_allclose(pytorch_result, pytorch_jit_result, rtol=1e-5, atol=1e-5)
if __name__ == "__main__":
    pytest.main()