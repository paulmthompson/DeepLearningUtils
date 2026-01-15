
import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import keras

from src.DeepLearningUtils.Layers.Blur2D.blur2d_keras import Blur2D as Blur2D_Keras
from src.DeepLearningUtils.Layers.Blur2D.blur2d_pytorch import Blur2D as Blur2D_PyTorch
@pytest.mark.parametrize("kernel_type, kernel_size",
                         [("Rect", 2), ("Triangle", 3), ("Binomial",5)])
def test_blur2d_layers(kernel_type, kernel_size):
    # Initialize input data
    input_data = np.random.rand(1, 10, 10, 3).astype(np.float32)  # Shape: (batch, height, width, channels)

    # Keras Blur2D
    keras_layer = Blur2D_Keras(
        kernel_size=kernel_size,
        stride=2,
        kernel_type=kernel_type,
        padding="same")
    keras_input = keras.Input(shape=(10, 10, 3))
    keras_output = keras_layer(keras_input)
    keras_model = keras.Model(inputs=keras_input, outputs=keras_output)
    keras_result = keras_model.predict(input_data)

    # PyTorch Blur2D
    pytorch_layer = Blur2D_PyTorch(
        kernel_size=kernel_size,
        stride=2,
        kernel_type=kernel_type,
        padding="same")
    pytorch_input = torch.tensor(input_data.transpose(0, 3, 1, 2))  # Convert to (batch, channels, height, width)
    pytorch_result = pytorch_layer(pytorch_input).detach().numpy().transpose(0, 2, 3, 1)  # Convert back to (batch, height, width, channels)

    # Compare results
    np.testing.assert_allclose(keras_result, pytorch_result, rtol=1e-5, atol=1e-5)

    # Create JIT compiled PyTorch model
    pytorch_model = torch.jit.script(pytorch_layer)
    pytorch_jit_result = pytorch_model(pytorch_input).detach().numpy().transpose(0, 2, 3, 1)

    # Compare JIT results
    np.testing.assert_allclose(pytorch_result, pytorch_jit_result, rtol=1e-5, atol=1e-5)


def test_blur2d_save_load(tmp_path):
    """Test that Blur2D can be saved and loaded with non-default parameters."""
    # Non-default parameters
    kernel_size = 5  # default is 2
    stride = 1  # default is 2
    kernel_type = "Binomial"  # default is "Rect"
    padding = "valid"  # default is "same"

    # Create Keras Blur2D layer with non-default parameters
    keras_layer = Blur2D_Keras(
        kernel_size=kernel_size,
        stride=stride,
        kernel_type=kernel_type,
        padding=padding
    )

    # Build functional model
    keras_input = keras.Input(shape=(16, 16, 3))
    keras_output = keras_layer(keras_input)
    model = keras.Model(inputs=keras_input, outputs=keras_output)

    # Create test input
    input_data = np.random.rand(1, 16, 16, 3).astype(np.float32)

    # Get output before saving
    original_output = model.predict(input_data)

    # Save the model
    model_path = tmp_path / "blur2d_model.keras"
    model.save(model_path)

    # Load the model
    loaded_model = keras.models.load_model(model_path)

    # Get output after loading
    loaded_output = loaded_model.predict(input_data)

    # Compare outputs
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)

    # Verify that non-default parameters are preserved
    loaded_layer = None
    for layer in loaded_model.layers:
        if isinstance(layer, Blur2D_Keras):
            loaded_layer = layer
            break

    assert loaded_layer is not None, "Blur2D layer not found in loaded model"
    assert loaded_layer.config.kernel_size == kernel_size, f"Expected kernel_size={kernel_size}, got {loaded_layer.config.kernel_size}"
    assert loaded_layer.config.stride == stride, f"Expected stride={stride}, got {loaded_layer.config.stride}"
    assert loaded_layer.config.kernel_type == kernel_type, f"Expected kernel_type={kernel_type}, got {loaded_layer.config.kernel_type}"
    assert loaded_layer.config.padding == padding, f"Expected padding={padding}, got {loaded_layer.config.padding}"


if __name__ == "__main__":
    pytest.main()