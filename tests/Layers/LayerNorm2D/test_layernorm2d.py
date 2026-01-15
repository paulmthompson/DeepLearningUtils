import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import keras

from src.DeepLearningUtils.Layers.LayerNorm2D.layernorm2d_keras import LayerNorm2d


def test_layernorm2d_save_load(tmp_path):
    """Test that LayerNorm2d can be saved and loaded with non-default parameters."""
    # Non-default parameters
    num_channels = 64
    eps = 1e-6  # default is 1e-3

    # Create LayerNorm2d layer with non-default parameters
    ln_layer = LayerNorm2d(
        num_channels=num_channels,
        eps=eps,
    )

    # Build functional model
    keras_input = keras.Input(shape=(16, 16, num_channels))
    keras_output = ln_layer(keras_input)
    model = keras.Model(inputs=keras_input, outputs=keras_output)

    # Create test input
    input_data = np.random.rand(2, 16, 16, num_channels).astype(np.float32)

    # Get output before saving
    original_output = model.predict(input_data)

    # Save the model
    model_path = tmp_path / "layernorm2d_model.keras"
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
        if isinstance(layer, LayerNorm2d):
            loaded_layer = layer
            break

    assert loaded_layer is not None, "LayerNorm2d layer not found in loaded model"
    assert loaded_layer.num_channels == num_channels, f"Expected num_channels={num_channels}, got {loaded_layer.num_channels}"
    assert loaded_layer.eps == eps, f"Expected eps={eps}, got {loaded_layer.eps}"


def test_layernorm2d_weights_preserved(tmp_path):
    """Test that LayerNorm2d weights are preserved after save/load."""
    num_channels = 32
    eps = 1e-5

    # Create LayerNorm2d layer
    ln_layer = LayerNorm2d(
        num_channels=num_channels,
        eps=eps,
    )

    # Build functional model
    keras_input = keras.Input(shape=(8, 8, num_channels))
    keras_output = ln_layer(keras_input)
    model = keras.Model(inputs=keras_input, outputs=keras_output)

    # Run a forward pass to ensure layer is built
    input_data = np.random.rand(1, 8, 8, num_channels).astype(np.float32)
    _ = model.predict(input_data)

    # Get weights before saving
    original_weight = ln_layer.weight.numpy().copy()
    original_bias = ln_layer.bias.numpy().copy()

    # Save the model
    model_path = tmp_path / "layernorm2d_weights_model.keras"
    model.save(model_path)

    # Load the model
    loaded_model = keras.models.load_model(model_path)

    # Find the loaded layer
    loaded_layer = None
    for layer in loaded_model.layers:
        if isinstance(layer, LayerNorm2d):
            loaded_layer = layer
            break

    assert loaded_layer is not None, "LayerNorm2d layer not found in loaded model"

    # Verify weights are preserved
    np.testing.assert_allclose(
        original_weight,
        loaded_layer.weight.numpy(),
        rtol=1e-5, atol=1e-5,
        err_msg="Weight not preserved after save/load"
    )
    np.testing.assert_allclose(
        original_bias,
        loaded_layer.bias.numpy(),
        rtol=1e-5, atol=1e-5,
        err_msg="Bias not preserved after save/load"
    )


if __name__ == "__main__":
    pytest.main()
