import pytest

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
import keras

from src.DeepLearningUtils.Layers.Normalization.bn_keras import AccumBatchNormalization


def test_accum_batch_normalization_save_load(tmp_path):
    """Test that AccumBatchNormalization can be saved and loaded with non-default parameters."""
    # Non-default parameters
    accum_steps = 4  # default is 1
    momentum = 0.9  # default is 0.99
    epsilon = 1e-5  # default is 1e-3

    # Create AccumBatchNormalization layer with non-default parameters
    bn_layer = AccumBatchNormalization(
        accum_steps=accum_steps,
        momentum=momentum,
        epsilon=epsilon,
        trainable=True
    )

    # Build functional model
    keras_input = keras.Input(shape=(16, 16, 32))
    keras_output = bn_layer(keras_input)
    model = keras.Model(inputs=keras_input, outputs=keras_output)

    # Create test input
    input_data = np.random.rand(2, 16, 16, 32).astype(np.float32)

    # Get output before saving (in inference mode)
    original_output = model.predict(input_data)

    # Save the model
    model_path = tmp_path / "accum_bn_model.keras"
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
        if isinstance(layer, AccumBatchNormalization):
            loaded_layer = layer
            break

    assert loaded_layer is not None, "AccumBatchNormalization layer not found in loaded model"
    assert loaded_layer.accum_steps == accum_steps, f"Expected accum_steps={accum_steps}, got {loaded_layer.accum_steps}"
    assert loaded_layer.momentum == momentum, f"Expected momentum={momentum}, got {loaded_layer.momentum}"
    assert loaded_layer.epsilon == epsilon, f"Expected epsilon={epsilon}, got {loaded_layer.epsilon}"


def test_accum_batch_normalization_training_mode(tmp_path):
    """Test that AccumBatchNormalization works correctly in training mode after save/load."""
    # Non-default parameters
    accum_steps = 2
    momentum = 0.95
    epsilon = 1e-4

    # Create AccumBatchNormalization layer
    bn_layer = AccumBatchNormalization(
        accum_steps=accum_steps,
        momentum=momentum,
        epsilon=epsilon,
    )

    # Build functional model
    keras_input = keras.Input(shape=(8, 8, 16))
    keras_output = bn_layer(keras_input)
    model = keras.Model(inputs=keras_input, outputs=keras_output)

    # Create test input and run a training step to update moving averages
    input_data = np.random.rand(4, 8, 8, 16).astype(np.float32)
    
    # Run forward pass in training mode to accumulate statistics
    _ = model(input_data, training=True)
    _ = model(input_data, training=True)  # Second step to trigger update

    # Get moving mean and variance before saving
    original_moving_mean = bn_layer.moving_mean.numpy().copy()
    original_moving_variance = bn_layer.moving_variance.numpy().copy()

    # Save the model
    model_path = tmp_path / "accum_bn_training_model.keras"
    model.save(model_path)

    # Load the model
    loaded_model = keras.models.load_model(model_path)

    # Find the loaded layer
    loaded_layer = None
    for layer in loaded_model.layers:
        if isinstance(layer, AccumBatchNormalization):
            loaded_layer = layer
            break

    assert loaded_layer is not None, "AccumBatchNormalization layer not found in loaded model"

    # Verify moving statistics are preserved
    np.testing.assert_allclose(
        original_moving_mean, 
        loaded_layer.moving_mean.numpy(), 
        rtol=1e-5, atol=1e-5,
        err_msg="Moving mean not preserved after save/load"
    )
    np.testing.assert_allclose(
        original_moving_variance, 
        loaded_layer.moving_variance.numpy(), 
        rtol=1e-5, atol=1e-5,
        err_msg="Moving variance not preserved after save/load"
    )


if __name__ == "__main__":
    pytest.main()
