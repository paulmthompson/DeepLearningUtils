import pytest
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import keras

keras.config.set_floatx('float32')
keras.config.set_dtype_policy("float32")

from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_keras import \
    RelativePositionalEmbedding2D as KerasRelativePositionalEmbedding2D
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_pytorch import \
    RelativePositionalEmbedding2D as TorchRelativePositionalEmbedding2D
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_pytorch import \
    load_positionaL_embedding_layer_weights
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

@pytest.mark.parametrize("query_shape, key_shape, query_dim, heads, drop_rate", [
    ((1, 32, 32, 64), (1, 32, 32, 64), 64, 8, 0.0),
    ((5, 16, 16, 128), (5, 16, 16, 128), 128, 4, 0.0)
])
def test_positional_embedding_layer(query_shape, key_shape, query_dim, heads, drop_rate):
    # Create Keras layer
    keras_layer = KerasRelativePositionalEmbedding2D(
        query_shape=query_shape,
        key_shape=key_shape,
        query_dim=query_dim,
        heads=heads,
        drop_rate=drop_rate
    )

    batch_size = 1
    query_seq_len, query_height, query_width, _ = query_shape
    key_seq_len, key_height, key_width, _ = key_shape

    keras_input_query = np.random.rand(batch_size, heads, query_seq_len * query_height * query_width, query_dim).astype(np.float32)
    keras_input_scores = np.random.rand(batch_size, heads, query_seq_len * query_height * query_width, key_seq_len * key_height * key_width).astype(np.float32)

    #keras_input = [keras.Input(shape=keras_input_query.shape[1:]),
    #                 keras.Input(shape=keras_input_scores.shape[1:])]
    #keras_output = keras_layer(keras_input)
    #keras_model = keras.Model(inputs=keras_input, outputs=keras_output)

    keras_output = keras.ops.convert_to_numpy(keras_layer([keras_input_query, keras_input_scores]))

    # Create PyTorch layer
    torch_layer = TorchRelativePositionalEmbedding2D(
        query_shape=query_shape,
        key_shape=key_shape,
        query_dim=query_dim,
        heads=heads,
        drop_rate=drop_rate
    )
    torch_layer.eval()
    # Load weights from Keras to PyTorch
    custom_loaders = {'RelativePositionalEmbedding2D': load_positionaL_embedding_layer_weights}
    load_positionaL_embedding_layer_weights(keras_layer, torch_layer)

    # Convert inputs to PyTorch tensors
    torch_input_query = torch.tensor(keras_input_query)
    torch_input_scores = torch.tensor(keras_input_scores)

    # Get PyTorch output
    torch_output = torch_layer(torch_input_query, torch_input_scores).detach().numpy()

    # Compare outputs
    np.testing.assert_allclose(keras_output, torch_output, rtol=1e-5, atol=1e-3)
    #np.testing.assert_allclose(keras_output, keras_input_scores, rtol=1e-5, atol=1e-3)
    #np.testing.assert_allclose(torch_output, torch_input_scores, rtol=1e-5, atol=1e-3)

    # Create JIT compiled PyTorch model
    pytorch_model_jit = torch.jit.script(torch_layer)
    pytorch_jit_result = pytorch_model_jit(torch_input_query, torch_input_scores).detach().numpy()

    # Compare JIT results
    np.testing.assert_allclose(torch_output, pytorch_jit_result, rtol=1e-5, atol=1e-5)


def test_positional_embedding_save_load(tmp_path):
    """Test that RelativePositionalEmbedding2D can be saved and loaded with non-default parameters."""
    # Non-default parameters
    query_shape = (2, 8, 8, 32)  # different from default
    key_shape = (2, 8, 8, 32)
    query_dim = 32
    heads = 4  # default test uses 8
    drop_rate = 0.1  # default is 0.0

    # Create Keras layer with non-default parameters
    keras_layer = KerasRelativePositionalEmbedding2D(
        query_shape=query_shape,
        key_shape=key_shape,
        query_dim=query_dim,
        heads=heads,
        drop_rate=drop_rate
    )

    batch_size = 1
    query_seq_len, query_height, query_width, _ = query_shape
    key_seq_len, key_height, key_width, _ = key_shape

    # Build functional model
    query_input = keras.Input(shape=(heads, query_seq_len * query_height * query_width, query_dim))
    scores_input = keras.Input(shape=(heads, query_seq_len * query_height * query_width, key_seq_len * key_height * key_width))

    output = keras_layer([query_input, scores_input])
    model = keras.Model(inputs=[query_input, scores_input], outputs=output)

    # Create test inputs
    input_query = np.random.rand(batch_size, heads, query_seq_len * query_height * query_width, query_dim).astype(np.float32)
    input_scores = np.random.rand(batch_size, heads, query_seq_len * query_height * query_width, key_seq_len * key_height * key_width).astype(np.float32)

    # Get output before saving
    original_output = model.predict([input_query, input_scores])

    # Save the model
    model_path = tmp_path / "positional_embedding_model.keras"
    model.save(model_path)

    # Load the model
    loaded_model = keras.models.load_model(model_path)

    # Get output after loading
    loaded_output = loaded_model.predict([input_query, input_scores])

    # Compare outputs
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)

    # Verify that non-default parameters are preserved
    loaded_layer = None
    for layer in loaded_model.layers:
        if isinstance(layer, KerasRelativePositionalEmbedding2D):
            loaded_layer = layer
            break

    assert loaded_layer is not None, "RelativePositionalEmbedding2D layer not found in loaded model"
    assert loaded_layer.config.query_shape == query_shape, f"Expected query_shape={query_shape}, got {loaded_layer.config.query_shape}"
    assert loaded_layer.config.key_shape == key_shape, f"Expected key_shape={key_shape}, got {loaded_layer.config.key_shape}"
    assert loaded_layer.config.query_dim == query_dim, f"Expected query_dim={query_dim}, got {loaded_layer.config.query_dim}"
    assert loaded_layer.config.heads == heads, f"Expected heads={heads}, got {loaded_layer.config.heads}"
    assert loaded_layer.config.drop_rate == drop_rate, f"Expected drop_rate={drop_rate}, got {loaded_layer.config.drop_rate}"