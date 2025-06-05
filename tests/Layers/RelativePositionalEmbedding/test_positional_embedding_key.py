import pytest
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import keras

from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_keras import \
    RelativePositionalEmbedding2DKey as KerasRelativePositionalEmbedding2DKey
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_pytorch import \
    RelativePositionalEmbedding2DKey as TorchRelativePositionalEmbedding2DKey
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_pytorch import \
    load_positionaL_embedding_layer_weights
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

@pytest.mark.parametrize("query_shape, key_shape, query_dim, heads, drop_rate", [
    ((1, 32, 32, 64), (1, 32, 32, 64), 64, 8, 0.0),
    ((5, 16, 16, 128), (5, 16, 16, 128), 128, 4, 0.0)
])
def test_positional_embedding_layer_key(query_shape, key_shape, query_dim, heads, drop_rate):
    # Create Keras layer
    keras_layer = KerasRelativePositionalEmbedding2DKey(
        query_shape=query_shape,
        key_shape=key_shape,
        query_dim=query_dim,
        heads=heads,
        drop_rate=drop_rate
    )
    batch_size = 1
    key_seq_len, key_height, key_width, _ = key_shape
    query_seq_len, query_height, query_width, _ = query_shape

    keras_input_key = np.random.rand(batch_size, heads, key_seq_len * key_height * key_width, query_dim).astype(np.float32)
    keras_input_scores = np.random.rand(batch_size, heads, query_seq_len * query_height * query_width, key_seq_len * key_height * key_width).astype(np.float32)

    keras_input = [keras.Input(shape=keras_input_key.shape[1:]),
                   keras.Input(shape=keras_input_scores.shape[1:])]
    keras_output = keras_layer(keras_input)
    keras_model = keras.Model(inputs=keras_input, outputs=keras_output)

    keras_output = keras_model.predict([keras_input_key, keras_input_scores])

    # Create PyTorch layer
    torch_layer = TorchRelativePositionalEmbedding2DKey(
        query_shape=query_shape,
        key_shape=key_shape,
        query_dim=query_dim,
        heads=heads,
        drop_rate=drop_rate
    )
    torch_layer.eval()

    # Load weights from Keras to PyTorch
    custom_loaders = {'RelativePositionalEmbedding2DKey': load_positionaL_embedding_layer_weights}
    load_positionaL_embedding_layer_weights(keras_layer, torch_layer)

    # Convert inputs to PyTorch tensors
    torch_input_key = torch.tensor(keras_input_key)
    torch_input_scores = torch.tensor(keras_input_scores)

    # Get PyTorch output
    torch_output = torch_layer(torch_input_key, torch_input_scores).detach().numpy()

    # Compare outputs
    np.testing.assert_allclose(keras_output, torch_output, rtol=1e-5, atol=1e-3)

    # Create JIT compiled PyTorch model
    pytorch_model_jit = torch.jit.script(torch_layer)
    pytorch_jit_result = pytorch_model_jit(torch_input_key, torch_input_scores).detach().numpy()

    # Compare JIT results
    np.testing.assert_allclose(torch_output, pytorch_jit_result, rtol=1e-5, atol=1e-5)