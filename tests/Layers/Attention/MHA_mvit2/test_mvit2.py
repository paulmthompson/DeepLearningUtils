import pytest
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import keras
from src.DeepLearningUtils.Layers.Attention.MHA_mvit2.attention_keras import MultiHeadAttention as KerasMultiHeadAttention
from src.DeepLearningUtils.Layers.Attention.MHA_mvit2.attention_pytorch import MultiHeadAttention as TorchMultiHeadAttention
from src.DeepLearningUtils.Layers.Attention.MHA_mvit2.attention_pytorch import load_mha_positional_layer_weights
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name, \
    load_keras_layer_weights

def reshape_input(input_data):
    batch_size = 1
    return np.reshape(input_data, (batch_size, -1, input_data.shape[-1]))





@pytest.mark.parametrize("query_shape, key_shape, value_dim, key_dim, heads, drop_rate", [
    ((1, 8, 8, 64), (1, 8, 8, 64), 128, 128, 8, 0.0),
    ((2, 16, 16, 128), (2, 8, 8, 128), 256, 256, 4, 0.0)
])
def test_multihead_attention(query_shape, key_shape, value_dim, key_dim, heads, drop_rate, keras_float32_policy):

    query_seq_len, query_height, query_width, _ = query_shape
    key_seq_len, key_height, key_width, _ = key_shape

    use_positional_embedding = True

    # Create Keras layer
    keras_layer = KerasMultiHeadAttention(
        (query_seq_len, query_height, query_width),
        (key_seq_len, key_height, key_width),
        heads,
        value_dim,
        key_dim,
        drop_rate,
        use_positional_embedding=use_positional_embedding,
        use_key_positional_embedding=use_positional_embedding)

    batch_size = 1

    keras_input_query = np.random.rand(batch_size, *query_shape).astype(np.float32)
    keras_input_key = np.random.rand(batch_size, *key_shape).astype(np.float32)
    keras_input_value = np.random.rand(batch_size, *key_shape).astype(np.float32)

    keras_input_query = reshape_input(keras_input_query)
    keras_input_key = np.reshape(keras_input_key, (batch_size, -1, key_shape[-1]))
    keras_input_value = np.reshape(keras_input_value, (batch_size, -1, key_shape[-1]))

    mask = np.ones((batch_size,
                    query_seq_len*query_height*query_width,
                    key_seq_len*key_height*key_width))

    keras_inputs =[
        keras.Input(shape=keras_input_query.shape[1:]),
        keras.Input(shape=keras_input_key.shape[1:]),
        keras.Input(shape=keras_input_value.shape[1:])
    ]

    mask_inputs = keras.Input(shape=(mask.shape[1], mask.shape[2]))

    #keras_model_output = keras_layer(keras_inputs[0], keras_inputs[1], keras_inputs[2], mask=mask_inputs)
    #keras_model = keras.Model(inputs=[keras_inputs[0], keras_inputs[1], keras_inputs[2], mask_inputs], outputs=keras_model_output)

    #keras_model.summary()
    #keras_output = keras_model.predict([keras_input_query, keras_input_key, keras_input_value, mask])

    keras_output = keras_layer(keras_input_query, keras_input_key, keras_input_value, mask=mask)
    # Create PyTorch layer

    keras_output = keras_output.numpy()
    torch_layer = TorchMultiHeadAttention(
        query_shape=query_shape,
        key_shape=key_shape,
        heads=heads,
        value_dim=value_dim,
        key_dim=key_dim,
        attention_drop_rate=drop_rate,
        use_positional_embedding=use_positional_embedding,
        use_key_positional_embedding=use_positional_embedding
    )
    torch_layer.eval()

    # Load weights from Keras to PyTorch
    load_mha_positional_layer_weights(keras_layer, torch_layer)

    # Convert inputs to PyTorch tensors
    torch_input_query = torch.tensor(keras_input_query)
    torch_input_key = torch.tensor(keras_input_key)
    torch_input_value = torch.tensor(keras_input_value)

    # Get PyTorch output
    torch_output = torch_layer(torch_input_query, torch_input_key, torch_input_value).detach().numpy()
    torch_output = torch_output.reshape(keras_output.shape)
    # Compare outputs
    np.testing.assert_allclose(keras_output, torch_output, rtol=1e-5, atol=1e-5)

    # Create JIT compiled PyTorch model
    torch_model_jit = torch.jit.script(torch_layer)
    torch_jit_result = torch_model_jit(torch_input_query, torch_input_key, torch_input_value).detach().numpy()
    torch_jit_result = torch_jit_result.reshape(keras_output.shape)

    # Compare JIT results
    np.testing.assert_allclose(torch_output, torch_jit_result, rtol=1e-5, atol=1e-5)