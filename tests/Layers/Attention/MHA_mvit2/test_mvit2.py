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


def test_multihead_attention_save_load(keras_float32_policy, tmp_path):
    """Test that MultiHeadAttention can be saved and loaded with non-default parameters."""
    query_shape = (1, 8, 8, 64)
    key_shape = (1, 8, 8, 64)
    query_seq_len, query_height, query_width, _ = query_shape
    key_seq_len, key_height, key_width, _ = key_shape

    # Non-default parameters
    heads = 4  # default is 8
    value_dim = 64  # default is 128
    key_dim = 64  # default is 128
    drop_rate = 0.1  # default is 0.0
    use_positional_embedding = False  # default is True
    use_key_positional_embedding = False  # default is True
    use_linear_attention = True  # default is False
    query_embedding = False  # default is True
    key_embedding = False  # default is True
    value_embedding = False  # default is True

    # Create Keras layer with non-default parameters
    keras_layer = KerasMultiHeadAttention(
        query_shape=(query_seq_len, query_height, query_width),
        key_shape=(key_seq_len, key_height, key_width),
        h=heads,
        value_dim=value_dim,
        key_dim=key_dim,
        attention_drop_rate=drop_rate,
        use_positional_embedding=use_positional_embedding,
        use_key_positional_embedding=use_key_positional_embedding,
        use_linear_attention=use_linear_attention,
        query_embedding=query_embedding,
        key_embedding=key_embedding,
        value_embedding=value_embedding,
    )

    batch_size = 1
    keras_input_query = np.random.rand(batch_size, *query_shape).astype(np.float32)
    keras_input_key = np.random.rand(batch_size, *key_shape).astype(np.float32)
    keras_input_value = np.random.rand(batch_size, *key_shape).astype(np.float32)

    keras_input_query = reshape_input(keras_input_query)
    keras_input_key = np.reshape(keras_input_key, (batch_size, -1, key_shape[-1]))
    keras_input_value = np.reshape(keras_input_value, (batch_size, -1, key_shape[-1]))

    mask = np.ones((batch_size,
                    query_seq_len * query_height * query_width,
                    key_seq_len * key_height * key_width))

    # Build functional model
    query_input = keras.Input(shape=keras_input_query.shape[1:])
    key_input = keras.Input(shape=keras_input_key.shape[1:])
    value_input = keras.Input(shape=keras_input_value.shape[1:])
    mask_input = keras.Input(shape=(mask.shape[1], mask.shape[2]))

    output = keras_layer(query_input, key_input, value_input, mask=mask_input)
    model = keras.Model(inputs=[query_input, key_input, value_input, mask_input], outputs=output)

    # Get output before saving
    original_output = model.predict([keras_input_query, keras_input_key, keras_input_value, mask])

    # Save the model
    model_path = tmp_path / "mha_model.keras"
    model.save(model_path)

    # Load the model
    loaded_model = keras.models.load_model(model_path)

    # Get output after loading
    loaded_output = loaded_model.predict([keras_input_query, keras_input_key, keras_input_value, mask])

    # Compare outputs
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)

    # Verify that non-default parameters are preserved
    loaded_layer = None
    for layer in loaded_model.layers:
        if isinstance(layer, KerasMultiHeadAttention):
            loaded_layer = layer
            break

    assert loaded_layer is not None, "MultiHeadAttention layer not found in loaded model"
    assert loaded_layer.h == heads, f"Expected h={heads}, got {loaded_layer.h}"
    assert loaded_layer.value_dim == value_dim, f"Expected value_dim={value_dim}, got {loaded_layer.value_dim}"
    assert loaded_layer.key_dim == key_dim, f"Expected key_dim={key_dim}, got {loaded_layer.key_dim}"
    assert loaded_layer.attention_drop_rate == drop_rate, f"Expected attention_drop_rate={drop_rate}, got {loaded_layer.attention_drop_rate}"
    assert loaded_layer.use_positional_embedding == use_positional_embedding, f"Expected use_positional_embedding={use_positional_embedding}, got {loaded_layer.use_positional_embedding}"
    assert loaded_layer.use_key_positional_embedding == use_key_positional_embedding, f"Expected use_key_positional_embedding={use_key_positional_embedding}, got {loaded_layer.use_key_positional_embedding}"
    assert loaded_layer.use_linear_attention == use_linear_attention, f"Expected use_linear_attention={use_linear_attention}, got {loaded_layer.use_linear_attention}"
    assert loaded_layer.query_embedding == query_embedding, f"Expected query_embedding={query_embedding}, got {loaded_layer.query_embedding}"
    assert loaded_layer.key_embedding == key_embedding, f"Expected key_embedding={key_embedding}, got {loaded_layer.key_embedding}"
    assert loaded_layer.value_embedding == value_embedding, f"Expected value_embedding={value_embedding}, got {loaded_layer.value_embedding}"