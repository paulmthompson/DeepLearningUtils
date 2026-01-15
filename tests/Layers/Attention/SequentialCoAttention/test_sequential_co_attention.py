import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn
import keras

from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_keras import CoAttentionModule as CoAttentionModule_Keras
from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_keras import CoMemoryAttentionModule as CoMemoryAttentionModule_Keras

from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_pytorch import CoMemoryAttentionModule as CoMemoryAttentionModule_PyTorch
from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_pytorch import CoAttentionModule as CoAttentionModule_PyTorch
from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_pytorch import load_coattention_weights



@pytest.mark.parametrize("input_shape, n_frames, key_dim, value_dim", [
    ((1, 256, 16, 16), 5, 128, 128),
    ((1, 256, 8, 8), 10, 128, 128)
])
def test_coattention_module(input_shape, n_frames, key_dim, value_dim, keras_float32_policy):
    # Create Keras CoAttentionModule
    keras_memory_attention_module = CoMemoryAttentionModule_Keras(
        key_dim,
        value_dim,
        query_seq_len=1,
        key_seq_len=1,
        use_norm=False,
        attention_drop_rate=0.0,
        use_positional_embedding=True,
        use_linear_attention=False,
        name=f"memory_self_attention")

    # Define your Keras memory attention module here
    keras_coattention = CoAttentionModule_Keras(
        memory_attention_module=keras_memory_attention_module)

    keras_input_shape = (input_shape[0], 1, input_shape[2], input_shape[3], input_shape[1])
    keras_input = np.random.rand(*keras_input_shape).astype(np.float32)
    keras_memory_bank = np.random.rand(input_shape[0], n_frames, *keras_input_shape[2:]).astype(np.float32)
    keras_mask = np.ones((input_shape[0], n_frames)).astype(np.float32)
    keras_output = keras_coattention([keras_input, keras_memory_bank, keras_mask]).numpy()

    # Create PyTorch CoAttentionModule
    pytorch_memory_attention_module = CoMemoryAttentionModule_PyTorch(
        query_shape=(1, input_shape[2], input_shape[3], key_dim),
        memory_shape=(1, input_shape[2], input_shape[3], key_dim),
        key_dim=key_dim,
        value_dim=value_dim,
        use_norm=False,
        attention_drop_rate=0.0,
        use_positional_embedding=True
    ) # Define your PyTorch memory attention module here
    pytorch_coattention = CoAttentionModule_PyTorch(
        pytorch_memory_attention_module,
        query_shape=(input_shape[0], 1, input_shape[2], input_shape[3], input_shape[1]),
        memory_shape=(input_shape[0], n_frames, input_shape[2], input_shape[3], input_shape[1]),
        key_dim=key_dim,
        value_dim=value_dim
    )

    # Load weights from Keras to PyTorch
    load_coattention_weights(keras_coattention, pytorch_coattention)

    # Convert inputs to PyTorch tensors
    pytorch_input = torch.tensor(keras_input)
    pytorch_memory_bank = torch.tensor(keras_memory_bank) # Change to (batch, frames, height, width, channels)
    pytorch_mask = torch.tensor(keras_mask)

    pytorch_coattention.eval()
    pytorch_output = pytorch_coattention(pytorch_input, pytorch_memory_bank, pytorch_mask).detach().numpy()

    # Compare outputs
    np.testing.assert_allclose(keras_output, pytorch_output, rtol=1e-5, atol=1e-5)

    # Test JIT
    pytorch_model_jit = torch.jit.script(pytorch_coattention)
    pytorch_jit_result = pytorch_model_jit(pytorch_input, pytorch_memory_bank, pytorch_mask).detach().numpy()

    np.testing.assert_allclose(keras_output, pytorch_jit_result, rtol=1e-5, atol=1e-5)


def test_coattention_module_save_load(keras_float32_policy, tmp_path):
    """Test that CoAttentionModule can be saved and loaded with non-default parameters."""
    input_shape = (1, 256, 16, 16)
    n_frames = 5

    # Non-default parameters for CoMemoryAttentionModule
    key_dim = 64  # default is 128
    value_dim = 64  # default is 256
    query_seq_len = 1
    key_seq_len = 1
    use_norm = True  # default is False
    attention_drop_rate = 0.1  # default is 0.0
    use_positional_embedding = False  # default is True
    attention_heads = 4  # default is 8
    use_linear_attention = True  # default is False
    use_qkv_embedding = True  # default is False

    # Create Keras CoMemoryAttentionModule with non-default parameters
    keras_memory_attention_module = CoMemoryAttentionModule_Keras(
        key_dim=key_dim,
        value_dim=value_dim,
        query_seq_len=query_seq_len,
        key_seq_len=key_seq_len,
        use_norm=use_norm,
        attention_drop_rate=attention_drop_rate,
        use_positional_embedding=use_positional_embedding,
        attention_heads=attention_heads,
        use_linear_attention=use_linear_attention,
        use_qkv_embedding=use_qkv_embedding,
        name="memory_self_attention"
    )

    # Create CoAttentionModule
    keras_coattention = CoAttentionModule_Keras(
        memory_attention_module=keras_memory_attention_module
    )

    keras_input_shape = (input_shape[0], 1, input_shape[2], input_shape[3], input_shape[1])
    keras_input = np.random.rand(*keras_input_shape).astype(np.float32)
    keras_memory_bank = np.random.rand(input_shape[0], n_frames, *keras_input_shape[2:]).astype(np.float32)
    keras_mask = np.ones((input_shape[0], n_frames)).astype(np.float32)

    # Build functional model
    query_input = keras.Input(shape=keras_input_shape[1:])
    memory_input = keras.Input(shape=(n_frames, *keras_input_shape[2:]))
    mask_input = keras.Input(shape=(n_frames,))

    output = keras_coattention([query_input, memory_input, mask_input])
    model = keras.Model(inputs=[query_input, memory_input, mask_input], outputs=output)

    # Get output before saving
    original_output = model.predict([keras_input, keras_memory_bank, keras_mask])

    # Save the model
    model_path = tmp_path / "coattention_model.keras"
    model.save(model_path)

    # Load the model
    loaded_model = keras.models.load_model(model_path)

    # Get output after loading
    loaded_output = loaded_model.predict([keras_input, keras_memory_bank, keras_mask])

    # Compare outputs
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)

    # Verify that non-default parameters are preserved
    loaded_coattention_layer = None
    for layer in loaded_model.layers:
        if isinstance(layer, CoAttentionModule_Keras):
            loaded_coattention_layer = layer
            break

    assert loaded_coattention_layer is not None, "CoAttentionModule layer not found in loaded model"

    # Check the memory_attention_module parameters
    loaded_memory_module = loaded_coattention_layer.memory_attention_module
    assert isinstance(loaded_memory_module, CoMemoryAttentionModule_Keras), "memory_attention_module is not CoMemoryAttentionModule"
    assert loaded_memory_module.key_dim == key_dim, f"Expected key_dim={key_dim}, got {loaded_memory_module.key_dim}"
    assert loaded_memory_module.value_dim == value_dim, f"Expected value_dim={value_dim}, got {loaded_memory_module.value_dim}"
    assert loaded_memory_module.query_seq_len == query_seq_len, f"Expected query_seq_len={query_seq_len}, got {loaded_memory_module.query_seq_len}"
    assert loaded_memory_module.key_seq_len == key_seq_len, f"Expected key_seq_len={key_seq_len}, got {loaded_memory_module.key_seq_len}"
    assert loaded_memory_module.use_norm == use_norm, f"Expected use_norm={use_norm}, got {loaded_memory_module.use_norm}"
    assert loaded_memory_module.attention_drop_rate == attention_drop_rate, f"Expected attention_drop_rate={attention_drop_rate}, got {loaded_memory_module.attention_drop_rate}"
    assert loaded_memory_module.use_positional_embedding == use_positional_embedding, f"Expected use_positional_embedding={use_positional_embedding}, got {loaded_memory_module.use_positional_embedding}"
    assert loaded_memory_module.attention_heads == attention_heads, f"Expected attention_heads={attention_heads}, got {loaded_memory_module.attention_heads}"
    assert loaded_memory_module.use_linear_attention == use_linear_attention, f"Expected use_linear_attention={use_linear_attention}, got {loaded_memory_module.use_linear_attention}"
    assert loaded_memory_module.use_qkv_embedding == use_qkv_embedding, f"Expected use_qkv_embedding={use_qkv_embedding}, got {loaded_memory_module.use_qkv_embedding}"