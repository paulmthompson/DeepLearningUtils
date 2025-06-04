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
        (1, input_shape[2], input_shape[3], key_dim),
        (1, input_shape[2], input_shape[3], key_dim),
        key_dim,
        value_dim,
        use_norm=False,
        attention_drop_rate=0.0,
        use_positional_embedding=True,
        ) # Define your PyTorch memory attention module here
    pytorch_coattention = CoAttentionModule_PyTorch(
        pytorch_memory_attention_module,
        (input_shape[0], 1, input_shape[2], input_shape[3], input_shape[1]),
        (input_shape[0], n_frames, input_shape[2], input_shape[3], input_shape[1]),
        key_dim,
        value_dim
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