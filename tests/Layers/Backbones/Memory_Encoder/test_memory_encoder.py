import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn
import keras

from src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_keras import EfficientViT_B as EfficientViT_Keras
from src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_pytorch import EfficientViT_B as EfficientViT_PyTorch
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_keras import create_mask_encoder
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_pytorch import MaskEncoder
from src.DeepLearningUtils.Layers.Backbones.Memory_Encoder.memory_encoder_keras import MemoryEncoderLayer as MemoryEncoder_Keras
from src.DeepLearningUtils.Layers.Backbones.Memory_Encoder.memory_encoder_pytorch import MemoryModelBase as MemoryEncoder_PyTorch
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

def load_memory_encoder_weights(keras_model, pytorch_model):
    # Load weights from Keras to PyTorch
    load_keras_weights_to_pytorch_by_name(keras_model.base_memory_model, pytorch_model.base_model)
    load_keras_weights_to_pytorch_by_name(keras_model.mask_encoder_model, pytorch_model.mask_encoder)

@pytest.mark.parametrize("input_shape, output_channels, seq_len, anti_aliasing, combine_operation", [
    ((3, 256, 256), 256, 1, True, 'add'),
    ((3, 256, 256), 256, 5, False, 'add')
])
def test_memory_encoder(input_shape, output_channels, seq_len, anti_aliasing, combine_operation):
    # Create Keras EfficientViT model

    use_norm = True

    keras_input_shape = (input_shape[1], input_shape[2], input_shape[0])

    keras_efficientvit = EfficientViT_Keras(
        input_shape=keras_input_shape,
        upsample_levels=1,
        anti_aliasing=False,
        use_norm=use_norm,
        output_filters=output_channels,
    )
    keras_mask_encoder = create_mask_encoder(
        (keras_input_shape[0], keras_input_shape[1], 1),
        output_channels,
        anti_aliasing,
        use_norm=use_norm)
    keras_memory_encoder = MemoryEncoder_Keras(
        base_memory_model=keras_efficientvit,
        mask_encoder_model=keras_mask_encoder,
        height=input_shape[1],
        width=input_shape[2],
        channels=input_shape[0],
        combine_operation=combine_operation,
        activation=keras.layers.Activation('linear')
    )
    keras_input = np.random.rand(1, seq_len, *keras_input_shape).astype(np.float32)
    keras_mask_input = np.random.rand(1, seq_len, input_shape[1], input_shape[2], 1).astype(np.float32)
    keras_memory_encoder.trainable = False
    keras_output = keras_memory_encoder([keras_input, keras_mask_input], training=False).detach().numpy()

    # Create PyTorch EfficientViT model
    pytorch_efficientvit = EfficientViT_PyTorch(
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        upsample_levels=1,
        anti_aliasing=anti_aliasing,
        use_norm=use_norm,
        output_filters=output_channels,)
    pytorch_mask_encoder = MaskEncoder(
        1,
        output_channels,
        anti_aliasing,
        use_norm=use_norm)
    pytorch_memory_encoder = MemoryEncoder_PyTorch(
        base_model=pytorch_efficientvit,
        mask_encoder=pytorch_mask_encoder,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        combine_operation=combine_operation,
        activation=nn.Identity(),
    )

    # Load weights from Keras to PyTorch
    load_memory_encoder_weights(keras_memory_encoder, pytorch_memory_encoder)

    # Convert inputs to PyTorch tensors
    pytorch_input = torch.tensor(keras_input).permute(0, 1, 4, 2, 3)  # Change to (batch, channels, height, width)
    pytorch_mask_input = torch.tensor(keras_mask_input).permute(0, 1, 4, 2, 3)  # Change to (batch, channels, height, width)

    pytorch_mask = torch.ones(1, seq_len).long()
    # Get PyTorch output
    pytorch_memory_encoder.eval()
    pytorch_output = pytorch_memory_encoder(pytorch_input, pytorch_mask_input,pytorch_mask).detach().numpy()

    # Compare outputs
    np.testing.assert_allclose(keras_output, pytorch_output, rtol=1e-5, atol=1e-3)

    # Create JIT compiled PyTorch model
    pytorch_model_jit = torch.jit.script(pytorch_memory_encoder)
    pytorch_jit_result = pytorch_model_jit(pytorch_input, pytorch_mask_input,pytorch_mask).detach().numpy()

    # Compare outputs
    np.testing.assert_allclose(pytorch_output, pytorch_jit_result, rtol=1e-5, atol=1e-3)