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
from src.DeepLearningUtils.Layers.Backbones.Memory_Encoder.memory_encoder_pytorch import load_memory_encoder_weights



@pytest.mark.parametrize("input_shape, output_channels, seq_len, anti_aliasing, combine_operation", [
    ((3, 256, 256), 256, 1, True, 'add'),
    ((3, 256, 256), 256, 5, False, 'add')
])
def test_memory_encoder(input_shape, output_channels, seq_len, anti_aliasing, combine_operation, keras_float32_policy):
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
    keras_output = keras_memory_encoder([keras_input, keras_mask_input], training=False).numpy()

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


def test_memory_encoder_save_load(keras_float32_policy, tmp_path):
    """Test that MemoryEncoderLayer can be saved and loaded with non-default parameters."""
    input_shape = (3, 256, 256)
    output_channels = 256
    seq_len = 5
    anti_aliasing = True
    use_norm = True

    # Non-default parameters
    combine_operation = 'dense'  # default is 'dense', but we're testing the whole flow

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
        activation=keras.layers.Activation('relu')  # Non-default activation
    )

    # Build functional model
    memory_input = keras.Input(shape=(seq_len, *keras_input_shape))
    mask_input = keras.Input(shape=(seq_len, input_shape[1], input_shape[2], 1))
    memory_frame_num = keras.Input(shape=(1,))

    output = keras_memory_encoder([memory_input, mask_input, memory_frame_num])
    model = keras.Model(inputs=[memory_input, mask_input, memory_frame_num], outputs=output)

    # Create test inputs
    keras_input = np.random.rand(1, seq_len, *keras_input_shape).astype(np.float32)
    keras_mask_input = np.random.rand(1, seq_len, input_shape[1], input_shape[2], 1).astype(np.float32)
    frame_num = np.array([[1]]).astype(np.float32)

    # Get output before saving
    original_output = model.predict([keras_input, keras_mask_input, frame_num])

    # Save the model
    model_path = tmp_path / "memory_encoder_model.keras"
    model.save(model_path)

    # Load the model
    loaded_model = keras.models.load_model(model_path)

    # Get output after loading
    loaded_output = loaded_model.predict([keras_input, keras_mask_input, frame_num])

    # Compare outputs
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)

    # Verify that non-default parameters are preserved
    loaded_memory_encoder = None
    for layer in loaded_model.layers:
        if isinstance(layer, MemoryEncoder_Keras):
            loaded_memory_encoder = layer
            break

    assert loaded_memory_encoder is not None, "MemoryEncoderLayer not found in loaded model"
    assert loaded_memory_encoder.height == input_shape[1], f"Expected height={input_shape[1]}, got {loaded_memory_encoder.height}"
    assert loaded_memory_encoder.width == input_shape[2], f"Expected width={input_shape[2]}, got {loaded_memory_encoder.width}"
    assert loaded_memory_encoder.channels == input_shape[0], f"Expected channels={input_shape[0]}, got {loaded_memory_encoder.channels}"
    assert loaded_memory_encoder.combine_operation == combine_operation, f"Expected combine_operation={combine_operation}, got {loaded_memory_encoder.combine_operation}"