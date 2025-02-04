
import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from torch import nn
import keras

from src.DeepLearningUtils.Layers.Decoders.decoder_keras import UNetDecoder as UNetDecoder_Keras
from src.DeepLearningUtils.Layers.Decoders.decoder_pytorch import UNetDecoder as UNetDecoder_PyTorch

from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_into_pytorch

def transform_layer_name(layer_name):
    parts = layer_name.split('.')
    if len(parts) < 3:
        return layer_name
    if parts[2] == 'conv':
        return f"conv_layers_{parts[1]}"
    elif parts[2] == 'bn':
        return f"bn_layers_{parts[1]}"
    else:
        return layer_name


@pytest.mark.parametrize("input_shape, filter_sizes", [
    ((1, 64, 64, 3), [64, 128, 256, 512]),
    ((1, 128, 128, 3), [32, 64, 128, 256])
])
def test_unet_decoder(input_shape, filter_sizes):
    # Create Keras UNetDecoder
    keras_input = keras.Input(shape=input_shape[1:])
    encoder_outputs = [keras.Input(shape=(input_shape[1] // (2 ** (i+1)), input_shape[2] // (2 ** (i+1)), f)) for i, f in enumerate(filter_sizes)]
    keras_unet_decoder = UNetDecoder_Keras(filter_sizes)
    keras_output = keras_unet_decoder([*encoder_outputs[::-1], keras_input])
    keras_model = keras.Model(inputs=[*encoder_outputs[::-1], keras_input], outputs=keras_output)

    decoder_inputs = [
        *encoder_outputs[::-1],
        torch.zeros(1, *input_shape)]

    decoder_input_channels = [output.shape[-1] for output in decoder_inputs]
    # Create PyTorch UNetDecoder
    pytorch_unet_decoder = UNetDecoder_PyTorch(filter_sizes, decoder_input_channels)

    # Load weights from Keras to PyTorch
    load_keras_into_pytorch(
        keras_model,
        pytorch_unet_decoder,
        (lambda name: transform_layer_name(name)),
    )

    # Generate random input data
    keras_input_data = np.random.rand(*input_shape).astype(np.float32)
    keras_encoder_outputs_data = \
        [np.random.rand(input_shape[0], input_shape[1] // (2 ** (i+1)), input_shape[2] // (2 ** (i+1)), f).astype(np.float32) for
        i, f in enumerate(filter_sizes)]

    # Get Keras output
    keras_output_data = keras_model.predict([*keras_encoder_outputs_data[::-1], keras_input_data])

    # Convert inputs to PyTorch tensors
    pytorch_input_data = torch.tensor(keras_input_data).permute(0, 3, 1,
                                                                2)  # Change to (batch, channels, height, width)
    pytorch_encoder_outputs_data = [torch.tensor(data).permute(0, 3, 1, 2) for data in keras_encoder_outputs_data]

    # Get PyTorch output
    pytorch_unet_decoder.eval()
    pytorch_output_data = pytorch_unet_decoder([*pytorch_encoder_outputs_data[::-1], pytorch_input_data]).detach().numpy()
    pytorch_output_data = np.transpose(pytorch_output_data, (0, 2, 3, 1))  # Change to (batch, height, width, channels)

    # Compare outputs
    np.testing.assert_allclose(keras_output_data, pytorch_output_data, rtol=1e-5, atol=1e-5)

    # Create JIT compiled PyTorch model
    pytorch_unet_decoder_jit = torch.jit.script(pytorch_unet_decoder)
    pytorch_jit_result = pytorch_unet_decoder_jit([*pytorch_encoder_outputs_data[::-1], pytorch_input_data]).detach().numpy()
    np.testing.assert_allclose(pytorch_output_data, np.transpose(pytorch_jit_result,(0,2,3,1)), rtol=1e-5, atol=1e-5)