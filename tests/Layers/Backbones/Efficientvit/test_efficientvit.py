import pytest

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np

import keras
from src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_keras import EfficientViT_B as KerasEfficientViT
from src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_pytorch import EfficientViT_B as PyTorchEfficientViT
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

"""
@pytest.fixture
def keras_model():
    return KerasEfficientViT()

@pytest.fixture
def pytorch_model():
    return PyTorchEfficientViT().cpu()
"""


def test_model_conversion(

):
    device = torch.device('cpu')

    #block_types=["conv", "conv", "conv", "conv"]
    stem_width = 16
    use_norm = True
    block_types = ["conv", "conv", "transform", "transform"]
    upsample_levels = 1
    anti_aliasing = True
    output_filters = 256

    keras_model = KerasEfficientViT(
        block_types=block_types,
        stem_width=stem_width,
        output_filters=output_filters,
        input_shape=(256, 256, 3),
        use_norm=use_norm,
        upsample_levels=upsample_levels,
        anti_aliasing=anti_aliasing
    ).to(device)

    #keras_model.load_weights("/mnt/c/Users/wanglab/Desktop/efficientvit_memory7_aa4.weights.h5")

    pytorch_model = PyTorchEfficientViT(
        block_types=block_types,
        stem_width=stem_width,
        output_filters=output_filters,
        input_shape=(3, 256, 256),
        use_norm=use_norm,
        upsample_levels=upsample_levels,
        anti_aliasing=anti_aliasing
    ).to(device)
    # Generate random input data
    input_data = np.random.rand(3, 256, 256, 3).astype(np.float32)
    input_tensor = torch.tensor(input_data).permute(0, 3, 1, 2).to(device)  # Convert to PyTorch format

    # Get Keras model output
    keras_output = keras_model.predict(input_data)

    # Load weights from Keras model to PyTorch model
    load_keras_weights_to_pytorch_by_name(keras_model, pytorch_model)

    # Get PyTorch model output
    # Put in eval mode
    pytorch_model.eval()
    pytorch_output = pytorch_model(input_tensor).cpu().detach().numpy().transpose(0, 2, 3, 1)

    # Compare outputs
    np.testing.assert_allclose(keras_output, pytorch_output, rtol=1e-5, atol=1e-5)

    # Create JIT compiled PyTorch model
    pytorch_model_jit = torch.jit.script(pytorch_model)
    pytorch_jit_result = pytorch_model_jit(input_tensor).detach().numpy().transpose(0, 2, 3, 1)

    # Compare JIT results
    np.testing.assert_allclose(pytorch_output, pytorch_jit_result, rtol=1e-5, atol=1e-5)

    encoder_backbone_outputs = []
    for layer in [
        "stem_MB_output",  # 128x128
        "stack_1_block_2_output",  # 64x64
        "stack_2_block_2_output",  # 32x32
        "features_conv",
    ]:
        encoder_backbone_outputs.append(keras_model.get_layer(layer).output)

    keras_model = keras.models.Model(
        inputs=keras_model.input,
        outputs=encoder_backbone_outputs,
        name="efficientvit")

    keras_output = keras_model.predict(input_data)

    pytorch_output = pytorch_model.forward_intermediate(input_tensor)

    # Compare outputs
    for i in range(len(encoder_backbone_outputs)):
        print(i)
        np.testing.assert_allclose(keras_output[i], pytorch_output[i].cpu().detach().numpy().transpose(0, 2, 3, 1), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])