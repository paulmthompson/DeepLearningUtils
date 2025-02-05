import os

import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from torch import nn
import keras
from src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_keras import EfficientViT_B as KerasEfficientViT
from src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_pytorch import EfficientViT_B as PyTorchEfficientViT
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

def create_and_save_models(
        device_type='cpu',
        keras_weights_path="path_to_keras_weights.h5",
        save_path='pytorch_model_jit.pth'):
    # Set device
    device = torch.device(device_type)

    input_shape = (256, 256, 3)
    use_norm = True
    upsample_levels = 1
    block_types = ["conv", "conv", "transform", "transform"]
    anti_aliasing = True
    num_blocks = [2, 2, 3, 3]
    out_channels = [16, 32, 64, 128]
    stem_width = 16
    expansions = 4
    is_fused = False
    head_dimension = 16
    output_filters = 256
    activation = "keras.activations.hard_swish"
    use_features_activation = False

    # Create Keras model
    keras_model = KerasEfficientViT(
        num_blocks=num_blocks,
        out_channels=out_channels,
        stem_width=stem_width,
        block_types=block_types,
        expansions=expansions,
        is_fused=is_fused,
        head_dimension=head_dimension,
        output_filters=output_filters,
        input_shape=input_shape,
        activation=activation,
        use_norm=use_norm,
        upsample_levels=upsample_levels,
        use_features_activation=use_features_activation,
        anti_aliasing=anti_aliasing
    )
    #keras_model.load_weights(keras_weights_path, skip_mismatch=True)
    keras_model.trainable = False

    if activation == "keras.activations.hard_swish":
        activation = nn.Hardswish()
    elif activation == "keras.activations.relu":
        activation = nn.ReLU()
    else:
        raise ValueError(f"Activation function {activation} is not supported")

    # Create PyTorch model
    pytorch_model = PyTorchEfficientViT(
        num_blocks=num_blocks,
        out_channels=out_channels,
        stem_width=stem_width,
        block_types=block_types,
        expansions=expansions,
        is_fused=is_fused,
        head_dimension=head_dimension,
        output_filters=output_filters,
        input_shape=(input_shape[2], input_shape[0], input_shape[1]),
        activation=activation,
        use_norm=use_norm,
        upsample_levels=upsample_levels,
        use_features_activation=use_features_activation,
        anti_aliasing=anti_aliasing
    ).to(device)

    # Load weights from Keras to PyTorch
    load_keras_weights_to_pytorch_by_name(keras_model, pytorch_model)

    # Generate random input data
    input_data = np.random.rand(1, 256, 256, 3).astype(np.float32)
    input_tensor = torch.tensor(input_data).permute(0, 3, 1, 2).to(device)  # Convert to PyTorch format

    # Get Keras model output
    keras_output = keras_model.predict(input_data)

    # Get PyTorch model output
    pytorch_model.eval()
    pytorch_output = pytorch_model(input_tensor).cpu().detach().numpy().transpose(0, 2, 3, 1)

    # Compare outputs
    np.testing.assert_allclose(keras_output, pytorch_output, rtol=1e-5, atol=1e-5)

    # Create JIT compiled PyTorch model
    pytorch_model_jit = torch.jit.script(pytorch_model)
    torch.jit.save(pytorch_model_jit, save_path)
    print(f"JIT model saved to {save_path}")

if __name__ == "__main__":
    create_and_save_models(
        device_type='cpu',
        keras_weights_path="/mnt/c/Users/wanglab/Desktop/efficientvit_memory7_aa4.weights.h5",
        save_path='pytorch_model_jit.pth')