
import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import torch
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_pytorch import MaskEncoder
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_keras import create_mask_encoder
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

def create_and_save_mask_encoder_models(
        device_type='cpu',
        keras_weights_path="path_to_keras_weights.h5",
        save_path='pytorch_mask_encoder_jit.pth'):
    # Set device
    device = torch.device(device_type)

    input_shape = (256, 256, 1)
    use_norm = True
    anti_aliasing = True
    output_channels = 256

    # Create Keras model
    keras_model = create_mask_encoder(
        input_shape=input_shape,
        output_channels=output_channels,
        use_norm=use_norm,
        anti_aliasing=anti_aliasing
    )
    keras_model.load_weights(keras_weights_path)

    # Create PyTorch model
    pytorch_model = MaskEncoder(
        input_channels=input_shape[2],
        output_channels=256,
        anti_aliasing=anti_aliasing,
        use_norm=use_norm).to(device)

    # Load weights from Keras to PyTorch
    pytorch_model.eval()
    load_keras_weights_to_pytorch_by_name(keras_model, pytorch_model, custom_loaders=None)

    # Generate random input data
    input_data = np.random.rand(10, *input_shape).astype(np.float32)

    # Get Keras model output
    keras_output = keras_model.predict(input_data)

    input_tensor = torch.tensor(input_data).permute(0, 3, 1, 2).to(device)  # Convert to PyTorch format
    # Get PyTorch model output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor).cpu().detach().numpy().transpose(0, 2, 3, 1)

    # Compare outputs
    np.testing.assert_allclose(keras_output, pytorch_output, rtol=1e-5, atol=1e-5)

    # Create JIT compiled PyTorch model
    pytorch_model_jit = torch.jit.script(pytorch_model)
    torch.jit.save(pytorch_model_jit, save_path)
    print(f"JIT model saved to {save_path}")

if __name__ == "__main__":
    create_and_save_mask_encoder_models(
        device_type='cpu',
        keras_weights_path="/mnt/c/Users/wanglab/Desktop/efficientvit_mask_encoder3_aa4.weights.h5",
        save_path='pytorch_mask_encoder_jit.pth'
    )