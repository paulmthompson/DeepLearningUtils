
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from torch import nn
import cv2
import keras

from typing import Tuple, List

from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_pytorch import CoAttentionModule
from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_pytorch import CoMemoryAttentionModule

from src.DeepLearningUtils.Layers.Backbones.Memory_Encoder.memory_encoder_pytorch import MemoryModelBase
from src.DeepLearningUtils.Layers.Decoders.decoder_pytorch import UNetDecoder

from src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_pytorch import EfficientViT_B
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_pytorch import MaskEncoder

from src.DeepLearningUtils.scripts.pytorch_conversion.create_segment_keras import create_model

from src.DeepLearningUtils.Layers.Decoders.decoder_pytorch import transform_layer_name
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_into_pytorch
from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_pytorch import load_coattention_weights
from src.DeepLearningUtils.Layers.Backbones.Memory_Encoder.memory_encoder_pytorch import load_memory_encoder_weights
from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

from PIL import Image
from torchvision import transforms

def load_images_and_masks(data_folder, image_size=(256, 256)):
    images_folder = os.path.join(data_folder, 'images')
    labels_folder = os.path.join(data_folder, 'labels')

    image_files = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.png')])
    label_files = sorted([os.path.join(labels_folder, f) for f in os.listdir(labels_folder) if f.endswith('.png')])

    images = []
    masks = []

    for image_file, label_file in zip(image_files, label_files):
        image = Image.open(image_file).convert('RGB')
        mask = Image.open(label_file).convert('L')

        image = image.resize(image_size)
        mask = mask.resize(image_size)

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0

        mask = mask > 0.1

        # convert to float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        images.append(image)
        masks.append(mask)

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    return images, masks


def prepare_data_for_keras(images, masks):
    masks = np.expand_dims(masks, axis=3)
    return images, masks


def prepare_data_for_pytorch(images, masks):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    images = torch.stack([transform(image) for image in images])
    masks = torch.stack([transform(mask) for mask in masks])

    # images = np.transpose(images, (0, 3, 1, 2))
    # masks = np.expand_dims(masks, axis=1)

    return images, masks

class CombinedModel(nn.Module):
    def __init__(self,
                 encoder_model,
                 mask_model,
                 input_shape: Tuple[int, int, int],
                 seq_len: int,
                 attention_stacks: int = 1,):
        super(CombinedModel, self).__init__()
        self.encoder_model = encoder_model
        self.mask_model = mask_model
        self.input_shape: Tuple[int, int, int] = input_shape
        self.seq_len: int = seq_len
        self.key_dim: int = 128
        self.value_dim: int = 128
        self.attention_stacks: int = attention_stacks

        encoder_outputs = self.encoder_model.forward_intermediate(
            torch.zeros(1, *self.input_shape))
        self.encoder_output_shape = encoder_outputs[-1].shape

        self.memory_model = MemoryModelBase(
            encoder_model,
            mask_model,
            input_shape,
            activation=nn.Identity(), )

        att_input = torch.zeros(1, 1, *self.encoder_output_shape[1:]).permute(0, 1, 3, 4, 2)
        mem_input = torch.zeros(1, seq_len, *self.encoder_output_shape[1:]).permute(0, 1, 3, 4, 2)

        self.co_attention = nn.ModuleList()
        for i in range(self.attention_stacks):
            memory_attention = CoMemoryAttentionModule(
                (1, att_input.shape[2], att_input.shape[3], att_input.shape[4]),
                (1, mem_input.shape[2], mem_input.shape[3], mem_input.shape[4]),
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                use_norm=False,
                attention_drop_rate=0.0,
                use_positional_embedding=True,
                use_key_positional_embedding=True,
                attention_heads=8,
                use_qkv_embedding=False,
            )

            self.co_attention.append(CoAttentionModule(
                memory_attention,
                att_input.shape,
                mem_input.shape,
                self.key_dim,
                self.value_dim,
            ))

        decoder_inputs = [
            *encoder_outputs[::-1],
            torch.zeros(1, *self.input_shape)]

        decoder_input_channels = [output.shape[1] for output in decoder_inputs]

        self.decoder = UNetDecoder(
            [128, 64, 32, 16],
            decoder_input_channels,
            activation=nn.Hardswish(), )

    def forward(self, encoder_input, memory_images, memory_labels, mask):
        # Process the encoder input through the encoder model
        # encoder_output = self.encoder_model(encoder_input)
        encoder_outputs = self.encoder_model.forward_intermediate(
            encoder_input)

        # Process the memory images and labels through the mask model
        memory_output = self.memory_model(
            memory_images,
            memory_labels,
            mask)

        # Combine the encoder output and memory output using the memory attention layer
        attention_input = torch.permute(encoder_outputs[-1], (0, 2, 3, 1))
        attention_input = attention_input.unsqueeze(1)
        for co_attention in self.co_attention:
            memory_attention_output = co_attention(
                attention_input,
                memory_output,
                mask
            )
            attention_input = memory_attention_output

        memory_attention_output = torch.squeeze(memory_attention_output, 1)
        decoder_inputs = [
            memory_attention_output.permute(0, 3, 1, 2),
            *encoder_outputs[::-1][1:],
            encoder_input]

        decoder_output = self.decoder(decoder_inputs)

        return decoder_output



def create_combined_model(
        device_type,
        keras_weights_path,
        output_path,):

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
    activation = nn.Hardswish()
    use_features_activation = False

    pytorch_model = EfficientViT_B(
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
    ).to('cpu')

    mask_encoder = MaskEncoder(
        input_channels=1,
        output_channels=256,
    ).to('cpu')

    model = CombinedModel(
        pytorch_model,
        mask_encoder,
        (input_shape[2], input_shape[0], input_shape[1]),
        seq_len=5
    ).to(device)

    keras_model = create_model()
    keras_model.load_weights(keras_weights_path)

    load_keras_weights_to_pytorch_by_name(
        keras_model.get_layer("efficientvit"),
        model.encoder_model)

    load_memory_encoder_weights(
        keras_model.get_layer("memory_model_features_conv").get_layer("memory_encoder_layer"),
        model.memory_model)

    load_coattention_weights(
        keras_model.get_layer("co_attention_module"),
        model.co_attention[0])

    #load_coattention_weights(
    #    keras_model.get_layer("co_attention_module_1"),
    #    model.co_attention[1])

    load_keras_into_pytorch(
        keras_model.get_layer("u_net_decoder"),
        model.decoder,
        (lambda name: transform_layer_name(name)),
    )

    data_folder = 'test_images'
    images, masks = load_images_and_masks(data_folder)

    # Prepare data for Keras
    keras_images, keras_masks = prepare_data_for_keras(images, masks)

    # Prepare data for PyTorch
    pytorch_images, pytorch_masks = prepare_data_for_pytorch(images, masks)

    model.eval()
    pytorch_output = model(pytorch_images[0:1].cpu(),
                       torch.unsqueeze(pytorch_images[1:6],0).cpu(),
                       torch.unsqueeze(pytorch_masks[1:6],0).cpu(),
                       torch.ones((1, 5)).cpu())

    keras_output = keras_model.predict([keras_images[0:1],
                                        keras.ops.expand_dims(keras_images[1:6],0),
                                        keras.ops.expand_dims(keras_masks[1:6],0),
                                        np.ones((1, 5))])

    # Save each output as an image
    cv2.imwrite("pytorch_output.png", pytorch_output.permute(0, 2, 3, 1).detach().numpy()[0, :, :, 0] * 255)
    cv2.imwrite("keras_output.png", keras_output[0, :, :, 0] * 255)

    """
    assert np.testing.assert_allclose(
        pytorch_output.permute(0, 2, 3, 1).detach().numpy(),
        keras_output, atol=1e-5)
    """

    # JIT compile the model
    model = torch.jit.script(model.eval().cuda())
    model.save(output_path)

    return model

if __name__ == "__main__":
    create_combined_model(
        device_type='cuda',
        keras_weights_path="/mnt/c/Users/wanglab/Downloads/unet_efficientvit_unet_1.weights.h5",
        output_path="/mnt/c/Users/wanglab/Desktop/efficientvit_pytorch_cuda.pt"
    )
