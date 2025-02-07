import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional

from src.DeepLearningUtils.utils.model_conversion_helpers import load_keras_weights_to_pytorch_by_name

class MemoryModelBase(nn.Module):
    def __init__(self,
                 base_model: nn.Module,
                 mask_encoder: nn.Module,
                 input_shape: Tuple[int, int, int],
                 combine_operation='add',
                 activation=nn.Tanh(),
                 ):
        super(MemoryModelBase, self).__init__()
        self.base_model = base_model
        self.mask_encoder = mask_encoder

        self.height = input_shape[1]
        self.width = input_shape[2]

        # Get the channel count from the base model
        self.base_out_shape = base_model(
            torch.rand(1, *input_shape)).shape

        self.channel_count = self.base_out_shape[1]

        assert len(self.base_out_shape) == 4

        self.mask_out_shape = mask_encoder(
            torch.rand(1, 1, self.height, self.width)).shape

        assert self.channel_count == self.mask_out_shape[1]

        assert len(self.mask_out_shape) == 4

        if combine_operation == 'add':
            self.combine_layer = nn.Identity()
        elif combine_operation == 'dense':
            self.combine_layer = nn.Linear(
                in_features=self.channel_count * 2,
                out_features=self.channel_count)
        else:
            raise ValueError('combine_operation must be "add" or "dense"')

        self.activation = activation

    def forward(self,
                images: torch.Tensor,
                label_masks: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        #Images should be of shape (batch, time, height, width, channels)
        #Label masks should be of shape (batch, time, height, width, channels)
        #Mask should be of shape (batch, time)

        assert images.shape[0] == label_masks.shape[0] == mask.shape[0]
        assert images.shape[1] == label_masks.shape[1] == mask.shape[1]
        assert len(images.shape) == 5
        assert len(label_masks.shape) == 5, f"Label masks shape: {label_masks.shape}"

        # Apply the base model to the sequence of images
        base_model_outputs = []
        for i, image in enumerate(images): #Across batch dimension
            #if mask[i] == 1:
            base_model_output = self.base_model(image)
            #else:
                #base_model_output = torch.zeros(self.base_out_shape)
            base_model_outputs.append(base_model_output)
        base_model_outputs = torch.stack(base_model_outputs, dim=0)

        # Apply the mask encoder to the sequence of masks
        mask_encoder_outputs = []
        for mask in label_masks:
            mask_encoder_output = self.mask_encoder(mask)
            mask_encoder_outputs.append(mask_encoder_output)
        mask_encoder_outputs = torch.stack(mask_encoder_outputs, dim=0)

        # Combine the outputs at the low resolution, high channel count encoded space
        combined_outputs = base_model_outputs + mask_encoder_outputs

        # Move features to last dimension
        combined_outputs = combined_outputs.permute(0, 1, 3, 4, 2).contiguous()

        synthesized_images = self.activation(combined_outputs)

        return synthesized_images

def load_memory_encoder_weights(keras_model, pytorch_model):
    # Load weights from Keras to PyTorch
    load_keras_weights_to_pytorch_by_name(keras_model.base_memory_model, pytorch_model.base_model)
    load_keras_weights_to_pytorch_by_name(keras_model.mask_encoder_model, pytorch_model.mask_encoder)