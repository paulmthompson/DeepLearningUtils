import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List

from src.DeepLearningUtils.Layers.Convolution.conv2d_same_pytorch import Conv2dSame


class UNetDecoder(nn.Module):
    def __init__(self,
                 filter_sizes,
                 input_sizes,
                 activation=nn.ReLU()):
        super(UNetDecoder, self).__init__()
        self.filter_sizes = filter_sizes
        self.activation = activation

        # Check that input sizes length is 1 greater than filter sizes length
        assert len(input_sizes) == len(filter_sizes) + 1

        self.levels = nn.ModuleList()
        previous_filters = input_sizes[0]
        for i, filters in enumerate(filter_sizes):
            in_channels = previous_filters + input_sizes[i + 1]
            self.levels.append(UNetDecoderLevel(
                in_channels,
                filters,
                activation))
            previous_filters = filters

        self.output_conv = nn.Conv2d(
            in_channels=filter_sizes[-1],
            out_channels=1,
            kernel_size=(1, 1),
            bias=True)
        nn.init.constant_(self.output_conv.bias, -5.0)

    def forward(self, inputs: List[torch.Tensor]):
        x = inputs[0]
        encoder_outputs = inputs[1:]

        for i, level in enumerate(self.levels):
            x = level(x, encoder_outputs[i])

        x = self.output_conv(x)
        x = torch.sigmoid(x)

        return x


class UNetDecoderLevel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=nn.ReLU()):
        super(UNetDecoderLevel, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False)
        self.conv = Conv2dSame(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            bias=True)
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=1e-3)
        self.activation = activation

    def forward(self, x, encoder_output):
        x = self.upsample(x)
        x = torch.cat([x, encoder_output], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x