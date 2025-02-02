
from src.DeepLearningUtils.Layers.LayerNorm2D.layernorm2d_pytorch import LayerNorm2d
from src.DeepLearningUtils.Layers.Convolution.conv2d_same_pytorch import Conv2dSame
from src.DeepLearningUtils.Layers.Blur2D.blur2d_pytorch import Blur2D


import torch
from torch import nn

class MaskEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, anti_aliasing=True, use_norm=True):
        super(MaskEncoder, self).__init__()
        self.anti_aliasing = anti_aliasing
        self.output_channels = output_channels

        if anti_aliasing:
            self.conv1 = Conv2dSame(input_channels, 4, kernel_size=2, stride=1)
            self.blur1 = Blur2D(5, 2, "Binomial", padding="same")
        else:
            self.conv1 = Conv2dSame(input_channels, 4, kernel_size=2, stride=2)
            self.blur1 = nn.Identity()

        self.norm1 = LayerNorm2d(4) if use_norm else nn.Identity()
        self.activation1 = nn.Hardswish()

        if anti_aliasing:
            self.conv2 = Conv2dSame(4, 64, kernel_size=4, stride=1)
            self.blur2_1 = Blur2D(3, 2, "Triangle", padding="same")
            self.blur2_2 = Blur2D(3, 2, "Triangle", padding="same")
        else:
            self.conv2 = Conv2dSame(4, 64, kernel_size=4, stride=4)
            self.blur2_1 = nn.Identity()
            self.blur2_2 = nn.Identity()

        self.norm2 = LayerNorm2d(64) if use_norm else nn.Identity()
        self.activation2 = nn.Hardswish()

        self.conv3 = Conv2dSame(64, 256, kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.blur1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.blur2_1(x)
        x = self.blur2_2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        x = self.conv3(x)

        return x


def load_mask_encoder_weights(keras_model, pytorch_model):
    # Load weights from Keras to PyTorch
    for keras_layer, pytorch_layer in zip(keras_model.layers, pytorch_model.children()):
        if isinstance(pytorch_layer, nn.Conv2d):
            pytorch_layer.weight.data = torch.tensor(keras_layer.get_weights()[0].transpose(3, 2, 0, 1))
            if len(keras_layer.get_weights()) > 1:
                pytorch_layer.bias.data = torch.tensor(keras_layer.get_weights()[1])
        elif isinstance(pytorch_layer, LayerNorm2d):
            pytorch_layer.weight.data = torch.tensor(keras_layer.gamma.numpy())
            pytorch_layer.bias.data = torch.tensor(keras_layer.beta.numpy())