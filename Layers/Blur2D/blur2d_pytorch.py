

"""
This implements a blur layer to be used with max pooling or convolutions for anti-aliasing.

Reference:
Zhang, R., 2019. Making Convolutional Networks Shift-Invariant Again. 
https://doi.org/10.48550/arXiv.1904.11486

Github repository for original implementation:
https://github.com/adobe/antialiased-cnns

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Blur2D(nn.Module):
    def __init__(self, kernel_size=5, stride=2, padding='same'):
        super(Blur2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define a simple averaging kernel
        self.kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size ** 2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        kernel = self.kernel.repeat(channels, 1, 1, 1).to(x.device)

        if self.padding == 'same':
            padding = self.kernel_size // 2
        else:
            padding = 0

        # Apply depthwise convolution for blurring
        blurred = F.conv2d(x, kernel, stride=self.stride, padding=padding, groups=channels)
        return blurred