

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
import numpy as np
import math

class Blur2D(nn.Module):
    def __init__(self, kernel_size=5, stride=2, kernel_type="Rect", padding='same'):
        super(Blur2D, self).__init__()
        
        self.kernel_size = kernel_size
        if kernel_type == "Rect":

            self.kernel2d = torch.ones((self.kernel_size, self.kernel_size))
            self.kernel2d /= self.kernel2d.sum()

        elif kernel_type == "Triangle":
            assert kernel_size > 2, "Kernel size must be greater than 2 for Triangle kernel"
            if kernel_size % 2 == 0:
                kernel_base = np.arange(1, (kernel_size + 1) // 2 + 1)
                kernel_base = np.concatenate([kernel_base, np.flip(kernel_base)])
            else:
                kernel_base = np.arange(1, (kernel_size + 1) // 2 + 1)
                kernel_base = np.concatenate([kernel_base, np.flip(kernel_base[:-1])])
            self.kernel2d = np.outer(kernel_base, kernel_base)
            self.kernel2d = torch.tensor(self.kernel2d, dtype=torch.float32)
            self.kernel2d /= self.kernel2d.sum()

        elif kernel_type == "Binomial":
            assert kernel_size == 5, "Binomial kernel only supports kernel size of 5"
            kernel = np.array([1, 4, 6, 4, 1])
            self.kernel2d = np.outer(kernel, kernel)
            self.kernel2d = torch.tensor(self.kernel2d, dtype=torch.float32)
            self.kernel2d /= self.kernel2d.sum()
        else:
            raise ValueError("Kernel type must be either Rect, Triangle, or Binomial")

        self.kernel2d = self.kernel2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        self.stride = stride
        self.padding = padding

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * 1 + 1 - i, 0)

    def forward(self, x):
        
        batch_size, channels, height, width = x.size()
        kernel = self.kernel2d.repeat(channels, 1, 1, 1).to(x.device)

        if self.padding == 'same':
            pad_h = self.calc_same_pad(i=height, k=self.kernel2d.shape[2], s=self.stride)
            pad_w = self.calc_same_pad(i=width, k=self.kernel2d.shape[3], s=self.stride)

            print('pad_h:', pad_h)
            print('pad_w:', pad_w)

            if pad_h > 0 or pad_w > 0:
                x = F.pad(
                    x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                )
            padding = 0
        else:
            padding = self.padding

        # Apply depthwise convolution for blurring
        blurred = F.conv2d(x, kernel, stride=self.stride, padding=padding, groups=channels)
        return blurred