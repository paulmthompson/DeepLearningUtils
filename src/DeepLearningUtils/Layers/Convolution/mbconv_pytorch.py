

"""
   Copyright [2023] [Han Cai]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Modified by Paul Thompson 2024

"""

from src.DeepLearningUtils.Layers.Blur2D.blur2d_pytorch import Blur2D
from src.DeepLearningUtils.Layers.Convolution.conv2d_same_pytorch import Conv2dSame

import torch
import torch.nn as nn

import collections


class MBConv(nn.Module):
    def __init__(self, 
                 input_channel,
                 output_channel,
                 shortcut=True,
                 strides=1,
                 expansion=4,
                 is_fused=False,
                 use_bias=True,
                 use_norm=False,
                 use_output_norm=False,
                 initializer=None,
                 drop_rate=0.0, 
                 anti_aliasing=False,
                 activation=nn.Hardswish(),
                 name=""):
        super(MBConv, self).__init__()
        self.shortcut = shortcut
        self.strides = strides
        self.expansion = expansion
        self.is_fused = is_fused
        self.use_norm = use_norm
        self.use_output_norm = use_output_norm
        self.drop_rate = drop_rate
        self.anti_aliasing = anti_aliasing
        self.activation = activation

        if initializer is None:
            initializer = nn.init.xavier_uniform_

        self.expand_block = torch.nn.Sequential(collections.OrderedDict())
        self.dw_block = torch.nn.Sequential(collections.OrderedDict())
        self.pw_block = torch.nn.Sequential(collections.OrderedDict())

        if is_fused:
            if anti_aliasing and strides > 1:
                self.expand_block.add_module(
                    name + "expand_conv", Conv2dSame(
                        input_channel,
                        int(input_channel * expansion),
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        bias=True,
                        ))
                self.expand_block.add_module(
                    name + "expand_conv_blur", Blur2D(
                        kernel_size=5,
                        kernel_type="Binomial",
                        stride=strides,
                        padding='same'
                        ))
            else:
                self.expand_block.append(
                    name + "expand_conv", Conv2dSame(
                        input_channel,
                        int(input_channel * expansion),
                        kernel_size=(3, 3),
                        stride=(strides, strides),
                        bias=True,
                        ))
            
            conv_output_channels = int(input_channel * expansion)
            if use_norm:
                self.expand_block.add_module(
                    name + "expand_bn", nn.BatchNorm2d(conv_output_channels, eps=1e-3))
            self.expand_block.add_module(f"{name}_activation", activation)
        elif expansion > 1:
            self.expand_block.add_module(
                name + "expand_conv", torch.nn.Conv2d(
                    input_channel,
                    int(input_channel * expansion),
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    bias=True,
                    ))
            if use_norm:
                self.expand_block.add_module(
                name + "expand_bn", nn.BatchNorm2d(int(input_channel * expansion), eps=1e-3))
            self.expand_block.add_module(f"{name}_activation", activation)

            conv_output_channels = int(input_channel * expansion)
        else:
            conv_output_channels = input_channel

        if not is_fused:
            if anti_aliasing and strides > 1:
                self.dw_block.add_module(
                    name + "dw_conv", Conv2dSame(
                        conv_output_channels,
                        conv_output_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        groups=conv_output_channels,
                        bias=True,
                        #padding="same"
                        ))
                self.dw_block.add_module(
                    name + "dw_conv_blur", Blur2D(
                        kernel_size=5,
                        stride=strides,
                        kernel_type="Binomial",
                        padding='same'))
            else:
                self.dw_block.add_module(
                    name + "dw_conv", Conv2dSame(
                        conv_output_channels,
                        conv_output_channels,
                        kernel_size=(3, 3),
                        stride=(strides, strides),
                        groups=conv_output_channels,
                        bias=True,
                        ))
            if use_norm:
                self.dw_block.add_module(
                    name + "dw_bn", nn.BatchNorm2d(conv_output_channels, eps=1e-3))
            self.dw_block.add_module(f"{name}_dw_activation", activation)

        pw_kernel_size = 3 if is_fused and expansion == 1 else 1
        self.pw_block.add_module(
            name + "pw_conv", Conv2dSame(
                conv_output_channels,
                output_channel,
                kernel_size=(pw_kernel_size, pw_kernel_size),
                stride=(1, 1),
                bias=True,
                ))
        if use_output_norm:
            self.pw_block.add_module(
                name + "pw_bn", nn.BatchNorm2d(output_channel, eps=1e-3))
        
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, input):
        x = input

        x = self.expand_block(x)
        x = self.dw_block(x)
        x = self.pw_block(x)

        x = self.dropout(x)

        if self.shortcut:
            return x + input
        else:
            return x