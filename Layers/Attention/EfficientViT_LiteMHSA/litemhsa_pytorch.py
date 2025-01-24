
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

from Layers.Convolution.conv2d_same_pytorch import Conv2dSame

import torch
import torch.nn as nn
import torch.nn.functional as F

import collections

from typing import List, Optional, Tuple

class LiteMHSA(nn.Module):
    def __init__(self,
                 input_channel,
                 num_heads=8,
                 key_dim=16,
                 sr_ratio=5,
                 qkv_bias=True, 
                 out_shape=None, 
                 out_bias=False,
                 use_norm=True,
                 dropout=0.0,
                 initializer=None,
                 activation=nn.ReLU(),
                 name=None):
        super(LiteMHSA, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim if key_dim > 0 else input_channel // num_heads
        self.out_shape = input_channel if out_shape is None else out_shape
        self.emb_dim = num_heads * self.key_dim
        self.use_norm = use_norm
        self.activation = activation

        if initializer is None:
            initializer = nn.init.xavier_uniform_

        self.qkv_conv = torch.nn.Sequential(collections.OrderedDict())
        self.qkv_dw_conv = torch.nn.Sequential(collections.OrderedDict())
        self.qkv_pw_conv = torch.nn.Sequential(collections.OrderedDict())

        self.qkv_conv.add_module(
            name + "qkv_conv", torch.nn.Conv2d(
                input_channel,
                self.emb_dim * 3,
                kernel_size=(1, 1),
                bias=qkv_bias))
        self.qkv_dw_conv.add_module(
            name + "qkv_dw_conv", Conv2dSame(
                self.emb_dim * 3,
                self.emb_dim * 3,
                kernel_size=(sr_ratio, sr_ratio),
                groups=self.emb_dim * 3,
                bias=qkv_bias))
        self.qkv_pw_conv.add_module(
            name + "qkv_pw_conv", torch.nn.Conv2d(
                self.emb_dim * 3,
                self.emb_dim * 3,
                kernel_size=(1, 1),
                groups=3 * num_heads,
                bias=qkv_bias))
        
        self.out_block = torch.nn.Sequential(collections.OrderedDict())
        self.out_block.add_module(
            name + "out_conv", torch.nn.Conv2d(
                self.emb_dim * 2,
                self.out_shape,
                kernel_size=(1, 1),
                bias=True))
        if use_norm:
            self.out_block.add_module(name + "out_bn", nn.BatchNorm2d(self.out_shape,eps=1e-3))

    def forward(self, input):
        batch_size, _, height, width = input.size()
        
        qkv = self.qkv_conv(input) # embed_dim * 3

        sr_qkv = self.qkv_dw_conv(qkv) #embed_dim * 3
        sr_qkv = self.qkv_pw_conv(sr_qkv) #embed_dim * 3
        qkv = torch.cat([qkv, sr_qkv], dim=1) #B x embed_dim * 3 * 2 x H x W

        qkv = torch.reshape(
            qkv,
            (
                batch_size,
                -1,
                3 * self.key_dim,
                height * width,
            ),
        )
        query, key, value = torch.split(qkv, self.key_dim, dim=-2)
        
        query = self.activation(query) # B x heads * key_dim x H * W
        key = self.activation(key) # B x heads * key_dim x H * W

        query_key = torch.matmul(query.transpose(-2, -1), key)
        scale = torch.sum(query_key, dim=-1, keepdim=True)
        attention_output = torch.matmul(query_key, value.transpose(-2, -1)) / (scale + 1e-7)
        # attention output (B x heads x h*w x key_dim)

        output = attention_output.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, height, width)
        output = self.out_block(output)
        output = output + input
        return output