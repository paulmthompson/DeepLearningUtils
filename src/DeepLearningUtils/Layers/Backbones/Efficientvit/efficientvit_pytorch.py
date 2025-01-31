
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
from src.DeepLearningUtils.Layers.Attention.EfficientViT_LiteMHSA.litemhsa_pytorch import LiteMHSA
from src.DeepLearningUtils.Layers.Blur2D.blur2d_pytorch import Blur2D
from src.DeepLearningUtils.Layers.Convolution.conv2d_same_pytorch import Conv2dSame
from src.DeepLearningUtils.Layers.Convolution.mbconv_pytorch import MBConv

import torch
import torch.nn as nn

from typing import List, Optional, Tuple


class EfficientViT_B(nn.Module):
    def __init__(self,
                 num_blocks=[2, 2, 3, 3],
                 out_channels=[16, 32, 64, 128],
                 stem_width=8, 
                 block_types=["conv", "conv", "transform", "transform"],
                 expansions=4,
                 is_fused=False,
                 head_dimension=16,
                 output_filters=1024,
                 input_shape=(3, 224, 224), 
                 activation=nn.ReLU(),
                 drop_connect_rate=0,
                 dropout=0.0,
                 use_norm=True,
                 initializer=None,
                 anti_aliasing=False,
                 upsample_levels=0,
                 use_features_activation=False,
                 model_name="efficientvit",
                 kwargs=None):
        super(EfficientViT_B, self).__init__()
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.stem_width = stem_width
        self.block_types = block_types
        self.expansions = expansions
        self.is_fused = is_fused if isinstance(is_fused, (list, tuple)) else [is_fused] * len(num_blocks)
        self.head_dimension = head_dimension
        self.output_filters = output_filters
        self.activation = activation
        self.drop_connect_rate = drop_connect_rate
        self.dropout = dropout
        self.use_norm = use_norm
        self.anti_aliasing = anti_aliasing
        self.upsample_levels = upsample_levels
        self.stack_output_indices = []
        self.use_features_activation = use_features_activation

        for i in range(len(num_blocks)):
            self.stack_output_indices.append(sum(num_blocks[:i + 1]))

        if initializer is None:
            initializer = nn.init.xavier_uniform_

        input_channels = input_shape[0]

        if anti_aliasing:
            self.stem_conv = Conv2dSame(
                input_channels,
                stem_width,
                kernel_size=(3, 3),
                stride=(1, 1))
            self.stem_bn = nn.BatchNorm2d(stem_width, eps=1e-3) if use_norm else nn.Identity()
            self.stem_activation = activation
            self.stem_blur = Blur2D(kernel_size=5, stride=2, kernel_type="Binomial", padding='same')
        else:
            self.stem_conv = Conv2dSame(
                input_channels,
                stem_width,
                kernel_size=(3, 3),
                stride=(2, 2))
            self.stem_bn = nn.BatchNorm2d(stem_width, eps=1e-3) if use_norm else nn.Identity()
            self.stem_activation = activation
            self.stem_blur = nn.Identity()

        self.stem_mb_conv = MBConv(
            stem_width,
            stem_width,
            shortcut=True,
            expansion=1,
            is_fused=self.is_fused[0],
            use_norm=use_norm,
            use_output_norm=use_norm,
            activation=activation,
            initializer=initializer,
            anti_aliasing=anti_aliasing,
            name="stem_MB_")

        self.blocks = nn.Sequential(*self._make_blocks(stem_width))

        self.upsample_blocks = nn.ModuleList()


        feature_conv_input = out_channels[-1]
        for i in range(upsample_levels):
            # upsample bilinearly and concatenate with the previous feature map

            self.upsample_blocks.append(nn.Sequential(
                torch.nn.UpsamplingBilinear2d(scale_factor=2)))
            feature_conv_input += out_channels[-1]

        if output_filters > 0:
            self.features_conv = torch.nn.Conv2d(
                feature_conv_input,
                output_filters,
                kernel_size=(1, 1),
                stride=(1, 1))
            use_feature_norm = use_norm and use_features_activation
            self.features_bn = nn.BatchNorm2d(output_filters,eps=1e-3) if use_feature_norm else nn.Identity()
            self.features_activation = activation if use_features_activation else nn.Identity()

    def _make_blocks(self, block_input_channels):
        blocks = []
        total_blocks = sum(self.num_blocks)
        global_block_id = 0
  
        for stack_id, (num_block, out_channel, block_type) in enumerate(zip(self.num_blocks, self.out_channels, self.block_types)):
        
            is_conv_block = block_type[0].lower() == "c"
            cur_expansions = self.expansions[stack_id] if isinstance(self.expansions, (list, tuple)) else self.expansions

            block_anti_aliasing = self.anti_aliasing if stack_id <= 2 else False

            block_use_bias, block_use_norm = (True, False) if stack_id >= 2 else (False, True)
            if not self.use_norm:
                block_use_norm = False
                block_use_bias = True

            cur_is_fused = self.is_fused[stack_id]
            for block_id in range(num_block):
                name = f"stack_{stack_id + 1}_block_{block_id + 1}_"
                this_block = []
                stride = 2 if block_id == 0 else 1
                shortcut = False if block_id == 0 else True
                cur_expansion = cur_expansions[block_id] if isinstance(cur_expansions, (list, tuple)) else cur_expansions

                block_drop_rate = self.drop_connect_rate * global_block_id / total_blocks

                if is_conv_block or block_id == 0:
                    cur_name = f"{name}downsample_" if stride > 1 else name
                    this_block.append(MBConv(
                        block_input_channels,
                        out_channel,
                        shortcut=shortcut,
                        strides=stride,
                        expansion=cur_expansion,
                        is_fused=cur_is_fused,
                        use_bias=block_use_bias,
                        use_norm=block_use_norm,
                        use_output_norm=self.use_norm,
                        drop_rate=block_drop_rate,
                        activation=self.activation,
                        initializer=None,
                        anti_aliasing=block_anti_aliasing,
                        name=cur_name))
                else:
                    num_heads = out_channel // self.head_dimension
                    this_block.append(LiteMHSA(
                        block_input_channels,
                        num_heads=num_heads,
                        key_dim=self.head_dimension,
                        sr_ratio=5,
                        use_norm=self.use_norm,
                        initializer=None,
                        name=f"{name}attn_"))
  
                    this_block.append(MBConv(
                        block_input_channels,
                        out_channel,
                        shortcut=shortcut,
                        strides=stride,
                        expansion=cur_expansion,
                        is_fused=cur_is_fused,
                        use_bias=block_use_bias,
                        use_norm=block_use_norm,
                        use_output_norm=self.use_norm,
                        drop_rate=block_drop_rate,
                        activation=self.activation,
                        initializer=None,
                        anti_aliasing=block_anti_aliasing,
                        name=name))

                block_input_channels = out_channel
                global_block_id += 1
                blocks.append(nn.Sequential(*this_block))

        return blocks

    def forward(self,
                x: torch.Tensor,
                intermediate_outputs: Optional[List[int]]=None):

        assert x.dim() == 4, "Input tensor must be 4-dimensional"

        outputs = []

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_activation(x)
        x = self.stem_blur(x)
        x = self.stem_mb_conv(x)
        outputs.append(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            outputs.append(x)

        for i, upsample_block in enumerate(self.upsample_blocks):
            x = upsample_block(x)
            x = torch.cat([x, outputs[self.stack_output_indices[-(2 + i)]]], dim=1)

        if self.output_filters > 0:
            x = self.features_conv(x)
            x = self.features_bn(x)
            x = self.features_activation(x)

        if intermediate_outputs is None:
            return [x]
        else:
            outputs.append(x)
            return outputs
        
    def forward_intermediate(
            self,
            x,
            intermediate_outputs: Tuple[int, int, int]):

        outputs: List[torch.Tensor] = []

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_activation(x)
        x = self.stem_blur(x)
        x = self.stem_mb_conv(x)
        outputs.append(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in intermediate_outputs:
                outputs.append(x)

        if self.output_filters[0] > 0:
            x = self.features_conv(x)
            x = self.features_bn(x)
            x = self.features_activation(x)

        outputs.append(x)
        return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
