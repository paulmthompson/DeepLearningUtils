"""
MIT License

Copyright (c) 2021 leondgarse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is a keras implementation of EfficientViT.
The original pytorch implementation can be found here:
https://github.com/mit-han-lab/efficientvit

This was modified October 2024 by Paul Thompson,
mostly to make it Keras3 compatible.

"""

from Layers.Blur2D.blur2d_keras import Blur2D
from Layers.Convolution.mbconv_keras import mb_conv
from Layers.Attention.EfficientViT_LiteMHSA.litemhsa_keras import lite_mhsa

import keras


def EfficientViT_B(
    num_blocks=[2, 2, 3, 3],
    out_channels=[16, 32, 64, 128],
    stem_width=8,
    block_types=["conv", "conv", "transform", "transform"],
    expansions=4,  # int or list, each element in list can also be an int or list of int
    is_fused=False,  # True for L models, False for B models
    head_dimension=16,  # `num_heads = channels // head_dimension`
    output_filters=1024,
    input_shape=(224, 224, 3),
    activation="keras.activations.relu",  # "keras.activations.hard_silu" is in the paper, but i find poor performance
    drop_connect_rate=0,
    dropout=0,
    use_norm=True,
    initializer=None,
    anti_aliasing=False,
    model_name="efficientvit",
    kwargs=None
):
    
    if initializer is None:
        initializer = keras.initializers.GlorotUniform()

    inputs = keras.layers.Input(input_shape)
    is_fused = is_fused if isinstance(is_fused, (list, tuple)) else ([is_fused] * len(num_blocks))

    activation_func = eval(activation)

    """ stage 0, Stem_stage """
    if anti_aliasing:
        nn = keras.layers.Conv2D(
            stem_width,
            3,
            strides=1,
            padding="same",
            kernel_initializer=initializer,
            name="stem_conv")(inputs)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name="stem_bn")(nn)
        nn = keras.layers.Activation(activation_func, name="stem_activation_")(nn)
        nn = Blur2D(kernel_size=5, stride=2, kernel_type="Binomial", padding='same')(nn)
    else:
        nn = keras.layers.Conv2D(
            stem_width,
            3,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            name="stem_conv")(inputs)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name="stem_bn")(nn)
        nn = keras.layers.Activation(activation_func, name="stem_activation_")(nn)

    nn = mb_conv(
        nn,
        stem_width,
        shortcut=True,
        expansion=1,
        is_fused=is_fused[0],
        use_norm=use_norm,
        use_output_norm=use_norm,
        activation=activation,
        initializer=initializer,
        anti_aliasing=anti_aliasing,
        name="stem_MB_")

    """ stage [1, 2, 3, 4] """ # 1/4, 1/8, 1/16, 1/32
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        is_conv_block = True if block_type[0].lower() == "c" else False
        cur_expansions = expansions[stack_id] if isinstance(expansions, (list, tuple)) else expansions

        block_use_bias, block_use_norm = (True, False) if stack_id >= 2 else (False, True)  # fewer_norm

        block_anti_aliasing = anti_aliasing if stack_id <= 2 else False

        if not use_norm:
            block_use_norm = False
            block_use_bias = True

        cur_is_fused = is_fused[stack_id]
        for block_id in range(num_block):

            name = "stack_{}_block_{}_".format(stack_id + 1, block_id + 1)
            stride = 2 if block_id == 0 else 1
            shortcut = False if block_id == 0 else True
            cur_expansion = cur_expansions[block_id] if isinstance(cur_expansions, (list, tuple)) else cur_expansions

            block_drop_rate = drop_connect_rate * global_block_id / total_blocks

            if is_conv_block or block_id == 0:
                cur_name = (name + "downsample_") if stride > 1 else name
                nn = mb_conv(
                    nn,
                    out_channel,
                    shortcut=shortcut,
                    strides=stride,
                    expansion=cur_expansion,
                    is_fused=cur_is_fused,
                    use_bias=block_use_bias,
                    use_norm=block_use_norm,
                    use_output_norm=use_norm,
                    drop_rate=block_drop_rate,
                    activation=activation,
                    initializer=initializer,
                    anti_aliasing=block_anti_aliasing,
                    name=cur_name)
            else:
                num_heads = out_channel // head_dimension
                attn = lite_mhsa(
                    nn,
                    num_heads=num_heads,
                    key_dim=head_dimension,
                    sr_ratio=5,
                    use_norm=use_norm,
                    initializer=initializer,
                    name=name + "attn_")

                nn = nn + attn

                nn = mb_conv(
                    nn,
                    out_channel,
                    shortcut=shortcut,
                    strides=stride,
                    expansion=cur_expansion,
                    is_fused=cur_is_fused,
                    use_bias=block_use_bias,
                    use_norm=block_use_norm,
                    use_output_norm=use_norm,
                    drop_rate=block_drop_rate,
                    activation=activation,
                    initializer=initializer,
                    anti_aliasing=block_anti_aliasing,
                    name=name)
            global_block_id += 1

    if output_filters > 0:
        nn = keras.layers.Conv2D(
            output_filters,
            1,
            kernel_initializer=initializer,
            name="features_conv")(nn)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name="features_bn")(nn)
        nn = keras.layers.Activation(activation_func, name="features_activation")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)

    return model