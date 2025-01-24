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

Modified by Paul Thompson 2024

"""

from Layers.Blur2D.blur2d_keras import Blur2D

import keras 
import numpy as np


def mb_conv(
    inputs,
    output_channel,
    shortcut=True,
    strides=1,
    expansion=4,
    is_fused=False,
    use_bias=False,
    use_norm=False,
    use_output_norm=False,
    initializer=None,
    drop_rate=0,
    anti_aliasing=False,
    activation="keras.activations.hard_silu",
    name=""
):

    activation_func = eval(activation)
    if initializer is None:
        initializer = keras.initializers.GlorotUniform()

    input_channel = inputs.shape[-1]
    if is_fused:
        if anti_aliasing and strides > 1:
            nn = keras.layers.Conv2D(
                int(input_channel * expansion),
                3,
                strides=1,
                padding="same",
                kernel_initializer=initializer,
                name=name and name + "expand_conv")(inputs)
            nn = Blur2D(
                kernel_size=5,
                stride=strides,
                kernel_type="Binomial",
                padding='same')(nn)
        else:
            nn = keras.layers.Conv2D(
                int(input_channel * expansion),
                3,
                strides=strides,
                padding="same",
                kernel_initializer=initializer,
                name=name and name + "expand_conv")(inputs)

        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name=name + "expand_bn")(nn) 
        nn = keras.layers.Activation(activation_func, name='{}_activation'.format(name))(nn)
    elif expansion > 1:
        nn = keras.layers.Conv2D(
            int(input_channel * expansion),
            1,
            strides=1,
            use_bias=True,
            kernel_initializer=initializer,
            name=name + "expand_conv")(inputs)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name=name + "expand_bn")(nn)
        nn = keras.layers.Activation(activation_func, name='{}_activation'.format(name))(nn)
    else:
        nn = inputs

    if not is_fused:
        if anti_aliasing and strides > 1:
            nn = keras.layers.DepthwiseConv2D(
                3,
                strides=1,
                use_bias=True,
                padding="same",
                depthwise_initializer=initializer,
                name=name + "dw_conv")(nn)
            nn = Blur2D(
                kernel_size=5, 
                stride=strides,
                kernel_type="Binomial",
                padding='same')(nn)
        else:
            nn = keras.layers.DepthwiseConv2D(
                3,
                strides=strides,
                use_bias=True,
                padding="same",
                depthwise_initializer=initializer,
                name=name + "dw_conv")(nn)
        if use_norm:
            nn = keras.layers.BatchNormalization(momentum=0.9, name=name + "dw_bn")(nn)
        nn = keras.layers.Activation(activation_func, name='{}_dw_activation'.format(name))(nn)

    pw_kernel_size = 3 if is_fused and expansion == 1 else 1

    nn = keras.layers.Conv2D(
        output_channel,
        pw_kernel_size,
        strides=1,
        padding="same",
        use_bias=True,
        kernel_initializer=initializer,
        name=name + "pw_conv")(nn)
    
    if use_output_norm and shortcut:
        nn = keras.layers.BatchNormalization(momentum=0.9, gamma_initializer="zeros", name=name + "pw_bn")(nn)
        #nn = keras.layers.BatchNormalization(momentum=0.9, name=name + "pw_bn")(nn)
    elif use_output_norm:
        nn = keras.layers.BatchNormalization(momentum=0.9, name=name + "pw_bn")(nn)
    nn = keras.layers.Dropout(rate=drop_rate, name=name + "dropout")(nn)

    return keras.layers.Add(name=name + "output")([inputs, nn]) if shortcut else keras.layers.Activation("linear", name=name + "output")(nn)

