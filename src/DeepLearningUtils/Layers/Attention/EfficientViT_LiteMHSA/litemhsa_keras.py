
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


import keras

#https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/efficientvit/efficientvit_b.py

def lite_mhsa(inputs,
              num_heads=8,
              key_dim=16,
              sr_ratio=5,
              qkv_bias=True,  # was False
              out_shape=None,
              out_bias=False,
              use_norm=True,
              dropout=0,
              initializer=None,
              activation="keras.activations.relu",
              name=None
              ):
    
    if initializer is None:
        initializer = keras.initializers.GlorotUniform()

    input_channel = inputs.shape[-1]
    height, width = inputs.shape[1:-1]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    out_shape = input_channel if out_shape is None else out_shape
    emb_dim = num_heads * key_dim

    # query = layers.Dense(emb_dim, use_bias=qkv_bias, name=name and name + "query")(inputs)
    qkv = keras.layers.Conv2D(
        emb_dim * 3,
        1,
        use_bias=qkv_bias,
        kernel_initializer=initializer,
        name=name and name + "qkv_conv")(inputs)
    sr_qkv = keras.layers.DepthwiseConv2D(
        kernel_size=sr_ratio,
        use_bias=qkv_bias,
        padding="same",
        depthwise_initializer=initializer,
        name=name and name + "qkv_dw_conv")(qkv)
    sr_qkv = keras.layers.Conv2D(
        emb_dim * 3,
        1,
        use_bias=qkv_bias,
        groups=3 * num_heads,
        kernel_initializer=initializer,
        name=name and name + "qkv_pw_conv")(sr_qkv)
    qkv = keras.ops.concatenate([qkv, sr_qkv], axis=-1)


    qkv = keras.ops.reshape(qkv, [-1, height * width, qkv.shape[-1] // (3 * key_dim), 3 * key_dim])
    query, key, value = keras.ops.split(qkv, 3, axis=-1)
    query = keras.ops.transpose(query, [0, 2, 1, 3]) # [batch, num_heads, q_blocks, key_dim]
    key = keras.ops.transpose(key, [0, 2, 3, 1]) # [batch, num_heads, key_dim, k_blocks]
    value = keras.ops.transpose(value, [0, 2, 1, 3])

    activation_func = eval(activation)
    query = keras.layers.Activation(activation_func, name='{}_query_activation'.format(name))(query)
    key = keras.layers.Activation(activation_func, name='{}_key_activation'.format(name))(key)

    query_key = query @ key
    scale = keras.ops.sum(query_key, axis=-1, keepdims=True)
    attention_output = query_key @ value / (scale + 1e-7)  # 1e-7 for also working on float16
    # print(f">>>> {inputs.shape = }, {emb_dim = }, {num_heads = }, {key_dim = }, {attention_output.shape = }")

    output = keras.ops.transpose(attention_output, [0, 2, 1, 3])  # [batch, q_blocks, num_heads * 2, key_dim]
    output = keras.ops.reshape(output, [-1, height, width, output.shape[2] * output.shape[3]])

    # print(f">>>> {output.shape = }")
    output = keras.layers.Conv2D(
        out_shape,
        1,
        use_bias=True,
        kernel_initializer=initializer,
        name=name and name + "out_conv")(output)
    if use_norm:
        output = keras.layers.BatchNormalization(
            momentum=0.9,
            name=name and name + "out_bn")(output)
    return output
