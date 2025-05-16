
import keras
import tensorflow as tf


class PixelShuffleUpsampleResidual(keras.layers.Layer):
    def __init__(self,
                 out_channels,
                 scale=2,
                 activation='keras.activations.relu'):
        super(PixelShuffleUpsampleResidual, self).__init__()
        self.out_channels = out_channels
        self.scale = scale
        self.out_ratio = scale**2
        self.activation = activation

        self.conv_upsample = ConvPixelShuffleUpsample(
            self.out_channels,
            self.scale,
            self.activation
            )
        self.duplicate_upsample = PixelShuffleUpsample(self.out_channels, self.scale)

    def build(self, inputs):
        self.in_channels = inputs[-1]

    def call(self, inputs):
        x = self.conv_upsample(inputs)
        x = keras.layers.Add()([x, self.duplicate_upsample(inputs)])
        return x


class PixelShuffleUpsample(keras.layers.Layer):
    def __init__(self, out_channels, scale=2):
        super(PixelShuffleUpsample, self).__init__()
        self.out_channels = out_channels
        self.scale = scale
        self.out_ratio = scale**2

    def build(self, inputs):

        self.in_channels = inputs[-1]
        self.repeats = self.out_channels * self.out_ratio // self.in_channels
        assert self.out_channels * self.out_ratio % self.in_channels == 0, "out_channels * scale**2 must be divisible by in_channels"

    def call(self, inputs):

        #Need to do a repeat interleave here
        x = keras.ops.repeat(inputs, self.repeats, axis=-1)

        return tf.nn.depth_to_space(x, self.scale)


class ConvPixelShuffleUpsample(keras.layers.Layer):
    def __init__(self,
                 out_channels,
                 scale=2,
                 activation='keras.activations.relu'):
        super(ConvPixelShuffleUpsample, self).__init__()
        self.out_channels = out_channels
        self.scale = scale
        self.out_ratio = scale**2
        self.activation = activation

    def build(self, inputs):

        self.in_channels = inputs[-1]

        assert self.out_channels * self.out_ratio % self.in_channels == 0, "out_channels * scale**2 must be divisible by in_channels"

        self.conv = keras.layers.Conv2D(
            self.out_channels * self.out_ratio,
            3,
            padding='same'
            )

        if self.activation is not None:
            if isinstance(self.activation, str):
                self.act = keras.layers.Activation(eval(self.activation))
            else:
                self.act = keras.layers.Activation(self.activation)

    def call(self, inputs):
        x = self.conv(inputs)
        if self.activation is not None:
            x = self.act(x)
        return tf.nn.depth_to_space(x, self.scale)
