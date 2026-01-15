import keras

from src.DeepLearningUtils.Layers.Normalization.bn_keras import AccumBatchNormalization


class UNetDecoder(keras.layers.Layer):
    def __init__(self,
                 filter_sizes,
                 activation='relu',
                 transpose=False,
                 use_norm=True,
                 accum_steps=1,
                 **kwargs):
        super(UNetDecoder, self).__init__(**kwargs)
        self.filter_sizes = filter_sizes
        self.activation = activation
        self.transpose = transpose
        self.use_norm = use_norm
        self.accum_steps = accum_steps
        self.conv_layers = []
        self.bn_layers = []
        self.activation_layers = []
        self.upsample_layers = []
        self.concat_layers = []

        for i, filters in enumerate(filter_sizes):
            self.conv_layers.append(keras.layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                name=f'conv_layers_{i}'))
            if use_norm:
                if accum_steps > 1:
                    self.bn_layers.append(AccumBatchNormalization(
                        accum_steps=accum_steps,
                        momentum=0.9,
                        name=f'bn_layers_{i}'))
                else:
                    self.bn_layers.append(keras.layers.BatchNormalization(
                        momentum=0.9,
                        name=f'bn_layers_{i}'))
            else:
                self.bn_layers.append(keras.layers.Lambda(lambda x: x, name=f'bn_layers_{i}'))
            self.activation_layers.append(keras.layers.Activation(
                activation,
                name=f'activation_layers_{i}'))
            if transpose:
                self.upsample_layers.append(keras.layers.Conv2DTranspose(
                    filters,
                    (3, 3),
                    strides=(2, 2),
                    padding='same',
                    name=f'upsample_layers_{i}'))
            else:
                self.upsample_layers.append(keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation='bilinear',
                    name=f'upsample_layers_{i}'))
            self.concat_layers.append(keras.layers.Concatenate(name=f'concat_{i}'))

        self.output_conv = keras.layers.Conv2D(
            1,
            (1, 1),
            activation='sigmoid',
            name='output_conv',
            bias_initializer=keras.initializers.Constant(-5.0))

    def call(self, inputs):
        x = inputs[0]
        encoder_outputs = inputs[1:]

        for i in range(len(self.filter_sizes)):
            x = self.upsample_layers[i](x)
            x = self.concat_layers[i]([x, encoder_outputs[i]])
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.activation_layers[i](x)

        x = self.output_conv(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filter_sizes": self.filter_sizes,
            "activation": self.activation,
            "transpose": self.transpose,
            "use_norm": self.use_norm,
            "accum_steps": self.accum_steps,
        })
        return config
