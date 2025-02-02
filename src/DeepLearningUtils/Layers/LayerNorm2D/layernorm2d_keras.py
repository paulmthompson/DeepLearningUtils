
import keras


class LayerNorm2d(keras.layers.Layer):
    def __init__(self,
                 num_channels: int,
                 eps: float = 1e-6,
                 **kwargs):
        """

        Adapted from here:
        https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_utils.py
        """
        super(LayerNorm2d, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.eps = eps

    def build(self, input_shape):

        self.weight = self.add_weight(
            shape=(self.num_channels,),
            initializer='ones',
            trainable=True,
            name='weight'
        )
        self.bias = self.add_weight(
            shape=(self.num_channels,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super(LayerNorm2d, self).build(input_shape)

    def call(self, x):
        #u = keras.ops.mean(x, axis=[1, 2], keepdims=True)
        #s = keras.ops.mean(keras.ops.square(x - u), axis=[1, 2], keepdims=True)
        u = keras.ops.mean(x, axis=-1, keepdims=True)
        s = keras.ops.mean(keras.ops.square(x - u), axis=-1, keepdims=True)
        x = (x - u) / keras.ops.sqrt(s + self.eps)
        x = self.weight[None, None, :] * x + self.bias[None, None, :]
        return x
