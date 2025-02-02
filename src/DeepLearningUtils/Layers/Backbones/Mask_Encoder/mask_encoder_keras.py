from src.DeepLearningUtils.Layers.Blur2D.blur2d_keras import Blur2D
from src.DeepLearningUtils.Layers.LayerNorm2D.layernorm2d_keras import LayerNorm2d

import keras


def create_mask_encoder(
        input_shape,
        output_channels,
        anti_aliasing=True,):
    """

    Sam mask encoder tries to keep the same number of h x w x c channels as the input tensor.


    Adapted from here:
    https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/memory_encoder.py

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor
    output_channels : int
        Number of output channels
    """

    inputs = keras.layers.Input(shape=input_shape)

    if anti_aliasing:
        x = keras.layers.Conv2D(
            4,
            (2, 2),
            strides=(1, 1),
            padding='same',
            name="conv1")(inputs)  # (b, 128, 128, 4)
        x = Blur2D(5, 2, "Binomial", padding="same", name="blur1")(x)
    else:
        x = keras.layers.Conv2D(
            4,
            (2, 2),
            strides=(2, 2),
            padding='same',
            name="conv1")(inputs)  # (b, 128, 128, 4)
    x = LayerNorm2d(4, name="norm1")(x)
    x = keras.layers.Activation(keras.activations.hard_swish)(x)

    if anti_aliasing:
        x = keras.layers.Conv2D(
            64,
            (4, 4),
            strides=(1, 1),
            padding='same',
            name="conv2")(x)
        x = Blur2D(3, 2, "Triangle", padding="same")(x)  # (b, 64, 64, 64)
        x = Blur2D(3, 2, "Triangle", padding="same")(x)  # (b, 32, 32, 64)
    else:
        x = keras.layers.Conv2D(
            64,
            (4, 4),
            strides=(4, 4),
            padding='same',
            name="conv2")(x)  # (b, 32, 32, 64)
    x = LayerNorm2d(64, name="norm2")(x)
    x = keras.layers.Activation(keras.activations.hard_swish)(x)


    x = keras.layers.Conv2D(
        256,
        (2, 2),
        strides=(2, 2),
        padding='same',
        name="conv3")(x)

    if output_channels != x.shape[-1]:
        x = LayerNorm2d(x.shape[-1])(x)
        x = keras.layers.Activation(keras.activations.hard_swish)(x)

        x = keras.layers.Conv2D(output_channels, (1, 1), padding='same', name="mask_embedding_conv")(x)

    return keras.Model(inputs, x, name='mask_encoder')
