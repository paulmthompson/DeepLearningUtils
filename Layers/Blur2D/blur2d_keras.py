

"""
This implements a blur layer to be used with max pooling or convolutions for anti-aliasing.

Reference:
Zhang, R., 2019. Making Convolutional Networks Shift-Invariant Again. 
https://doi.org/10.48550/arXiv.1904.11486

Github repository for original implementation:
https://github.com/adobe/antialiased-cnns

"""

import keras
import numpy as np


class Blur2D(keras.layers.Layer):
    def __init__(self, 
                 kernel_size=2, 
                 stride=2, 
                 kernel_type="Rect", 
                 padding="valid",):
        super(Blur2D, self).__init__()

        self.kernel_size = kernel_size
        if kernel_type == "Rect":

            self.kernel2d = keras.ops.ones((self.kernel_size, self.kernel_size, 1, 1))
            self.kernel2d = keras.ops.divide(
                self.kernel2d, keras.ops.sum(self.kernel2d)
            )

        elif kernel_type == "Triangle":
            # Assert that kernel is > 2 else print error
            assert (
                kernel_size > 2
            ), "Kernel size must be greater than 2 for Triangle kernel"

            if kernel_size % 2 == 0:
                kernel_base = np.arange(1, (kernel_size + 1) // 2 + 1)
                # mirror the kernel_base
                kernel_base = np.concatenate([kernel_base, np.flip(kernel_base)])
            else:
                kernel_base = np.arange(1, (kernel_size + 1) // 2 + 1)
                # mirror the kernel_base
                kernel_base = np.concatenate([kernel_base, np.flip(kernel_base[:-1])])

            self.kernel2d = keras.ops.expand_dims(
                keras.ops.outer(kernel_base, kernel_base), axis=(2, 3)
            )
            self.kernel2d = keras.ops.cast(self.kernel2d, "float32")
            self.kernel2d = keras.ops.divide(
                self.kernel2d, keras.ops.sum(self.kernel2d)
            )
        elif kernel_type == "Binomial":

            assert kernel_size == 5, "Binomial kernel only supports kernel size of 5"

            kernel = np.array([1, 4, 6, 4, 1])
            self.kernel2d = keras.ops.expand_dims(
                keras.ops.outer(kernel, kernel), axis=(2, 3)
            )
            self.kernel2d = keras.ops.cast(self.kernel2d, "float32")
            self.kernel2d = keras.ops.divide(
                self.kernel2d, keras.ops.sum(self.kernel2d)
            )
        else:
            raise ValueError("Kernel type must be either Rect, Triangle, or Binomial")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def build(self, input_shape):
        self.kernel = keras.ops.tile(self.kernel2d, [1, 1, input_shape[-1], 1])

    def call(self, inputs):
        # Apply depthwise convolution for blurring
        blurred = keras.ops.depthwise_conv(
            inputs,
            self.kernel,
            strides=(self.stride, self.stride),
            padding=self.padding,
        )
        return blurred
