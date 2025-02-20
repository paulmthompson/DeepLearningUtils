
import keras
import math
from keras import ops
import numpy as np
import tensorflow as tf


class RotaryPositionalEncoding2D(keras.layers.Layer):
    """
    2D Rotary Positional Encoding layer for Keras 3.  This layer computes and applies
    2D rotary positional embeddings to query (q) and key (k) tensors.  It's
    designed to work with input tensors of shape (batch_size, height*width, sequence_len, dimension),
    where the rotary embeddings are applied to the height*width dimension and
    broadcast across the sequence_len dimension.

    Args:
        dim (int): The dimensionality of the embeddings.
        height (int): The height of the 2D input.
        width (int): The width of the 2D input.
        theta (float, optional): The theta value for frequency calculation.
            Defaults to 10.0.
        rotate (bool, optional): Whether to use random rotations. Defaults to True.
        max_freq (int, optional): maximum frequency. Defaults to 64.

    Call args:
        q (Tensor): The query tensor. Shape: [batch_size, sequence_len, height*width, dimension]

    Returns:
        Tuple[Tensor, Tensor]: The rotated query and key tensors, with the same
            shape as the input q and k tensors.
    """

    def __init__(self, dim, height, width, theta=10.0, rotate=True, max_freq=64, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.theta = theta
        self.rotate = rotate
        self.height = height
        self.width = width
        self.max_freq = max_freq  

        self.freqs = self.add_weight(
            name="freqs",
            shape=(2, dim // 2),
            initializer=self.init_2d_freqs,  # Use a custom initializer
            trainable=True,
        )

    def init_2d_freqs(self, input_shape, dtype="complex64"):

      dim = input_shape[1] * 2 #! changed this
      mag = 1 / (
          self.theta
          ** (ops.arange(0, dim, 2) / dim)  #step by two
      )
      angles = (
          np.random.uniform((1,)) * 2 * math.pi
          if self.rotate
          else ops.zeros((1,))
      )
      fx = mag * ops.cos(angles)
      fy = mag * ops.sin(angles)

      freqs = ops.stack([fx, fy], axis=0)
      return ops.cast(freqs, dtype)

    def init_t_xy(self):
        # t is a sequence from 0 to height*width - 1
        t = ops.arange(self.height * self.width, dtype="float32")
        t_x = ops.cast(t % self.width, "float32") # x coordinate
        t_y = ops.cast(ops.floor(t / self.width), "float32") # y coordinate
        return t_x, t_y

    def compute_mixed_cis(self, t_x, t_y):
        n = ops.shape(t_x)[0]  # n = height * width

        freqs_x = ops.expand_dims(t_x, -1) @ ops.expand_dims(self.freqs[0], 0)
        freqs_y = ops.expand_dims(t_y, -1) @ ops.expand_dims(self.freqs[1], 0)

        freq_sum = freqs_x + freqs_y

        #freq_sum_complex = ops.cast(freq_sum, "complex64")
        freq_sum_complex = tf.complex(freq_sum, 0.0)

        freqs_cis = ops.exp(1.0j * freq_sum_complex)

        freqs_cis = ops.concatenate([freqs_cis, freqs_cis], axis = -1)

        freqs_cis = freqs_cis[..., :self.dim]
        return freqs_cis 

    def apply_rotary_emb(self, q, freqs_cis):

        #q_complex = ops.cast(q, "complex64")
        q_complex = tf.complex(q, 0.0)

        q_rotated = q_complex * freqs_cis[None, None, :, :] # batch, seq_len, height*width, dim

        q_rotated_real = ops.real(q_rotated)

        return q_rotated_real

    def call(self, q):

        t_x, t_y = self.init_t_xy()
        freqs_cis = self.compute_mixed_cis(t_x, t_y)

        # Apply rotary embeddings
        q_rotated = self.apply_rotary_emb(q, freqs_cis)
        return q_rotated
