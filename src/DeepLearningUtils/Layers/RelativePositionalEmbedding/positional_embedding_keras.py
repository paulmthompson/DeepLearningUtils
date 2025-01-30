"""
Copyright 2019, Facebook, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

"""
This keras implementation of relative positional encoding from MVit2 was
adapted from the original pytorch implementation:
https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py

Adapted by Paul Thompson 2024
"""

import keras
from typing import Tuple

class RelativePositionalEmbedding2D(keras.layers.Layer):
    def __init__(self,
                 query_shape: Tuple[int, int, int, int],
                 key_shape: Tuple[int, int, int, int],
                 query_dim: int,
                 heads: int,
                 drop_rate=0.0,
                 **kwargs):
        """

        Weights:
        height_embeddings: [2 * max_height_dist + 1, query_dim]
        width_embeddings: [2 * max_width_dist + 1, query_dim]
        time_embeddings: [seq_len, query_dim]

        Parameters
        ----------
        query_shape: Tuple[int, int, int, int]
            seq_len, height, width, and channels of the query tensor
        key_shape: Tuple[int, int, int, int]
            seq_len, height, width, and channels of the key tensor
        query_dim : int
            Embedding dimension of the query tensor
        heads : int
            Number of heads in the multi-head attention
        drop_rate : float, optional
            Dropout rate applied to the embeddings, by default 0.0

        """
        super(RelativePositionalEmbedding2D, self).__init__(**kwargs)

        self.query_seq_len, self.query_height, self.query_width, self.query_channels = query_shape
        self.key_seq_len, self.key_height, self.key_width, self.key_channels = key_shape

        self.query_dim = query_dim
        self.drop_rate = drop_rate

        self.heads = heads

        q_h = self.query_height
        q_w = self.query_width

        k_h = self.key_height
        k_w = self.key_width

        # Calculate max distance for each dimension
        max_height_dist = max(self.query_height, self.key_height) - 1
        max_width_dist = max(self.query_width, self.key_width) - 1
        max_t_dist = max(self.query_seq_len, self.key_seq_len) - 1

        # Create embedding matrices
        self.height_embeddings = self.add_weight(
            name='height_embeddings',
            shape=(2 * max_height_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        self.width_embeddings = self.add_weight(
            name='width_embeddings',
            shape=(2 * max_width_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )

        self.time_embeddings = self.add_weight(
            name='time_embeddings',
            shape=(2 * max_t_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )

        self.height_dropout = keras.layers.Dropout(rate=self.drop_rate)
        self.width_dropout = keras.layers.Dropout(rate=self.drop_rate)
        self.time_dropout = keras.layers.Dropout(rate=self.drop_rate)

        self.query_reshape_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len, self.query_height, self.query_width, self.query_dim)
        )

        self.score_reshape_pre_embedding_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len, q_h, q_w, self.key_seq_len, k_h, k_w)
        )

        self.score_reshape_after_embedding_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len * q_h * q_w, self.key_seq_len * k_h * k_w)
        )

    def build(self, input_shape):

        query_shape, scores_shape = input_shape

        super(RelativePositionalEmbedding2D, self).build(input_shape)

    def call(self, inputs):

        query, scores = inputs

        q_h = self.query_height
        k_h = self.key_height
        q_h_ratio = round(max(k_h / q_h, 1.0))
        k_h_ratio = round(max(q_h / k_h, 1.0))
        dist_h = (
            keras.ops.arange(q_h)[:, None] * q_h_ratio - keras.ops.arange(k_h)[None, :] * k_h_ratio
        )
        dist_h += (k_h - 1) * k_h_ratio # [qh, kh]

        q_w = self.query_width
        k_w = self.key_width
        q_w_ratio = round(max(k_w / q_w, 1.0))
        k_w_ratio = round(max(q_w / k_w, 1.0))
        dist_w = (
            keras.ops.arange(q_w)[:, None] * q_w_ratio - keras.ops.arange(k_w)[None, :] * k_w_ratio
        )
        dist_w += (k_w - 1) * k_w_ratio # [qw, kw]

        q_t_ratio = round(max(self.key_seq_len / self.query_seq_len, 1.0))
        k_t_ratio = round(max(self.query_seq_len / self.key_seq_len, 1.0))

        dist_t = (
            keras.ops.arange(self.query_seq_len)[:, None] * q_t_ratio - keras.ops.arange(self.key_seq_len)[None, :] * k_t_ratio
        )
        dist_t += (self.key_seq_len - 1) * k_t_ratio # [q_seq_len, key_seq_len]

        Rh = keras.ops.take(self.height_embeddings, dist_h, axis=0)
        Rw = keras.ops.take(self.width_embeddings, dist_w, axis=0)
        Rt = keras.ops.take(self.time_embeddings, dist_t, axis=0)

        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)

        query_reshaped = self.query_reshape_layer(query)

        rel_h = keras.ops.einsum("bNthwc,hkc->bNthwk", query_reshaped, Rh) # [B, head, qseq, qh, qw, kh]
        rel_w = keras.ops.einsum("bNthwc,wkc->bNthwk", query_reshaped, Rw) # [B, head, qseq, qh, qw, kw]
        rel_t = keras.ops.einsum("bNthwc,tkc->bNthwk", query_reshaped, Rt) # [B, head, qseq, qh, qw, seq_len]

        scores = self.score_reshape_pre_embedding_layer(scores)

        scores += rel_h[:, :, :, :, :, None, :, None] # [B, head, seq_len, qh, qw, key_seq_len, kh, kw]
        scores += rel_w[:, :, :, :, :, None, None, :]
        scores += rel_t[:, :, :, :, :, :, None, None]

        scores = self.score_reshape_after_embedding_layer(scores)

        return scores


class RelativePositionalEmbedding2DKey(keras.layers.Layer):
    def __init__(self,
                 query_shape: Tuple[int, int, int, int],
                 key_shape: Tuple[int, int, int, int],
                 query_dim: int,
                 heads: int,
                 drop_rate=0.0,
                 **kwargs):
        """

        Weights:
        height_embeddings: [2 * max_height_dist + 1, query_dim]
        width_embeddings: [2 * max_width_dist + 1, query_dim]
        time_embeddings: [seq_len, query_dim]

        Parameters
        ----------
        query_shape: Tuple[int, int, int, int]
            seq_len, height, width, and channels of the query tensor
        key_shape: Tuple[int, int, int, int]
            seq_len, height, width, and channels of the key tensor
        query_dim : int
            Dimension of the query tensor
        heads : int
            Number of heads in the multi-head attention
        drop_rate : float, optional
            Dropout rate applied to the embeddings, by default 0.0

        """
        super(RelativePositionalEmbedding2DKey, self).__init__(**kwargs)

        self.query_seq_len, self.query_height, self.query_width, self.query_channels = query_shape
        self.key_seq_len, self.key_height, self.key_width, self.key_channels = key_shape

        self.query_dim = query_dim
        self.drop_rate = drop_rate

        self.heads = heads

        q_h = self.query_height
        q_w = self.query_width

        k_h = self.key_height
        k_w = self.key_width

        # Calculate max distance for each dimension
        max_height_dist = max(self.query_height, self.key_height) - 1
        max_width_dist = max(self.query_width, self.key_width) - 1
        max_time_dist = max(self.query_seq_len, self.key_seq_len) - 1

        # Create embedding matrices
        self.height_embeddings = self.add_weight(
            name='height_embeddings',
            shape=(2 * max_height_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        self.width_embeddings = self.add_weight(
            name='width_embeddings',
            shape=(2 * max_width_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )
        self.time_embeddings = self.add_weight(
            name='time_embeddings',
            shape=(2 * max_time_dist + 1, self.query_dim),
            initializer='uniform',
            trainable=True,
        )

        self.height_dropout = keras.layers.Dropout(rate=self.drop_rate)
        self.width_dropout = keras.layers.Dropout(rate=self.drop_rate)
        self.time_dropout = keras.layers.Dropout(rate=self.drop_rate)

        self.key_reshape_layer = keras.layers.Reshape(
            (self.heads, self.key_seq_len, self.key_height, self.key_width, self.query_dim)
        )

        self.score_reshape_pre_embedding_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len, q_h, q_w, self.key_seq_len, k_h, k_w)
        )

        self.score_reshape_after_embedding_layer = keras.layers.Reshape(
            (self.heads, self.query_seq_len * q_h * q_w, self.key_seq_len * k_h * k_w)
        )

    def build(self, input_shape):

        key_shape, scores_shape = input_shape

        super(RelativePositionalEmbedding2DKey, self).build(input_shape)

    def call(self, input):

        key, scores = input

        q_h = self.query_height
        k_h = self.key_height

        q_h_ratio = round(max(k_h / q_h, 1.0))
        k_h_ratio = round(max(q_h / k_h, 1.0))

        dist_h = (
            keras.ops.arange(k_h)[:, None] * k_h_ratio - keras.ops.arange(q_h)[None, :] * q_h_ratio
        )
        dist_h += (q_h - 1) * q_h_ratio # [kh, qh]

        q_w = self.query_width
        k_w = self.key_width

        q_w_ratio = round(max(k_w / q_w, 1.0))
        k_w_ratio = round(max(q_w / k_w, 1.0))
        dist_w = (
            keras.ops.arange(k_w)[:, None] * k_w_ratio - keras.ops.arange(q_w)[None, :] * q_w_ratio
        )
        dist_w += (q_w - 1) * q_w_ratio # [kw, qw]

        q_t_ratio = round(max(self.key_seq_len / self.query_seq_len, 1.0))
        k_t_ratio = round(max(self.query_seq_len / self.key_seq_len, 1.0))
        dist_t = (
            keras.ops.arange(self.key_seq_len)[:, None] * k_t_ratio - keras.ops.arange(self.query_seq_len)[None, :] * q_t_ratio
        )
        dist_t += (self.query_seq_len - 1) * q_t_ratio # [key_seq_len, q_seq_len]

        Rh = keras.ops.take(self.height_embeddings, dist_h, axis=0)
        Rw = keras.ops.take(self.width_embeddings, dist_w, axis=0)
        Rt = keras.ops.take(self.time_embeddings, dist_t, axis=0)

        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)

        key_reshaped = self.key_reshape_layer(key)

        rel_h = keras.ops.einsum("hqc,bNshwc->bNqshw", Rh, key_reshaped) # [B, head, qh, ks, kh, kw]
        rel_w = keras.ops.einsum("wqc,bNshwc->bNqshw", Rw, key_reshaped) # [B, head, qw, ks, kw, kw]
        rel_t = keras.ops.einsum("sqc,bNshwc->bNqshw", Rt, key_reshaped) # [B, head, qs, ks, kh, kw]

        scores = self.score_reshape_pre_embedding_layer(scores)

        scores += rel_h[:, :, None, :, None, :, :, :] # [B, h, qs, qh, qw, seq_len, kh, kw]
        scores += rel_w[:, :, None, None, :, :, :, :]
        scores += rel_t[:, :, :, None, None, :, :, :]

        scores = self.score_reshape_after_embedding_layer(scores)

        return scores
