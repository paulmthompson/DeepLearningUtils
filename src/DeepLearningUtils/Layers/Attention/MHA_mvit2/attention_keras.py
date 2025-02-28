import keras

from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_keras import \
    RelativePositionalEmbedding2D
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_keras import \
    RelativePositionalEmbedding2DKey

from typing import Tuple


class DotProductAttention(keras.layers.Layer):
    def __init__(self,
                 query_shape: Tuple[int, int, int],
                 key_shape: Tuple[int, int, int],
                 use_scale=True,
                 drop_rate=0.0,
                 use_positional_embedding=True,
                 use_key_positional_embedding=True,
                 use_linear_attention=True,
                 heads=8,
                 name="",
                 **kwargs
                 ):
        """

        Parameters
        ----------
        query_shape : Tuple[int, int, int]
            Shape of the query tensor (seq_len x height x width)
        key_shape : Tuple[int, int, int]
            Shape of the key tensor (seq_len x height x width)
        use_scale : bool, optional
            Whether to scale the dot product, by default True
        drop_rate : float, optional
            Dropout rate applied after softmax, by default 0.0

        """
        super(DotProductAttention, self).__init__(**kwargs)
        self.use_scale = use_scale
        self.query_seq_len, self.query_height, self.query_width = query_shape
        self.key_seq_len, self.key_height, self.key_width = key_shape
        self.drop_rate = drop_rate
        self.use_positional_embedding = use_positional_embedding
        self.use_key_positional_embedding = use_key_positional_embedding
        self.name = name
        self.use_linear_attention: bool = use_linear_attention
        self.heads: int = heads

    def build(self, query_shape, key_shape, value_shape, mask_shape=None):
        #query_shape, key_shape, value_shape = inputs_shape

        if self.use_scale:
            dim_k = keras.ops.cast(query_shape[-1], keras.backend.floatx())
            self.scale = 1 / keras.ops.sqrt(dim_k)
        else:
            self.scale = None

        if self.use_positional_embedding:
            self.query_positional_embedding = RelativePositionalEmbedding2D(
                (self.query_seq_len, self.query_height, self.query_width, query_shape[-1]),
                (self.key_seq_len, self.key_height, self.key_width, key_shape[-1]),
                query_shape[-1],
                self.heads,
                drop_rate=self.drop_rate,
                name=f"Query_Embedding_{self.name}"
            )
            self.query_positional_embedding.build([query_shape, query_shape])
        else:
            self.query_positional_embedding = None

        if self.use_key_positional_embedding:
            self.key_positional_embedding = RelativePositionalEmbedding2DKey(
                (self.query_seq_len, self.query_height, self.query_width, query_shape[-1]),
                (self.key_seq_len, self.key_height, self.key_width, key_shape[-1]),
                query_shape[-1],
                self.heads,
                drop_rate=self.drop_rate,
                name=f"Key_Embedding_{self.name}"
            )
            self.key_positional_embedding.build([key_shape, key_shape])
        else:
            self.key_positional_embedding = None

        self.drop = keras.layers.Dropout(rate=self.drop_rate)

    def linear_attention(self, query_input, key_input, value, mask=None):
        #query_input, key_input, value = inputs

        query = keras.activations.relu(query_input)  # batch x heads x S x dim
        key = keras.activations.relu(key_input)  # batch x heads x S x dim

        key = keras.ops.transpose(key, axes=[0, 1, 3, 2])

        score = keras.ops.matmul(query, key)

        # -2 in original implementation
        score = score / keras.ops.sum(score, axis=1, keepdims=True)

        if self.query_positional_embedding is not None:
            score = self.query_positional_embedding([query_input, score])
        if self.key_positional_embedding is not None:
            score = self.key_positional_embedding([key_input, score])

        score = self.drop(score)

        score *= mask

        score = keras.ops.matmul(score, value)

        score += query_input  # Residual connection in MVit2

        return score

    def softmax_attention(self, query, key_input, value, mask=None):
        #query, key_input, value = inputs
        key = keras.ops.transpose(key_input, axes=[0, 1, 3, 2])
        score = keras.ops.matmul(query, key)
        if self.use_scale:
            scale = keras.ops.cast(1 / keras.ops.sqrt(query.shape[-1]), score.dtype)
            score = score * scale

        if self.query_positional_embedding is not None:
            score = self.query_positional_embedding([query, score])
        if self.key_positional_embedding is not None:
            score = self.key_positional_embedding([key_input, score])

        if mask is not None:
            score -= 1.e9 * (keras.ops.cast(1.0, mask.dtype) - mask)

        score = keras.activations.softmax(score, axis=-1)
        score = self.drop(score)
        score = keras.ops.matmul(score, value)

        score += query  # Residual connection in MVit2

        return score

    def call(self, query, key, value, mask=None):
        if self.use_linear_attention:
            return self.linear_attention(query, key, value, mask=mask)
        else:
            return self.softmax_attention(query, key, value, mask=mask)


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self,
                 query_shape: Tuple[int, int, int],
                 key_shape: Tuple[int, int, int],
                 h=8,
                 value_dim=128,
                 key_dim=128,
                 attention_drop_rate=0.0,
                 use_positional_embedding=True,
                 use_key_positional_embedding=True,
                 use_linear_attention=False,
                 output_activation=None,
                 query_embedding=True,
                 key_embedding=True,
                 value_embedding=True,
                 **kwargs):
        """

        Parameters
        ----------
        query_shape : Tuple[int, int, int]
            Shape of the query tensor (seq_len x height x width)
        key_shape : Tuple[int, int, int]
            Shape of the key tensor (seq_len x height x width)
        h : int, optional
            Number of heads, by default 8
        value_dim : int, optional
            Dimension of the value, by default 128
        key_dim : int, optional
            Dimension of the key, by default 128
        attention_drop_rate : float, optional
            Dropout rate applied after softmax, by default 0.0
        use_positional_embedding : bool, optional
            Whether to use positional embedding, by default True
        use_key_positional_embedding : bool, optional
            Whether to use key positional embedding, by default True
        use_linear_attention : bool, optional
            Whether to use linear attention, by default True
        output_activation : str, optional
            Activation function for the output, by default None

        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.h: int = h
        self.query_seq_len, self.query_height, self.query_width = query_shape
        self.key_seq_len, self.key_height, self.key_width = key_shape
        self.attention_drop_rate: float = attention_drop_rate
        self.use_positional_embedding: bool = use_positional_embedding
        self.use_key_positional_embedding: bool = use_key_positional_embedding
        self.value_dim: int = value_dim
        self.key_dim: int = key_dim
        self.use_linear_attention: bool = use_linear_attention
        self.output_activation = output_activation
        self.query_embedding: bool = query_embedding
        self.key_embedding: bool = key_embedding
        self.value_embedding: bool = value_embedding

    def build(self, query_shape, key_shape, value_shape, mask_shape=None):
        #query_shape, key_shape, value_shape = inputs_shape
        output_shape = query_shape
        d_model = self.value_dim

        # Note: units can be anything, but this is what the paper does
        units = d_model // self.h

        if self.query_embedding:
            self.query_dense = keras.layers.Dense(
                d_model,
                activation=None,
                use_bias=False,
                name=f"query_dense")
            self.query_dense.build(query_shape)
        else:
            self.query_dense = keras.layers.Activation("linear")

        if self.key_embedding:
            self.key = keras.layers.Dense(
                d_model,
                activation=None,
                use_bias=False,
                name=f"key_dense")
            self.key.build(key_shape)
        else:
            self.key = keras.layers.Activation("linear")

        if self.value_embedding:
            self.value = keras.layers.Dense(
                d_model,
                activation=None,
                use_bias=False,
                name=f"value_dense")
            self.value.build(value_shape)
        else:
            self.value = keras.layers.Activation("linear")

        self.att = DotProductAttention(
            query_shape=(self.query_seq_len, self.query_height, self.query_width),
            key_shape=(self.key_seq_len, self.key_height, self.key_width),
            use_scale=True,
            drop_rate=self.attention_drop_rate,
            use_positional_embedding=self.use_positional_embedding,
            use_key_positional_embedding=self.use_key_positional_embedding,
            use_linear_attention=self.use_linear_attention,
            heads=self.h,
            name="attention"
        )

        """
        self.att.build([
            (query_shape[0], self.h, query_shape[1], units),
            (key_shape[0], self.h, key_shape[1], units),
            (value_shape[0], self.h, value_shape[1], units)
        ])
        """
        self.att.build((query_shape[0], self.h, query_shape[1], units),
            (key_shape[0], self.h, key_shape[1], units),
            (value_shape[0], self.h, value_shape[1], units))

        self.reshape_q = keras.layers.Reshape((-1, self.h, units))
        self.reshape_k = keras.layers.Reshape((-1, self.h, units))
        self.reshape_v = keras.layers.Reshape((-1, self.h, units))

        self.reshape_q.build((query_shape[0], query_shape[1], d_model))
        self.reshape_k.build((key_shape[0], key_shape[1], d_model))
        self.reshape_v.build((value_shape[0], value_shape[1], d_model))

        self.permute_q = keras.layers.Permute((2, 1, 3))
        self.permute_k = keras.layers.Permute((2, 1, 3))
        self.permute_v = keras.layers.Permute((2, 1, 3))

        self.permute_q.build((query_shape[0], query_shape[1], self.h, units))
        self.permute_k.build((key_shape[0], key_shape[1], self.h, units))
        self.permute_v.build((value_shape[0], value_shape[1], self.h, units))

        self.permute_att = keras.layers.Permute((2, 1, 3))
        self.permute_att.build((query_shape[0], self.h, query_shape[1], units))

        self.reshape_att = keras.layers.Reshape((-1, d_model))
        self.reshape_att.build((query_shape[0], query_shape[1], self.h, units))

        self.out = keras.layers.Dense(
            output_shape[-1],
            activation=None,
            use_bias=True,
            name="dense_out"
        )
        self.out.build((query_shape[0], query_shape[1], self.h * units))

        self.output_dropout = keras.layers.Dropout(self.attention_drop_rate)

    def call(self, query, key, value, mask=None):
        #query, key, value = inputs

        query = self.query_dense(query)
        key = self.key(key)
        value = self.value(value)

        query = self.reshape_q(query)
        key = self.reshape_k(key)
        value = self.reshape_v(value)

        query = self.permute_q(query)
        key = self.permute_k(key)
        value = self.permute_v(value)

        out = self.att(query, key, value, mask=mask)

        out = self.permute_att(out)
        out = self.reshape_att(out)

        out = self.out(out)

        out = self.output_dropout(out)

        return out
