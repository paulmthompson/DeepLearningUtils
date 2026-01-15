import keras

from src.DeepLearningUtils.Layers.Attention.MHA_mvit2.attention_keras import MultiHeadAttention


class CoAttentionModule(keras.layers.Layer):
    def __init__(self,
                 memory_attention_module,
                 **kwargs):
        super(CoAttentionModule, self).__init__(**kwargs)
        self.memory_attention_module = memory_attention_module

    def build(self, input_shape):
        query_sequence, memory_bank_sequence, mask = input_shape

        self.layer_norm1 = keras.layers.LayerNormalization(name="layer_norm1")

        n_frames = memory_bank_sequence[1]

        self.query_norms = [
            keras.layers.LayerNormalization(name=f"query_norm_{i}") for i in range(n_frames)]

        self.query_denses = [
            keras.layers.Dense(
                128,
                activation=None,
                use_bias=False,
                name=f"query_dense_{i}") for i in range(n_frames)]

        self.key_dense = keras.layers.Dense(
            128,
            activation=None,
            use_bias=False)
        self.value_dense = keras.layers.Dense(
            128,
            activation=None,
            use_bias=False)

        self.out_dense = keras.layers.Dense(
            query_sequence[-1],
            activation=None,
            name="out_dense"
        )

        self.layer_norm2 = keras.layers.LayerNormalization(name="layer_norm2")

        hidden_dims = 512

        self.mlp1 = keras.layers.Dense(
            hidden_dims,
            activation='hard_swish',
            name="mlp1"
        )

        self.mlp2 = keras.layers.Dense(
            query_sequence[-1],
            activation=None,
            name="mlp2"
        )

        self.noise = keras.layers.GaussianNoise(1.0)

        self.spatial_drop = keras.layers.TimeDistributed(keras.layers.SpatialDropout2D(0.1))

    def call(self, inputs):
        query_sequence, memory_bank_sequence, mask = inputs

        # query_sequence = self.noise(query_sequence)
        # memory_bank_sequence = self.noise(memory_bank_sequence)

        # query_sequence = self.spatial_drop(query_sequence)
        # memory_bank_sequence = self.spatial_drop(memory_bank_sequence)

        memory_bank_sequence = self.layer_norm1(memory_bank_sequence)

        key = self.key_dense(memory_bank_sequence)
        value = self.value_dense(memory_bank_sequence)

        # Split the memory bank sequence into N sequences of 1 frame
        memory_bank_frames = keras.ops.split(
            memory_bank_sequence,
            indices_or_sections=memory_bank_sequence.shape[1],
            axis=1)

        # Apply attention between the query and each of these sequences independently
        attention_results = []
        att_input = query_sequence
        for i, memory_frame in enumerate(memory_bank_frames):

            key = self.key_dense(memory_frame)
            value = self.value_dense(memory_frame)
            att_input = self.query_norms[i](att_input)
            att_input = self.query_denses[i](att_input)
            attention_result = self.memory_attention_module(
                [att_input, key, value, mask[:, i: i +1]])
            attention_result *= mask[:, i: i + 1, None, None, None]
            attention_result = attention_result + att_input
            attention_results.append(attention_result)
            att_input = attention_result

        # Add together the results
        att_out = keras.layers.Add()(attention_results) / keras.ops.sum(mask, axis=1)[:, None, None, None, None]

        att_out = self.out_dense(att_out)
        att_out = att_out + query_sequence

        att_out = self.layer_norm2(att_out)

        att_out = self.mlp1(att_out)

        att_out = self.mlp2(att_out)

        return att_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "memory_attention_module": keras.saving.serialize_keras_object(self.memory_attention_module),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["memory_attention_module"] = keras.saving.deserialize_keras_object(
            config["memory_attention_module"]
        )
        return cls(**config)


class CoMemoryAttentionModule(keras.layers.Layer):
    def __init__(self,
                 key_dim=128,
                 value_dim=256,
                 query_seq_len=1,
                 key_seq_len=5,
                 use_norm=False,
                 attention_drop_rate=0.0,
                 use_positional_embedding=True,
                 attention_heads=8,
                 use_linear_attention=False,
                 use_qkv_embedding=False,
                 **kwargs):

        super(CoMemoryAttentionModule, self).__init__(**kwargs)

        self.key_dim: int = key_dim
        self.value_dim: int = value_dim
        self.query_seq_len: int = query_seq_len
        self.key_seq_len: int = key_seq_len
        self.use_norm: bool = use_norm
        self.attention_drop_rate: float = attention_drop_rate
        self.use_positional_embedding: bool = use_positional_embedding
        self.attention_heads: int = attention_heads
        self.use_linear_attention: bool = use_linear_attention
        self.use_qkv_embedding: bool = use_qkv_embedding

    def build(self, input_shape):

        encoder_output, memory_key, memory_value, memory_mask = input_shape

        # assert that encoder_out is 5D tensor
        assert len(encoder_output) == 5

        # assert that memory is 5D tensor
        assert len(memory_key) == 5

        self.h: int = encoder_output[2]
        self.w: int = encoder_output[3]
        self.key_height: int = memory_key[2]
        self.key_width: int = memory_key[3]

        if self.use_norm:
            self.query_norm = keras.layers.LayerNormalization(name="query_norm")
            self.memory_norm = keras.layers.LayerNormalization(name="memory_norm")
        else:
            self.query_norm = keras.layers.Activation('linear')
            self.memory_norm = keras.layers.Activation('linear')

        self.query_reshape = keras.layers.Reshape((self.h * self.w * self.query_seq_len, encoder_output[-1]))
        self.memory_reshape = keras.layers.Reshape \
            ((self.key_height * self.key_width * self.key_seq_len, memory_key[-1]))

        # self.residual_path = keras.layers.Dense(encoder_output[-1], 3, padding="same")

        self.att = MultiHeadAttention(
            query_shape=(self.query_seq_len, self.h, self.w),
            key_shape=(self.key_seq_len, self.key_height, self.key_width),
            h=self.attention_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim,
            attention_drop_rate=self.attention_drop_rate,
            use_positional_embedding=self.use_positional_embedding,
            use_linear_attention=self.use_linear_attention,
            query_embedding=self.use_qkv_embedding,
            key_embedding=self.use_qkv_embedding,
            value_embedding=self.use_qkv_embedding
        )

        self.output_reshape = keras.layers.Reshape((self.query_seq_len, self.h, self.w, encoder_output[-1]))

    def call(self, inputs):

        encoder_output, memory_key, memory_value, memory_mask = inputs

        query_encoder = encoder_output

        query_encoder = self.query_reshape(query_encoder)
        memory_key = self.memory_reshape(memory_key)
        memory_value = self.memory_reshape(memory_value)

        # MVit2 uses layer normalization after query embedding
        query_encoder = self.query_norm(query_encoder)
        memory_key = self.memory_norm(memory_key)
        memory_value = self.memory_norm(memory_value)

        mh_mask = create_encoder_memory_mask(
            memory_mask,
            self. h *self.w * self.query_seq_len,
            self.key_height *self.key_width *self.key_seq_len,
            self.query_seq_len,
            self.key_seq_len,
            self.attention_heads)

        att_out = self.att(query_encoder, memory_key, memory_value, mask=mh_mask)

        att_out = self.output_reshape(att_out)

        return att_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "query_seq_len": self.query_seq_len,
            "key_seq_len": self.key_seq_len,
            "use_norm": self.use_norm,
            "attention_drop_rate": self.attention_drop_rate,
            "use_positional_embedding": self.use_positional_embedding,
            "attention_heads": self.attention_heads,
            "use_linear_attention": self.use_linear_attention,
            "use_qkv_embedding": self.use_qkv_embedding,
        })
        return config


def create_encoder_memory_mask(
        mask,
        encoder_seq_length: int,
        memory_seq_length: int,
        query_frame_num: int,
        key_frame_num: int,
        heads: int = 1, ):

    mask_expanded = keras.ops.expand_dims(mask, axis=1)  # Shape: (batch_size, 1, SEQ_LEN)
    mask_expanded = keras.ops.expand_dims(mask_expanded, axis=-1)  # Shape: (batch_size, 1, SEQ_LEN, 1)
    mask_expanded = keras.ops.tile(mask_expanded, [1, encoder_seq_length, 1,
                                                       memory_seq_length // key_frame_num])  # Shape: (batch_size, h * w, SEQ_LEN, h * w)
    mask_expanded = keras.layers.Reshape((encoder_seq_length, memory_seq_length))(
            mask_expanded)  # Shape: (batch_size, h * w, SEQ_LEN * h * w)

    mask_expanded = keras.ops.expand_dims(mask_expanded, axis=1)  # Shape: (batch_size, 1, T, S)
    mask_expanded = keras.ops.tile(mask_expanded, [1, heads, 1, 1])  # Shape: (batch_size, heads, T, S)
    return mask_expanded