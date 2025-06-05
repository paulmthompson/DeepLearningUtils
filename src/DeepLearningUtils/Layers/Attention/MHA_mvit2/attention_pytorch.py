import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DeepLearningUtils.Layers.RelativePositionalEmbedding.positional_embedding_pytorch import \
    RelativePositionalEmbedding2D, \
    RelativePositionalEmbedding2DKey, \
    load_positionaL_embedding_layer_weights



from collections import OrderedDict
from typing import Optional, Tuple


class DotProductAttention(nn.Module):
    def __init__(self,
                 query_shape: Tuple[int, int, int, int],
                 memory_shape: Tuple[int, int, int, int],
                 query_dim,
                 heads=8,
                 use_scale=True,
                 drop_rate=0.0,
                 use_positional_embedding=True,
                 use_key_positional_embedding=True,
                 name=""):
        super(DotProductAttention, self).__init__()
        self.use_scale: bool = use_scale
        query_seq_len, query_height, query_width, query_channels = query_shape
        key_seq_len, key_height, key_width, key_channels = memory_shape
        self.query_height: int = query_height
        self.query_width: int = query_width
        self.key_height: int = key_height
        self.key_width: int = key_width
        self.query_seq_len: int = query_seq_len
        self.key_seq_len: int = key_seq_len
        self.drop_rate: float = drop_rate
        self.use_positional_embedding: bool = use_positional_embedding
        self.use_key_positional_embedding: bool = use_key_positional_embedding
        self.name = name
        self.heads: int = heads

        self.query_positional_embedding = nn.ModuleDict()
        if self.use_positional_embedding:
            self.query_positional_embedding[f"Query_Embedding_{name}"] = RelativePositionalEmbedding2D(
                query_shape=query_shape,
                key_shape=memory_shape,
                query_dim=query_dim,
                heads=self.heads,
                drop_rate=self.drop_rate
            )

        self.key_positional_embedding = nn.ModuleDict()
        if self.use_key_positional_embedding:
            self.key_positional_embedding[f"Key_Embedding_{name}"] = RelativePositionalEmbedding2DKey(
                query_shape=query_shape,
                key_shape=memory_shape,
                query_dim=query_dim,
                heads=self.heads,
                drop_rate=self.drop_rate
            )

        self.drop = nn.Dropout(p=self.drop_rate)

    def forward(self, query, key_input, value, mask: Optional[torch.Tensor] = None):
        key = key_input.transpose(-2, -1)
        score = torch.matmul(query, key)
        if self.use_scale:
            score = score / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))

        if self.use_positional_embedding:
            positional_embedding: RelativePositionalEmbedding2D = self.query_positional_embedding[
                f"Query_Embedding_{self.name}"]
            score = positional_embedding.forward(query, score)
        if self.use_key_positional_embedding:
            positional_embedding: RelativePositionalEmbedding2DKey = self.key_positional_embedding[
                f"Key_Embedding_{self.name}"]
            score = positional_embedding.forward(key_input, score)
            # score = self.key_positional_embedding[f"Key_Embedding_{self.name}"](key_input, score)

        if mask is not None:
            # score = score.masked_fill(mask == 0, -1e9)
            score = score - 1e9 * (1 - mask)

        score = F.softmax(score, dim=-1)
        score = self.drop(score)
        score = torch.matmul(score, value)

        score += query  # Residual connection in MVit2

        return score


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_shape: Tuple[int, int, int, int],
                 memory_shape: Tuple[int, int, int, int],
                 heads=8,
                 value_dim=128,
                 key_dim=128,
                 attention_drop_rate=0.0,
                 use_positional_embedding=True,
                 use_key_positional_embedding=True,
                 use_query_embedding=True,
                 use_key_embedding=True,
                 use_value_embedding=True):
        super(MultiHeadAttention, self).__init__()
        self.heads: int = heads
        query_seq_len, query_height, query_width, query_channels = query_shape
        key_seq_len, key_height, key_width, key_channels = memory_shape
        self.query_height: int = query_height
        self.query_width: int = query_width
        self.key_height: int = key_height
        self.key_width: int = key_width
        self.query_seq_len: int = query_seq_len
        self.key_seq_len: int = key_seq_len
        self.attention_drop_rate: float = attention_drop_rate
        self.use_positional_embedding: bool = use_positional_embedding
        self.use_key_positional_embedding: bool = use_key_positional_embedding
        self.value_dim: int = value_dim
        self.key_dim: int = key_dim
        self.use_query_embedding: bool = use_query_embedding
        self.use_key_embedding: bool = use_key_embedding
        self.use_value_embedding: bool = use_value_embedding

        d_model = self.value_dim
        units = d_model // self.heads

        if self.use_query_embedding:
            self.query_dense = nn.Linear(query_channels, d_model, bias=False)
        else:
            self.query_dense = nn.Identity()
        if self.use_key_embedding:
            self.key_dense = nn.Linear(key_channels, d_model, bias=False)
        else:
            self.key_dense = nn.Identity()
        if self.use_value_embedding:
            self.value_dense = nn.Linear(key_channels, d_model, bias=False)
        else:
            self.value_dense = nn.Identity()

        self.attention = DotProductAttention(
            query_shape,
            memory_shape,
            units,
            self.heads,
            use_scale=True,
            drop_rate=self.attention_drop_rate,
            use_positional_embedding=self.use_positional_embedding,
            use_key_positional_embedding=self.use_key_positional_embedding,
            name="attention")

        self.dense_out = nn.Linear(
            self.heads * units,
            query_channels,
            bias=True)
        self.output_dropout = nn.Dropout(self.attention_drop_rate)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        q = self.query_dense(query)
        k = self.key_dense(key)
        v = self.value_dense(value)

        q = q.view(
            -1,
            self.query_height * self.query_width * self.query_seq_len,
            self.heads,
            self.value_dim // self.heads).transpose(2, 1).contiguous()
        k = k.view(
            -1,
            self.key_height * self.key_width * self.key_seq_len,
            self.heads,
            self.value_dim // self.heads).transpose(2, 1).contiguous()
        v = v.view(
            -1,
            self.key_height * self.key_width * self.key_seq_len,
            self.heads,
            self.value_dim // self.heads).transpose(2, 1).contiguous()

        head = self.attention(q, k, v, mask=mask)

        head = head.transpose(2, 1).contiguous().view(
            -1,
            self.query_seq_len,
            self.query_height,
            self.query_width,
            self.value_dim)
        """
        q = [vv_q(query) for kk_q, vv_q in self.layersQ.items()]
        k = [vv_k(key) for kk_k, vv_k in self.layersK.items()]
        v = [vv_v(value) for kk_v, vv_v in self.layersV.items()]

        head = []
        for i, vv_a in enumerate(self.layersAttention):
            head.append(vv_a(q[i], k[i], v[i], mask=mask))
        """
        out = self.dense_out(head)
        out = self.output_dropout(out)

        return out


class MemoryAttentionModule(nn.Module):
    def __init__(self,
                 query_shape: Tuple[int, int, int, int],
                 memory_shape: Tuple[int, int, int, int],
                 attention_drop_rate=0.0,
                 use_norm=True,
                 use_positional_embedding=True,
                 use_key_positional_embedding=True,
                 attention_heads=8):
        super(MemoryAttentionModule, self).__init__()

        query_seq_len, query_height, query_width, query_channels = query_shape
        key_seq_len, key_height, key_width, key_channels = memory_shape
        self.query_height: int = query_height
        self.query_width: int = query_width
        self.key_height: int = key_height
        self.key_width: int = key_width
        self.query_seq_len: int = query_seq_len
        self.key_seq_len: int = key_seq_len
        self.attention_drop_rate: float = attention_drop_rate
        self.use_positional_embedding: bool = use_positional_embedding
        self.use_key_positional_embedding: bool = use_key_positional_embedding
        self.attention_heads: int = attention_heads

        if use_norm:
            self.query_norm = nn.LayerNorm(query_channels, eps=1e-3)
            self.memory_norm = nn.LayerNorm(key_channels, eps=1e-3)
        else:
            self.query_norm = nn.Identity()
            self.memory_norm = nn.Identity()

        self.att = MultiHeadAttention(
            query_shape,
            memory_shape,
            attention_drop_rate=attention_drop_rate,
            use_positional_embedding=use_positional_embedding,
            use_key_positional_embedding=use_key_positional_embedding,
            h=self.attention_heads
        )

        if use_norm:
            self.layer_norm2 = nn.LayerNorm(query_channels, eps=1e-3)
        else:
            self.layer_norm2 = nn.Identity()

        hidden_dim = 512

        self.mlp1 = nn.Linear(
            query_channels,
            hidden_dim)

        self.mlp_activation = nn.Hardswish()

        self.mlp2 = nn.Linear(
            hidden_dim,
            query_channels)

    def forward(self,
                query_encoder,
                memory,
                memory_mask):

        # Query is B x S x H x W x C
        # Memory is B x S x H x W x C
        # Need to permute to have channels in dim 1 for layer norm,
        # then permute back to original shape

        encoder_output = query_encoder

        # query_encoder = query_encoder.permute(0, 3, 1, 2).contiguous()
        # memory = memory.permute(0, 4, 1, 2, 3).contiguous()

        query_encoder = self.query_norm(query_encoder)
        memory = self.memory_norm(memory)

        # query_encoder = query_encoder.permute(0, 2, 3, 1).contiguous()
        # memory = memory.permute(0, 2, 3, 4, 1).contiguous()

        mh_mask = create_encoder_memory_mask(
            memory_mask,
            self.query_height * self.query_width * self.query_seq_len,
            self.key_height * self.key_width * self.key_seq_len,
            self.key_seq_len,
            self.attention_heads
        )

        # Now reshape so that query is B x (H*W) x C
        # and memory is B x (S*H*W) x C
        query_encoder = query_encoder.view(
            -1,
            self.query_height * self.query_width,
            query_encoder.shape[-1])
        memory = memory.view(
            -1,
            self.key_height * self.key_width * self.seq_len,
            memory.shape[-1])

        memory_key = memory
        memory_value = memory

        att_out = self.att(query_encoder, memory_key, memory_value, mask=mh_mask)

        # Reshape back to original shape
        att_out = att_out.view(
            -1,
            self.query_height,
            self.query_width,
            att_out.shape[-1])

        att_out = att_out + encoder_output

        att_out = self.layer_norm2(att_out)

        att_out = self.mlp1(att_out)

        att_out = self.mlp_activation(att_out)

        att_out = self.mlp2(att_out)

        return att_out



def load_mha_positional_layer_weights(keras_layer, pytorch_module):

    print(f"Loading custom layer weights for {keras_layer.name}")

    # Load query weights
    if keras_layer.query_embedding:
        pytorch_module.query_dense.weight.data = torch.tensor(keras_layer.query_dense.get_weights()[0].T)

    # Load key weights
    if keras_layer.key_embedding:
        pytorch_module.key_dense.weight.data = torch.tensor(keras_layer.key.get_weights()[0].T)

    # Load value weights
    if keras_layer.value_embedding:
        pytorch_module.value_dense.weight.data = torch.tensor(keras_layer.value.get_weights()[0].T)

    #Load positional embedding weights
    if keras_layer.use_positional_embedding:
        load_positionaL_embedding_layer_weights(keras_layer.att.query_positional_embedding,
                                                pytorch_module.attention.query_positional_embedding[f"Query_Embedding_attention"])

    if keras_layer.use_key_positional_embedding:
        load_positionaL_embedding_layer_weights(keras_layer.att.key_positional_embedding,
                                                pytorch_module.attention.key_positional_embedding[f"Key_Embedding_attention"])
    # Load output dense weights and bias
    pytorch_module.dense_out.weight.data = torch.tensor(keras_layer.out.get_weights()[0].T)
    pytorch_module.dense_out.bias.data = torch.tensor(keras_layer.out.get_weights()[1])
