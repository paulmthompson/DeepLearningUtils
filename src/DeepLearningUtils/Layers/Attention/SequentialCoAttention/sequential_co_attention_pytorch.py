

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from src.DeepLearningUtils.Layers.Attention.MHA_mvit2.attention_pytorch import MultiHeadAttention

class CoAttentionModule(nn.Module):
    def __init__(self,
                 memory_attention_module,
                 query_shape: Tuple[int, int, int, int, int],
                 memory_shape: Tuple[int, int, int, int, int],
                 key_dim=128,
                 value_dim=128,

                 ):
        super(CoAttentionModule, self).__init__()
        self.memory_attention_module = memory_attention_module

        self.layer_norm1 = nn.LayerNorm(memory_shape[-1], eps=1e-3)

        n_frames = memory_shape[1]

        self.query_norms = nn.ModuleList()
        self.query_denses = nn.ModuleList()

        self.query_norms.append(nn.LayerNorm(query_shape[-1], eps=1e-3))
        self.query_denses.append(nn.Linear(query_shape[-1], key_dim, bias=False))
        for i in range(1, n_frames):
            self.query_norms.append(nn.LayerNorm(key_dim, eps=1e-3))
            self.query_denses.append(nn.Linear(key_dim, key_dim, bias=False))

        self.key_dense = nn.Linear(memory_shape[-1], key_dim, bias=False)
        self.value_dense = nn.Linear(memory_shape[-1], value_dim, bias=False)

        self.out_dense = nn.Linear(key_dim, query_shape[-1], bias=False)
        self.layer_norm2 = nn.LayerNorm(query_shape[-1], eps=1e-3)

        hidden_dims = 512
        self.mlp1 = nn.Linear(query_shape[-1], hidden_dims)
        self.mlp2 = nn.Linear(hidden_dims, query_shape[-1])

    def forward(self, query_sequence, memory_bank_sequence, mask):
        memory_bank_sequence = self.layer_norm1(memory_bank_sequence)

        key = self.key_dense(memory_bank_sequence)
        value = self.value_dense(memory_bank_sequence)

        memory_bank_frames = torch.split(memory_bank_sequence, 1, dim=1)

        attention_results = []
        att_input = query_sequence
        for i, (query_norm, query_dense) in enumerate(zip(self.query_norms, self.query_denses)):
            key = self.key_dense(memory_bank_frames[i])
            value = self.value_dense(memory_bank_frames[i])
            att_input = query_norm(att_input)
            att_input = query_dense(att_input)
            attention_result = self.memory_attention_module(att_input, key, value, mask[:, i:i+1])
            attention_result *= mask[:, i:i+1, None, None, None]
            attention_result = attention_result + att_input
            attention_results.append(attention_result)
            att_input = attention_result

        att_out = torch.stack(attention_results, dim=1).mean(dim=1)
        att_out = self.out_dense(att_out)
        att_out = att_out + query_sequence

        att_out = self.layer_norm2(att_out)
        att_out = F.hardswish(self.mlp1(att_out))
        att_out = self.mlp2(att_out)

        return att_out


class CoMemoryAttentionModule(nn.Module):
    def __init__(self,
                 query_shape: Tuple[int, int, int, int],
                 memory_shape: Tuple[int, int, int, int],
                 key_dim=128,
                 value_dim=256,
                 use_norm=False,
                 attention_drop_rate=0.0,
                 use_positional_embedding=True,
                 use_key_positional_embedding=True,
                 attention_heads=8,
                 use_qkv_embedding=False):
        super(CoMemoryAttentionModule, self).__init__()
        query_seq_len, query_height, query_width, query_channels = query_shape
        key_seq_len, key_height, key_width, key_channels = memory_shape
        self.query_height: int = query_height
        self.query_width: int = query_width
        self.key_height: int = key_height
        self.key_width: int = key_width
        self.query_seq_len: int = query_seq_len
        self.key_seq_len: int = key_seq_len
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_norm = use_norm
        self.attention_drop_rate = attention_drop_rate
        self.use_positional_embedding = use_positional_embedding
        self.use_key_positional_embedding = use_key_positional_embedding
        self.attention_heads = attention_heads
        self.use_qkv_embedding = use_qkv_embedding

        if self.use_norm:
            self.query_norm = nn.LayerNorm(query_channels, eps=1e-3)
            self.memory_norm = nn.LayerNorm(key_channels, eps=1e-3)
        else:
            self.query_norm = nn.Identity()
            self.memory_norm = nn.Identity()

        self.att = MultiHeadAttention(
            query_shape,
            memory_shape,
            heads=self.attention_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim,
            attention_drop_rate=self.attention_drop_rate,
            use_positional_embedding=self.use_positional_embedding,
            use_key_positional_embedding=self.use_key_positional_embedding,
            use_query_embedding=self.use_qkv_embedding,
            use_key_embedding=self.use_qkv_embedding,
            use_value_embedding=self.use_qkv_embedding
        )

    def forward(self, encoder_output, memory_key, memory_value, memory_mask):

        assert len(encoder_output.shape) == 5
        assert len(memory_key.shape) == 5
        assert len(memory_value.shape) == 5

        query_encoder = encoder_output

        query_encoder = self.query_norm(query_encoder)
        memory_key = self.memory_norm(memory_key)
        memory_value = self.memory_norm(memory_value)

        mh_mask = create_encoder_memory_mask(
            memory_mask,
            self.query_height * self.query_width * self.query_seq_len,
            self.key_height * self.key_width * self.key_seq_len,
            self.query_seq_len,
            self.key_seq_len,
            self.attention_heads
        )

        query_encoder = query_encoder.view(
            -1,
            self.query_seq_len * self.query_height * self.query_width,
            query_encoder.shape[-1])
        memory_key = memory_key.view(
            -1,
            self.key_height * self.key_width * self.key_seq_len,
            memory_key.shape[-1])
        memory_value = memory_value.view(
            -1,
            self.key_height * self.key_width * self.key_seq_len,
            memory_value.shape[-1])

        att_out = self.att(query_encoder, memory_key, memory_value, mask=mh_mask)

        att_out = att_out.view(
            -1,
            self.query_seq_len,
            self.query_height,
            self.query_width,
            att_out.shape[-1])

        return att_out


def create_encoder_memory_mask(
        mask,
        encoder_seq_length: int,
        key_seq_length: int,
        query_frame_num: int,
        key_frame_num: int,
        num_heads: int) -> torch.Tensor:

    # Assuming mask is a tensor of shape [batch_size, seq_len]
    mask = mask.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1,  seq_len, 1]
    mask = torch.tile(mask, [1, encoder_seq_length // query_frame_num, 1,
                                 key_seq_length // key_frame_num])  # [batch_size, encoder_seq_length, seq_len]
    mask = mask.view(mask.shape[0], encoder_seq_length, key_seq_length)  # [batch_size, encoder_seq_length, key_seq_length]

    mask = mask.unsqueeze(1)  # [batch_size, 1,  encoder_seq_length, key_seq_length]
    mask = torch.tile(mask, [1, num_heads, 1, 1])  # [batch_size, num_heads, encoder_seq_length, key_seq_length]
    return mask