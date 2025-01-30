
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
This pytorch implementation of relative positional encoding from MVit2 was
adapted from the original pytorch implementation:
https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py

Adapted by Paul Thompson 2024
"""

import torch
import torch.nn as nn

from typing import Tuple


class RelativePositionalEmbedding2D(nn.Module):
    def __init__(self,
                 query_shape: Tuple[int, int, int, int],
                 memory_shape: Tuple[int, int, int, int],
                 query_dim: int,
                 heads: int,
                 drop_rate=0.0):
        super(RelativePositionalEmbedding2D, self).__init__()

        query_seq_len, query_height, query_width, query_channels = query_shape
        key_seq_len, key_height, key_width, key_channels = memory_shape

        self.heads = heads

        self.query_seq_len = query_seq_len
        self.key_seq_len = key_seq_len

        self.query_height = query_height
        self.query_width = query_width

        self.key_height = key_height
        self.key_width = key_width

        self.query_dim = query_dim
        self.drop_rate = drop_rate

        # Calculate max distance for each dimension
        max_height_dist = max(self.query_height, self.key_height) - 1
        max_width_dist = max(self.query_width, self.key_width) - 1
        max_t_dist = max(self.query_seq_len, self.key_seq_len) - 1

        self.height_embeddings = nn.Parameter(
            torch.randn(2 * max_height_dist + 1, self.query_dim)
        )
        self.width_embeddings = nn.Parameter(
            torch.randn(2 * max_width_dist + 1, self.query_dim)
        )
        self.time_embeddings = nn.Parameter(
            torch.randn(2 * max_t_dist + 1, self.query_dim)
        )

        self.height_dropout = nn.Dropout(p=self.drop_rate)
        self.width_dropout = nn.Dropout(p=self.drop_rate)
        self.time_dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, query, scores):
        q_h = self.query_height
        k_h = self.key_height
        q_h_ratio = round(max(k_h / q_h, 1.0))
        k_h_ratio = round(max(q_h / k_h, 1.0))
        dist_h = (
            torch.arange(q_h).unsqueeze(1) * q_h_ratio - torch.arange(k_h).unsqueeze(0) * k_h_ratio
        )
        dist_h += (k_h - 1) * k_h_ratio # [qh, kh]

        q_w = self.query_width
        k_w = self.key_width
        q_w_ratio = round(max(k_w / q_w, 1.0))
        k_w_ratio = round(max(q_w / k_w, 1.0))
        dist_w = (
            torch.arange(q_w).unsqueeze(1) * q_w_ratio - torch.arange(k_w).unsqueeze(0) * k_w_ratio
        )
        dist_w += (k_w - 1) * k_w_ratio # [qw, kw]

        q_t_ratio = round(max(self.key_seq_len / self.query_seq_len, 1.0))
        k_t_ratio = round(max(self.query_seq_len / self.key_seq_len, 1.0))

        dist_t = torch.arange(1).unsqueeze(1) * q_t_ratio - torch.arange(self.seq_len).unsqueeze(0) * k_t_ratio
        dist_t += (self.key_seq_len - 1) * k_t_ratio # [q_seq_len, key_seq_len]

        Rh = self.height_embeddings[dist_h]
        Rw = self.width_embeddings[dist_w]
        Rt = self.time_embeddings[dist_t]

        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)

        query_reshaped = query.view(-1, self.heads, self.query_seq_len, q_h, q_w, self.query_dim)

        rel_h = torch.einsum("bNthwc,hkc->bNthwk", query_reshaped, Rh)
        rel_w = torch.einsum("bNthwc,wkc->bNthwk", query_reshaped, Rw)
        rel_t = torch.einsum("bNthwc,tkc->bNthwk", query_reshaped, Rt)

        scores = scores.view(-1, self.heads, self.query_seq_len, q_h, q_w, self.seq_len, k_h, k_w)

        scores += rel_h[:, :, :, :, :, None, :, None]
        scores += rel_w[:, :, :, :, :, None, None, :]
        scores += rel_t[:, :, :, :, :, :, None, None]

        scores = scores.view(
            -1,
            self.heads,
            self.query_seq_len * q_h * q_w,
            self.key_seq_len * k_h * k_w
        )

        return scores


class RelativePositionalEmbedding2DKey(nn.Module):
    def __init__(self,
                 query_shape: Tuple[int, int, int, int],
                 memory_shape: Tuple[int, int, int, int],
                 query_dim: int,
                 heads: int,
                 drop_rate=0.0):
        super(RelativePositionalEmbedding2DKey, self).__init__()

        query_seq_len, query_height, query_width, query_channels = query_shape

        self.heads = heads

        key_seq_len, key_height, key_width, key_channels = memory_shape

        self.query_seq_len = query_seq_len
        self.key_seq_len = key_seq_len

        self.query_height = query_height
        self.query_width = query_width

        self.key_height = key_height
        self.key_width = key_width

        self.query_dim = query_dim
        self.drop_rate = drop_rate

        max_height_dist = max(self.query_height, self.key_height) - 1
        max_width_dist = max(self.query_width, self.key_width) - 1
        max_time_dist = max(self.query_seq_len, self.key_seq_len) - 1

        self.height_embeddings = nn.Parameter(
            torch.randn(2 * max_height_dist + 1, self.query_dim)
        )
        self.width_embeddings = nn.Parameter(
            torch.randn(2 * max_width_dist + 1, self.query_dim)
        )
        self.time_embeddings = nn.Parameter(
            torch.randn(2 * max_time_dist + 1, self.query_dim)
        )

        self.height_dropout = nn.Dropout(p=self.drop_rate)
        self.width_dropout = nn.Dropout(p=self.drop_rate)
        self.time_dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, key, scores):

        q_h = self.query_height
        k_h = self.key_height

        q_h_ratio = round(max(k_h / q_h, 1.0))
        k_h_ratio = round(max(q_h / k_h, 1.0))

        dist_h = (
            torch.arange(k_h).unsqueeze(1) * k_h_ratio - torch.arange(q_h).unsqueeze(0) * q_h_ratio
        )
        dist_h += (q_h - 1) * q_h_ratio # [kh, qh]

        q_w = self.query_width
        k_w = self.key_width

        q_w_ratio = round(max(k_w / q_w, 1.0))
        k_w_ratio = round(max(q_w / k_w, 1.0))

        dist_w = (
            torch.arange(k_w).unsqueeze(1) * k_w_ratio - torch.arange(q_w).unsqueeze(0) * q_w_ratio
        )
        dist_w += (q_w - 1) * q_w_ratio # [kw, qw]

        q_t_ratio = round(max(self.key_seq_len / self.query_seq_len, 1.0))
        k_t_ratio = round(max(self.query_seq_len / self.key_seq_len, 1.0))

        dist_t = (
            torch.arange(self.key_seq_len).unsqueeze(1) * k_t_ratio - torch.arange(self.query_seq_len).unsqueeze(0) * q_t_ratio
        )
        dist_t += (self.query_seq_len - 1) * q_t_ratio # [key_seq_len, q_seq_len]

        Rh = self.height_embeddings[dist_h]
        Rw = self.width_embeddings[dist_w]
        Rt = self.time_embeddings[dist_t]

        Rh = self.height_dropout(Rh)
        Rw = self.width_dropout(Rw)
        Rt = self.time_dropout(Rt)

        key_reshaped = key.view(-1, self.heads, self.key_seq_len, self.key_height, self.key_width, self.query_dim)

        rel_h = torch.einsum("hqc,bNshwc->bNqshw", Rh, key_reshaped)
        rel_w = torch.einsum("wqc,bNshwc->bNqshw", Rw, key_reshaped)
        rel_t = torch.einsum("sqc,bNshwc->bNqshw", Rt, key_reshaped)

        scores = scores.view(-1, self.heads, self.query_seq_len, q_h, q_w, self.key_seq_len, k_h, k_w)

        scores += rel_h[:, :, None, :, None, :, :, :]  # [B, h, qs, qh, qw, seq_len, kh, kw]
        scores += rel_w[:, :, None, None, :, :, :, :]
        scores += rel_t[:, :, :, None, None, :, :, :]

        scores = scores.view(
            -1,
            self.heads,
            self.query_seq_len * q_h * q_w,
            self.key_seq_len * k_h * k_w
        )

        return scores
