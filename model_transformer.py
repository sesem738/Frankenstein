import math
from collections import deque

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from typing import Optional

DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)

#         # This PE implementation is 5x faster than following Attention is All You Need formula.
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
#         )
#         # add positional embedding directly to the state/state+action embedding
#         # note: dim is a bit different from NLP embeddings
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2, :] = torch.sin(position * div_term)
#         pe[:, 1::2 ] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[: x.size(0)]
#         return self.dropout(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ResidueGate(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

class BERT(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, encoder_norm
        )

        self.d_model = d_model

        # self.decoder = nn.Linear(d_model, ntoken)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, generate_square_subsequent_mask(src.size(-2))
        )
        # print("out",output)
        # print(output.shape)
        # output = self.decoder(output)
        return output

class GTrXLLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        n_layers=1,
        dim_feedforward=256,
        activation="relu",
        dropout=0,
        layer_norm_eps=1e-5,
        batch_first=True,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
        )
        self.gru1 = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.gru2 = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        h = (src).sum(dim=1).unsqueeze(dim=0)
        src = self.norm1(src)
        attn = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        emb, h = self.gru1(attn, h)

        mlp = self.norm2(emb)
        mlp = self.activation(self.linear1(mlp))
        mlp = self.activation(self.linear2(mlp))

        output, h = self.gru2(mlp, h)

        return output


class GTrXL(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        layer_norm_eps: float = 1e-5,
    ):
        super(GTrXL, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = GTrXLLayer(d_model, nhead, 1, dim_feedforward=d_hid)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers, encoder_norm
        )

        self.d_model = d_model

        # self.decoder = nn.Linear(d_model, ntoken)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, generate_square_subsequent_mask(src.size(-2))
        )
        return output