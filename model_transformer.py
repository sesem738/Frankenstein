import math
from collections import deque

import numpy as np
import torch 
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from torchvision import models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # add positional embedding directly to the state/state+action embedding
        # note: dim is a bit different from NLP embeddings
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
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
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(DEVICE)

class BERT_Torch(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 layer_norm_eps: float = 1e-5):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, encoder_norm)

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
        output = self.transformer_encoder(src, generate_square_subsequent_mask(src.size(0)))
        # output = self.decoder(output)
        return output