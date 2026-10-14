import os
import random
import math
import torch
import einops

import torch.nn as nn

import torch.nn.functional as F


class SymAE(nn.Module):
    """
    Symmetrical Autoencoder:
      Encoder: Linear(N -> M)
      Decoder: Linear(M -> N), final ReLU
    """
    def __init__(self, input_dim: int, latent_dim: int, sym: bool = True, is_relu : bool = True):
        super().__init__()
        self.sym = sym
        self.is_relu = is_relu

        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(input_dim, latent_dim), nonlinearity='linear'
            )
        )


        if not sym:
            self.W_dec = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(latent_dim, input_dim), nonlinearity='relu'
                )
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.sym:
            h = x @ self.W_enc + self.b_enc
        else:
            h = x @ self.W_enc
        return h

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.sym:
            h = x @ self.W_dec + self.b_dec
        else:
            h = x @ self.W_enc.T + self.b_dec
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        x_recon = self.decode(h)
        if self.is_relu:
            x_recon = F.relu(x_recon)
        return x_recon


class TopKSAE(nn.Module):
    """
    A "top-k SAE" that tries to reconstruct a vector in R^latent_dim but enforces
    a top-k activation in the bottleneck or hidden layer.

    Architecture:
      - Encoder: Linear(M -> hidden_dim)
      - Apply top-k
      - Decoder: Linear(hidden_dim -> M)
    """
    def __init__(self, input_dim: int, dict_size: int, topk: int):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(input_dim, dict_size), nonlinearity='linear'
            )
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(dict_size, input_dim), nonlinearity='relu'
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.topk = topk
        self.dict_size = dict_size

    def topk_activation(self, x: torch.Tensor, k: int):
        acts = F.relu(x)
        acts_topk = torch.topk(acts, k, dim=-1)
        return torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W_enc + self.b_enc
        h_topk = self.topk_activation(h, self.topk)
        return h_topk

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W_dec + self.b_dec
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_topk = self.encode(x)
        x_rec = self.decode(h_topk)
        return x_rec


class KronSAE(nn.Module):
    """
    A "top-k SAE" that tries to reconstruct a vector in R^latent_dim but enforces
    a top-k activation in the bottleneck or hidden layer.

    Architecture:
      - Encoder: Linear(M -> hidden_dim)
      - Apply top-k
      - Decoder: Linear(hidden_dim -> M)
    """
    def __init__(self, input_dim: int, dict_size: int, topk: int, num_heads: int, m_keys: int, n_keys):
        super().__init__()

        router_depth = 2
        self.num_keys = int(math.sqrt(dict_size // num_heads))
        self.m = m_keys
        self.n = n_keys
        
        self.p = router_depth
        self.h = num_heads
        self.dict_size = dict_size
        
        _t = torch.nn.init.normal_(
                torch.empty(input_dim, num_heads * (self.m + self.n)) 
            ) / math.sqrt(dict_size) * math.sqrt(2.0) ## fan_in for enc

        self.W_enc = nn.Parameter(
            _t 
        )
        self.b_enc = nn.Parameter(torch.zeros(num_heads * (self.m + self.n)))

        # Init decoder
        W_dec_data = _t.t().clone()
        W_dec_v0 = einops.rearrange(W_dec_data, '(h mn) d -> h mn d', h=num_heads, mn=self.m + self.n)[:, :self.m]
        W_dec_v1 = einops.rearrange(W_dec_data, '(h mn) d -> h mn d', h=num_heads, mn=self.m + self.n)[:, self.m:]
        cartesian = (W_dec_v0[..., None, :] + W_dec_v1[..., None, :, :])
        cartesian = einops.rearrange(cartesian, 'h m n d -> (h m n) d')
        
        self.W_dec = nn.Parameter(
            cartesian
        )
        self.W_dec.data[:] = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True)

        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.topk = topk
    
    def _standard_expert_retrieval(self, acts: torch.Tensor):
        B, H, MN = acts.shape
        m_acts = acts[..., :self.m].contiguous()
        n_acts = acts[..., self.m:].contiguous()
        scores_x, scores_y = m_acts, n_acts
        all_scores = torch.sqrt(scores_x[..., None] * scores_y[..., None, :] + 1e-5).view(B, -1)

        # #top-k to choose final K candidates of K^2
        k = self.topk
        if k > all_scores.shape[-1]:
            k = all_scores.shape[-1]

        scores, indices = all_scores.topk(k, dim=-1, sorted=False)

        return scores, indices

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        acts = F.relu(x @ self.W_enc + self.b_enc).view(B, self.h, self.m + self.n)

        scores, indices = self._standard_expert_retrieval(acts)
        acts_topk = torch.zeros((B, self.dict_size), device=scores.device, dtype=scores.dtype).scatter(
            -1, indices, scores,
        )

        return acts_topk

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W_dec + self.b_dec
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_topk = self.encode(x)
        x_rec = self.decode(h_topk)
        return x_rec
