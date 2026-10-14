from toy_models import *

from torch import Tensor
import torch.nn.functional as F
import torch.autograd as autograd

from typing import Optional, Dict, Union

import einops
import math

class BinarySignSTE(torch.autograd.Function):
    """STE for binarizing values to -1 or 1 while preserving gradients."""
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        return torch.where(input >= 0, torch.ones_like(input), -torch.ones_like(input))
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


def ste_sign(x: Tensor) -> Tensor:
    """Applies the STE-based sign function."""
    return BinarySignSTE.apply(x)


def smooth_sign(x: Tensor, b: Tensor = 20) -> Tensor:
    s = x / torch.sqrt(x**2 + 1 / b**3)
    return s


def or_op(u: torch.Tensor, v: torch.Tensor, normalize_act: bool) -> torch.Tensor:
    """Implements the OR operation in logit space."""
    # see https://arxiv.org/pdf/2110.11940 for analytic derivation
    mu = 0.6810370721753108  # 1/sqrt(2*pi)+1/(2*sqrt(pi))
    sig = 0.9453434494147759  # 5/4-1/(sqrt(2)*pi)-1/(4*pi)
    cond = (u > 0) & (v > 0)
    logical_acts = torch.where(cond, u + v, torch.maximum(u, v)).type_as(u)
    return logical_acts.sub_(mu).div_(sig) if normalize_act else logical_acts


def and_op(u: Tensor, v: Tensor, normalize_act: bool) -> Tensor:
    """Implements the AND operation in logit space."""
    # see https://arxiv.org/pdf/2110.11940 for analytic derivation
    mu = -0.6810370721753108  # -1/sqrt(2*pi)-1/(2*sqrt(pi))
    sig = 0.9722877400310959  # sqrt(5/4-1/(sqrt(2)*pi)-1/(4*pi))
    cond = (u < 0) & (v < 0)
    logical_acts = torch.where(cond, u + v, torch.minimum(u, v)).type_as(u)
    return logical_acts.sub_(mu).div_(sig) if normalize_act else logical_acts


def xnor_op(u: torch.Tensor, v: torch.Tensor, normalize_act: bool) -> torch.Tensor:
    """Implements the OR operation in logit space."""
    # see https://arxiv.org/pdf/2110.11940 for analytic derivation
    mu = 0.6810370721753108  # 1/sqrt(2*pi)+1/(2*sqrt(pi))
    sig = 0.9453434494147759  # 5/4-1/(sqrt(2)*pi)-1/(4*pi)
    logical_acts = torch.sgn(u*v) * (torch.minimum(torch.abs(u), torch.abs(v)))
    return logical_acts.sub_(mu).div_(sig) if normalize_act else logical_acts


def mand_op(u: torch.Tensor, v: torch.Tensor, normalize_act: bool) -> torch.Tensor:
    """Implements the AND operation in logit space."""
    # see https://arxiv.org/pdf/2110.11940 for analytic derivation
    mu = -0.6810370721753108  # -1/sqrt(2*pi)-1/(2*sqrt(pi))
    sig = 0.9722877400310959  # sqrt(5/4-1/(sqrt(2)*pi)-1/(4*pi))
    cond = (u > 0) & (v > 0)
    logical_acts = torch.where(cond, u * v, torch.minimum(u, v)).type_as(u)
    return logical_acts


class LogicMixer(nn.Module):
    """
    A neural module that learns binary logic operations using an AND basis with STE-based negation.
    It accepts inputs with shape (B, H, N) and (B, H, M) respectively. If y is None, x is used for both.
    Logical operations are computed independently for each head, and K sets of weights are learned for each head.
    
    The output shape is (B, H, K, N, M) where K is the number of sets of weights.
    """
    def __init__(self, M: int, N: int = None, H: int = 1, K: int = 1, use_ste: bool = True, normalize_act: bool = True):
        super(LogicMixer, self).__init__()
        self.use_ste = use_ste
        self.normalize_act = normalize_act
        # self.s = nn.Parameter(torch.randn(6, K, H, M, N or M))
        self.s = nn.Parameter(torch.randn(10, K, H, M, N or M) * 0.1 + 0.3)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # encoder weights should be initialized as:
        # nn.init.kaiming_normal_(W_enc.weight, nonlinearity='linear')
        # nn.init.zeros_(W_enc.bias)
        # decoder weights for op(F1, F2) should be initialized as: (F1 + F2).normalize(dim=-1)

        # x: (B, H, N); y: (B, H, M) if provided, otherwise use x for both.
        x_i = x.unsqueeze(1).unsqueeze(-1)  # (B, 1, H, N, 1)
        x_j = (y if y is not None else x).unsqueeze(1).unsqueeze(-2)  # (B, 1, H, 1, M)
        
        #s_all = (ste_sign(self.s) if self.use_ste else self.s).unsqueeze(0)  #  (1, 6, K, H, N, M)
        s_all = (smooth_sign(self.s) if self.use_ste else self.s).unsqueeze(0)  #  (1, 6, K, H, N, M)
        
        # Compute intermediate logical operations:
        a = and_op(s_all[:, 0] * x_i, s_all[:, 1] * x_j, normalize_act=self.normalize_act)  # (B, K, H, N, M)
        b = and_op(s_all[:, 2] * x_i, s_all[:, 3] * x_j, normalize_act=self.normalize_act)  # (B, K, H, N, M)
        dict_acts = and_op(s_all[:, 4] * a, s_all[:, 5] * b, normalize_act=self.normalize_act)  # (B, K, H, N, M)
        
        # Permute dimensions to yield (B, H, K, N, M)
        dict_acts = dict_acts.permute(0, 2, 1, 3, 4)
        return dict_acts


class LogicTopKSAE(nn.Module):
 
    def __init__(self, input_dim: int, dict_size: int, topk: int, num_heads: int, m_keys: int, n_keys):
        super().__init__()

        self.num_keys = int(math.sqrt(dict_size // num_heads))
        self.m = m_keys
        self.n = n_keys
        
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
        W_dec_data = _t.contiguous().t().clone()
        W_dec_v0 = einops.rearrange(W_dec_data, '(h mn) d -> h mn d', h=num_heads, mn=self.m + self.n)[:, :self.m]
        W_dec_v1 = einops.rearrange(W_dec_data, '(h mn) d -> h mn d', h=num_heads, mn=self.m + self.n)[:, self.m:]
        cartesian = (W_dec_v0[..., None, :] + W_dec_v1[..., None, :, :])
        cartesian = einops.rearrange(cartesian, 'h m n d -> (h m n) d')
        
        self.W_dec = nn.Parameter(
            cartesian
        )
        self.W_dec.data[:] = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True)

        self.logic_mixer = LogicMixer(M=self.m, N=self.n, K=1, H=self.h, use_ste=True)

        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self.topk = topk
        
    def _standard_expert_retrieval(self, acts: torch.Tensor):
        B, H, MN = acts.shape

        # scores_a = einops.repeat(acts, 'B H k -> B (H k r)', r=D)
        # scores_b = einops.repeat(acts, 'B H k -> B (H r k)', r=D)
        # scores_ab = einops.rearrange(all_scores, 'B H k1 k2 -> B (H k1 k2)')
        m_acts = acts[..., :self.m].contiguous()
        n_acts = acts[..., self.m:self.m+self.n].contiguous()
        all_scores = F.relu(self.logic_mixer(m_acts, n_acts).view(B, -1))

        # top-k to choose final K candidates of K^2
        scores, indices = all_scores.topk(k=self.topk, dim=-1, sorted=False)
        return all_scores, scores, indices
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor to sparse features with top-k
        """
        B, H = x.shape

        acts = (x @ self.W_enc + self.b_enc).view(B, self.h, self.m + self.n)
        acts, scores, indices = self._standard_expert_retrieval(acts) 
        
        acts_topk = torch.zeros((B, self.dict_size), device=scores.device, dtype=scores.dtype).scatter(
            -1, indices, scores, 
        )
        
        return acts_topk
    
    def decode(self, acts_topk: torch.Tensor):
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        return x_reconstruct

    def forward(self, x):
        acts_topk = self.encode(x)
        x_reconstruct = self.decode(acts_topk)
        return x_reconstruct
