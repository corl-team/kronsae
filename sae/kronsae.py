import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import einops
import math

from .config import SAEConfig
from .model import BaseSAE
from .utils import *

class KronSAE(BaseSAE):
 
    def __init__(self, config: SAEConfig):
        super().__init__(config)
        self.config = config

        assert self.config.cartesian_op in ["sum", "mul"]
        assert self.config.dict_size == self.config.num_heads * self.config.num_mkeys * self.config.num_nkeys
        
        self.num_keys = None
        self.p = self.config.router_depth
        self.h = self.config.num_heads
        self.m, self.n = self.config.num_mkeys, self.config.num_nkeys

        self.W_enc = nn.Parameter(torch.empty(
            self.config.act_size,
            self.config.num_heads * (self.m + self.n),
        ))
        nn.init.normal_(
            self.W_enc,
            mean=0.0,
            std=math.sqrt(2.0 / self.config.dict_size),
        )
        self.b_enc = nn.Parameter(torch.zeros(self.config.num_heads * (self.m + self.n)))

        decoder_parts = self.W_enc.detach().T.reshape(
            self.h,
            self.m + self.n,
            self.config.act_size,
        )
        W_dec_v_base, W_dec_v_ext = decoder_parts.split(
            [self.m, self.n],
            dim=1,
        )
        cartesian = (
            W_dec_v_base.unsqueeze(2)
            + W_dec_v_ext.unsqueeze(1)
        ).flatten(0, 2)
        self.W_dec = nn.Parameter(
            F.normalize(cartesian, dim=-1)
        )

        self.b_dec = nn.Parameter(torch.zeros(self.config.act_size))

        self.register_buffer('num_batches_not_active', torch.zeros((self.config.dict_size,)))
        self.to(self.config.get_torch_dtype(self.config.dtype))

        # Initialize input normalization parameters if provided
        if config.input_mean is not None and config.input_std is not None:
            self.register_buffer('input_mean', torch.tensor(config.input_mean))
            self.register_buffer('input_std', torch.tensor(config.input_std))
        else:
            self.input_mean = None
            self.input_std = None

        # Initialize CUDA events for timing
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self.end_event = None

    def _standard_expert_retrieval(self, acts: torch.Tensor):
        B, H, MN = acts.shape
        m_acts = acts[..., :self.m].contiguous()
        n_acts = acts[..., self.m:].contiguous()

        # mand variant
        all_scores = torch.sqrt(m_acts[..., None] * n_acts[..., None, :] + 1e-5).view(B, -1)

        # top-k to choose final K candidates of K^2
        scores, indices = all_scores.topk(self.config.topk, dim=-1, sorted=False)

        return all_scores, scores, indices

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)

        _, acts_topk = self.encode_forward(x)
        return acts_topk

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs into the pre-latents used to build KronSAE features."""
        if self.config.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)

        B, D = x.shape
        return F.relu(x @ self.W_enc + self.b_enc).view(B, self.h, self.m + self.n)

    def get_masked_pre_acts(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Keep only pre-latents that contribute to selected post-latents."""
        B, h, mn = pre_acts.shape
        if h != self.h or mn != self.m + self.n:
            raise ValueError(
                f"Expected pre-latents with shape (batch, {self.h}, {self.m + self.n}), "
                f"got {tuple(pre_acts.shape)}"
            )

        with torch.no_grad():
            _, _, indices = self._standard_expert_retrieval(pre_acts)

        head_indices = indices // (self.m * self.n)
        positions = indices % (self.m * self.n)
        row_indices = positions // self.n
        column_indices = positions % self.n
        batch_indices = torch.arange(B, device=pre_acts.device)[:, None].expand_as(indices)

        mask_m = torch.zeros((B, self.h, self.m), dtype=torch.bool, device=pre_acts.device)
        mask_n = torch.zeros((B, self.h, self.n), dtype=torch.bool, device=pre_acts.device)
        mask_m[batch_indices, head_indices, row_indices] = True
        mask_n[batch_indices, head_indices, column_indices] = True

        m_acts = pre_acts[..., :self.m] * mask_m
        n_acts = pre_acts[..., self.m:] * mask_n
        return torch.cat([m_acts, n_acts], dim=-1)
    
    def encode_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        acts = F.relu(x @ self.W_enc + self.b_enc).view(B, self.h, self.m + self.n)

        acts, scores, indices = self._standard_expert_retrieval(acts)
        acts_topk = torch.zeros((B, self.config.dict_size), device=scores.device, dtype=scores.dtype).scatter(
            -1, indices, scores,
        )
        return acts, acts_topk
    
    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def decode(self, acts_topk: torch.Tensor):
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        return x_reconstruct

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        acts, acts_topk = self.encode_forward(x)
        x_reconstruct = self.decode(acts_topk)
        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.config.l1_coeff * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        
        loss = l2_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.config.n_batches_to_dead
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        per_token_l2_loss_A = (x_reconstruct.float() - x.float()).pow(2).sum(-1).squeeze()
        total_variance_A = (x.float() - x.float().mean(0)).pow(2).sum(-1).squeeze()
        explained_variance = (1 - per_token_l2_loss_A / total_variance_A).mean()
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "pre_topk_mean": acts.mean(),
            "pre_topk_std": acts.std(),
            "pre_topk_mean_group_std": acts.std(dim=-1).mean(),
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "explained_variance": explained_variance,
            "topk": self.config.topk,
            "aux_loss": aux_loss,
        }
        return output
    
    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.config.n_batches_to_dead
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.config.topk_aux, dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.config.aux_penalty
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)

KronSAE.register_for_auto_class()
