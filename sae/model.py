# MIT License

# Copyright (c) 2025 bartbussmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from transformers import PreTrainedModel
from typing import Optional, Dict, Union

import torch
import einops

import torch.nn as nn
import torch.nn.functional as F
from .config import SAEConfig
from math import sqrt, isqrt, log2
import math 

from .utils import *


class BaseSAE(PreTrainedModel):
    """Base class for autoencoder models."""
    config_class = SAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: SAEConfig):
        super().__init__(config)
        # print(config)
        self.config = config

        _t = torch.nn.init.normal_(
            torch.empty(self.config.act_size, self.config.dict_size) 
        ) 
        
        self.b_dec = nn.Parameter(torch.zeros(self.config.act_size))
        self.b_enc = nn.Parameter(torch.zeros(self.config.dict_size))
        # self.W_enc = nn.Parameter(
        #     torch.nn.init.kaiming_uniform_(
        #         torch.empty(self.config.act_size, self.config.dict_size)
        #     )
        #     # _t / math.sqrt(self.config.act_size) * math.sqrt(2)
        # )
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(self.config.act_size, self.config.dict_size), nonlinearity='relu'
            )
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config.dict_size, self.config.act_size)
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
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

    def preprocess_input(self, x):
        x = x.to(self.config.get_torch_dtype(self.config.sae_dtype))
        if self.config.input_unit_norm:
            if self.input_mean is not None and self.input_std is not None:
                # Use pre-computed statistics
                x = (x - self.input_mean) / (self.input_std + 1e-5)
                return x, self.input_mean, self.input_std
            else:
                if self.config.batch_norm_on_queries:
                    # Compute mean and variance along batch dimension (B)
                    x_mean = x.mean(dim=0, keepdim=True)  # Shape: (1, P, H)
                    x = x - x_mean
                    x_std = x.std(dim=0, keepdim=True)  # Shape: (1, P, H)
                    x = x / (x_std + 1e-5)
                else:
                    # Compute statistics on the fly
                    x_mean = x.mean(dim=-1, keepdim=True)
                    x = x - x_mean
                    x_std = x.std(dim=-1, keepdim=True)
                    x = x / (x_std + 1e-5)
                return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.config.input_unit_norm:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):

        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor to sparse features
        Args:
            x: Input tensor of shape (batch_size, act_size)
        Returns:
            Encoded features of shape (batch_size, dict_size)
        """
        if self.config.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
        
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to input space
        Args:
            h: Encoded features of shape (batch_size, dict_size)
        Returns:
            Reconstructed input of shape (batch_size, act_size)
        """
        return h @ self.W_dec + self.b_dec

    def forward(self, x):
        # Start timing if CUDA is available
        if self.start_event is not None:
            self.start_event.record()
            
        x, x_mean, x_std = self.preprocess_input(x)
        acts = F.relu(x @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        
        # End timing if CUDA is available
        if self.end_event is not None:
            self.end_event.record()
            torch.cuda.synchronize()
            output["forward_time_ms"] = self.start_event.elapsed_time(self.end_event)
        
        return output

    @torch.no_grad()
    def fold_stats_into_weights(self, mean: torch.Tensor = None, std: torch.Tensor = None) -> "BaseSAE":
        """
        Fold normalization statistics into the encoder weights and biases
        """
        print("Folding statistics into encoder...")
        
        if mean is not None and std is not None:
            mean = mean.to(self.W_enc.device)
            std = std.to(self.W_enc.device)
        else:
            mean = self.input_mean
            std = self.input_std
        
        # Original forward pass:
        # x_norm = (x - mean) / std
        # acts = relu(x_norm @ W_enc + b_enc)
        # x_hat = acts @ W_dec + b_dec
        
        # Folding steps:
        # 1. x_norm = (x - mean) / std
        # 2. acts = relu(x_norm @ W_enc + b_enc)
        #        = relu((x/std - mean/std) @ W_enc + b_enc)
        #        = relu(x @ (W_enc/std) - mean @ (W_enc/std) + b_enc)
        
        # First scale encoder weights
        self.W_enc.data = self.W_enc / std
        
        # Then adjust encoder bias
        self.b_enc.data = self.b_enc - mean * (self.W_enc.sum(0))
        
        # Scale decoder to preserve reconstruction
        self.W_dec.data = self.W_dec * std
        self.b_dec.data = self.b_dec * std + mean
        
        # Turn off input normalization
        self.config.input_unit_norm = False
        
        return self
    
    @torch.no_grad()
    def fold_W_dec_norm(self):
        """
        Make decoder weights unit norm and adjust encoder accordingly
        """
        # Get current decoder norms
        W_dec_norm = self.W_dec.norm(dim=-1, keepdim=True)
        
        # Original: acts @ W_dec + b_dec
        # After: acts @ (W_dec/norm) + b_dec
        # Need: (acts * norm) @ (W_dec/norm) + b_dec
        # So: scale W_enc by norm.T
        
        # Scale encoder weights first
        self.W_enc.data = self.W_enc * W_dec_norm.t()
        
        # Then normalize decoder weights
        self.W_dec.data = self.W_dec / W_dec_norm
        
        # Scale encoder bias to compensate for the scaling of activations
        self.b_enc.data = self.b_enc * W_dec_norm.squeeze(-1)
        
        return self

    def set_mean_std(self, mean: float, std: float):
        """
        Set input normalization statistics after model initialization
        
        Args:
            mean: Mean scalar value for input normalization
            std: Standard deviation scalar value for input normalization
        """
        self.register_buffer('input_mean', torch.tensor(mean, device=self.device))
        self.register_buffer('input_std', torch.tensor(std, device=self.device))
        self.config.input_unit_norm = True
        return self


class BatchTopKSAE(BaseSAE):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor to sparse features with batch-wise top-k
        """
        if self.config.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            
        acts = F.relu(x @ self.W_enc + self.b_enc)
        acts_topk = torch.topk(acts.flatten(), self.config.topk * x.shape[0], dim=-1)
        return (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        acts = F.relu(x @ self.W_enc + self.b_enc)
        acts_topk = torch.topk(acts.flatten(), self.config.topk * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.config.l1_coeff * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts) ############
        loss = l2_loss + aux_loss #######
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
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss, #$####
            "explained_variance": explained_variance,
            "topk": self.config.topk
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


class TopKSAE(BaseSAE):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor to sparse features with per-sample top-k
        """
        if self.config.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            
        acts = F.relu(x @ self.W_enc + self.b_enc)
        acts_topk = torch.topk(acts, self.config.topk, dim=-1)
        return torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )

    def forward(self, x, x_out=None):
        # if x_out is None:
        #     x_out = x
        x, x_mean, x_std = self.preprocess_input(x)
        acts = F.relu(x @ self.W_enc + self.b_enc)
        acts_topk = torch.topk(acts, self.config.topk, dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
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
            #"acts_mean_value": acts_topk[acts_topk > 0.0].mean(),
            "pre_topk_mean": acts.mean(),
            "pre_topk_std": acts.std(),
            "pre_topk_mean_group_std": acts.std(dim=-1).mean(),
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "explained_variance": explained_variance,
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


class VanillaSAE(BaseSAE):
    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        acts = F.relu(x @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.config.l1_coeff * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.config.n_batches_to_dead
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        per_token_l2_loss_A = (x_reconstruct.float() - x.float()).pow(2).sum(-1).squeeze()
        total_variance_A = (x.float() - x.float().mean(0)).pow(2).sum(-1).squeeze()
        explained_variance = (1 - per_token_l2_loss_A / total_variance_A).mean()
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "explained_variance": explained_variance,
        }
        
        # Add timing if available
        if hasattr(self, "start_event") and self.start_event is not None:
            output["forward_time_ms"] = self.start_event.elapsed_time(self.end_event)
        
        return output


class JumpReLUSAE(BaseSAE):
    def __init__(self, config: SAEConfig):
        super().__init__(config)
        self.jumprelu = JumpReLU(
            feature_size=config.dict_size,
            bandwidth=config.bandwidth,
            device=config.device if hasattr(config, 'device') else 'cpu'
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor to sparse features using JumpReLU
        """
        if self.config.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
        
        pre_activations = F.relu(x @ self.W_enc + self.b_enc)
        return self.jumprelu(pre_activations)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)
        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec
        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.config.bandwidth).sum(dim=-1).mean()
        l0_loss = self.config.l1_coeff * l0
        l1_loss = l0_loss

        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.config.n_batches_to_dead
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        per_token_l2_loss_A = (x_reconstruct.float() - x.float()).pow(2).sum(-1).squeeze()
        total_variance_A = (x.float() - x.float().mean(0)).pow(2).sum(-1).squeeze()
        explained_variance = (1 - per_token_l2_loss_A / total_variance_A).mean()
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
            "explained_variance": explained_variance,
        }
        return output
    
    @torch.no_grad()
    def fold_W_dec_norm(self):
        """
        Make decoder weights unit norm and adjust encoder and thresholds accordingly
        """
        # Get current decoder norms
        W_dec_norm = self.W_dec.norm(dim=-1, keepdim=True)
        
        # Call parent method to handle weights
        super().fold_W_dec_norm()
        
        # Scale thresholds to preserve sparsity (remove keepdim to match threshold shape)
        self.jumprelu.log_threshold.data = self.jumprelu.log_threshold + torch.log(W_dec_norm.squeeze(-1))
        
        return self


SAEConfig.register_for_auto_class("AutoConfig")
BatchTopKSAE.register_for_auto_class()
JumpReLUSAE.register_for_auto_class()
VanillaSAE.register_for_auto_class()
