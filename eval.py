#MIT License

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

import torch
from functools import partial
import os
import json
import transformer_lens.utils as utils
from dataclasses import asdict

from sae import TrainingConfig, SAEConfig
from tqdm.auto import tqdm

# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    activation[:, 1:] = sae_out
    return activation

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

@torch.no_grad()
def eval_model_performance(model, config: TrainingConfig, activations_store, sae, index=None, batch_tokens=None):
    num_batches_before_refresh = (config.seq_len * config.model_batch_size) // config.batch_size

    tokens_per_batch = config.model_batch_size * config.seq_len
    total_batches = config.num_tokens // tokens_per_batch
    
    # Check if we're using a TopK variant
    pbar = tqdm(range(total_batches), desc="Training SAE")
    for i in pbar:
        add_new = (i+1) % num_batches_before_refresh == 0
        batch = activations_store.get_batch(add_new=add_new)

        if batch_tokens is None:
            batch_tokens = activations_store.get_batch_tokens()
            if config.batch_size > config.seq_len:
                batch_tokens = batch_tokens[:config.batch_size // config.seq_len]
        batch = activations_store.get_activations(batch_tokens).reshape(-1, config.act_size)
        sae_output = sae(batch)["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1]-1, -1)

        original_loss = model(batch_tokens, return_type="loss").item()
        reconstr_loss = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(utils.get_act_name(config.hook_point, config.layer), partial(reconstr_hook, sae_out=sae_output))],
            return_type="loss",
        ).item()
        zero_loss = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(utils.get_act_name(config.hook_point, config.layer), zero_abl_hook)],
            return_type="loss",
        ).item()
        mean_loss = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(utils.get_act_name(config.hook_point, config.layer), mean_abl_hook)],
            return_type="loss",
        ).item()

        ce_degradation = original_loss - reconstr_loss
        zero_degradation = original_loss - zero_loss
        mean_degradation = original_loss - mean_loss

        per_token_l2_loss_A = (sae_output.float() - batch.float()).pow(2).sum(-1).squeeze()
        total_variance_A = (batch.float() - batch.float().mean(0)).pow(2).sum(-1).squeeze()
        explained_variance = (1 - per_token_l2_loss_A / total_variance_A).mean()

        log_dict = {
            "performance/ce_degradation": ce_degradation,
            "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
            "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
            "explained_variance": explained_variance
        }

        if index is not None:
            log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}
