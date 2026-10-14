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


import comet_ml
import torch
from functools import partial
import os
import transformer_lens.utils as utils
from typing import Union

from activation_store import ActivationsStore
from sae import TrainingConfig
from sae.checkpoint import save_checkpoint_files


def init_comet(cfg):
    experiment = comet_ml.start(project_name=cfg["comet_project"])
    experiment.set_name(cfg["name"])
    experiment.log_parameters(cfg)
    return experiment

def log_comet(output, step, comet_experiment, index=None, model=None):
    metrics_to_log = ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features", "explained_variance", "topk"]
    log_dict = {k: output[k].item() if hasattr(output[k], 'item') else output[k] for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()
    if model is not None:
        log_dict['batch_topk'] = model.cfg['topk']
    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    comet_experiment.log_metrics(log_dict, step=step)

# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    activation[:, 1:] = sae_out
    return activation

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

def get_model_performance(
    model, 
    config: TrainingConfig, 
    activations_store: ActivationsStore, 
    sae, 
    index=None,
    batch_tokens=None):

    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()
        if config.batch_size > config.seq_len:
            batch_tokens = batch_tokens[:config.batch_size // config.seq_len]
    batch = activations_store.get_activations(batch_tokens)
    
    if isinstance(config.hook_point, tuple):
        batch = (batch[0].reshape(-1, config.act_size), batch[1].reshape(-1, config.act_size)) 
        sae_output = sae(batch[0], batch[1])["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1]-1, -1)
        hp = config.hook_point[1]
    else:
        batch = batch.reshape(-1, config.act_size)
        sae_output = sae(batch)["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1]-1, -1)
        hp = config.hook_point

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            original_loss = model(batch_tokens, return_type="loss").item()
            reconstr_loss = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(utils.get_act_name(hp, config.layer), partial(reconstr_hook, sae_out=sae_output))],
                return_type="loss",
            ).item()
            zero_loss = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(utils.get_act_name(hp, config.layer), zero_abl_hook)],
                return_type="loss",
            ).item()
            mean_loss = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(utils.get_act_name(hp, config.layer), mean_abl_hook)],
                return_type="loss",
            ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}
    
    return log_dict


@torch.no_grad()
def log_model_performance(
    comet_experiment,
    step, 
    model, 
    config: TrainingConfig, 
    activations_store, 
    sae, 
    index=None,
    batch_tokens=None):

    log_dict = get_model_performance(model, config, activations_store, sae, index,batch_tokens)
    
    if comet_experiment is not None:
        comet_experiment.log_metrics(log_dict, step=step)

def save_checkpoint(comet_experiment, sae, cfg: TrainingConfig, step: Union[int, str]):
    save_dir = os.path.join(
        cfg.output_dir,
        f"checkpoints_{cfg.model_name.replace('/', '_')}_{utils.get_act_name(cfg.hook_point, cfg.layer)[:10]}_{cfg.sae_name}_{step}",
    )
    os.makedirs(save_dir, exist_ok=True)

    save_checkpoint_files(sae, cfg, save_dir)

    if comet_experiment is not None:
        comet_experiment.log_model(
            name=f"{cfg.sae_name.replace('/', '-')}_{cfg.model_name.replace('/', '_')}_{utils.get_act_name(cfg.hook_point, cfg.layer)[:10]}",
            file_or_folder=save_dir,
            metadata={"step": step},
        )

    print(f"Model and config saved to {save_dir}")
