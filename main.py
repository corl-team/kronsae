#%%
import comet_ml
import torch
import transformer_lens.utils as utils
from training import train_sae
from sae import *
from activation_store import ActivationsStore
from sae import TrainingConfig, SAEConfig
from sae.config import post_init_cfg
from loading import load_with_retries
from transformer_lens import HookedTransformer
import pyrallis
import ast
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

@pyrallis.wrap()
def main(cfg: TrainingConfig):
    if cfg.sweep_pair:
        for key, value in cfg.sweep_pair.items():
            setattr(cfg, key, value)

    comet_experiment = None
    if cfg.enable_comet:
        comet_experiment = comet_ml.start(project_name=cfg.comet_project)
        comet_experiment.set_name(
            f"{cfg.dict_size}_{cfg.sae_type}_{cfg.comet_run_suffix}"
        )
        
    model = load_with_retries(
        lambda: HookedTransformer.from_pretrained_no_processing(
            cfg.model_name, dtype=torch.bfloat16
        ),
        "loading transformer model",
    ).to(cfg.device)

    cfg.act_size = model.cfg.d_model
    
    print(f"act_size: {cfg.act_size}")
    if "," in cfg.hook_point:
        cfg.hook_point = ast.literal_eval(cfg.hook_point)
        if  isinstance(cfg.hook_point , list):
            cfg.hook_point = tuple(cfg.hook_point)
    activations_store = ActivationsStore(model, cfg)
    cfg = post_init_cfg(cfg, activations_store)
    
    model_config = SAEConfig.from_training_config(cfg)
    if cfg.sae_type == "vanilla":
        sae = VanillaSAE(model_config)
    elif cfg.sae_type == "topk":
        sae = TopKSAE(model_config)
    elif cfg.sae_type == "batchtopk":
        sae = BatchTopKSAE(model_config)
    elif cfg.sae_type == 'jumprelu':
        sae = JumpReLUSAE(model_config)
    elif cfg.sae_type == "kronsae":
        model_config.cartesian_op = "mul"
        sae = KronSAE(model_config)
    else:
        raise ValueError(f"SAE type {cfg.sae_type} is not understood.")
    
    train_sae(
        sae,
        activations_store,
        model,
        cfg,
        train_transcoder=isinstance(cfg.hook_point, tuple),
        comet_experiment=comet_experiment,
    )


if __name__ == "__main__":
    main()
