from typing import Optional, Union
import torch
from tqdm.auto import tqdm
from sae import TrainingConfig
from logs import get_model_performance, log_model_performance, save_checkpoint

from sae import (
    VanillaSAE, 
    TopKSAE, 
    BatchTopKSAE, 
    JumpReLUSAE, 
    KronSAE
    )

from sae import count_parameters, set_seed

from activation_store import ActivationsStore
from transformers import get_scheduler

from transformer_lens import HookedTransformer
import os
from dataclasses import asdict
from sae.checkpoint import save_checkpoint_files

from sae.train_utils import configure_optimizers, count_kronsae_flops, count_topk_flops


def save_sae(sae, cfg: TrainingConfig, step: Union[int, str] = 'final'):
    os.makedirs(cfg.output_dir, exist_ok=True)
    base_path = os.path.join(
        cfg.output_dir,
        f"sae_{cfg.model_name.replace('/', '_')}_{cfg.layer}_{cfg.dict_size}_{cfg.sae_type}_{cfg.comet_run_suffix}",
    )
    
    save_checkpoint_files(sae, cfg, os.path.join(base_path, str(step)))


def train_sae(
    sae: VanillaSAE | TopKSAE | BatchTopKSAE | JumpReLUSAE,
    activations_store: ActivationsStore,
    model: HookedTransformer,
    cfg: TrainingConfig,
    train_transcoder: bool = False,
    comet_experiment=None,
) -> None:
    """
    Train a sparse autoencoder on the activations of a transformer model
    """

    set_seed(cfg.seed)

    #optim_groups = configure_optimizers(sae, cfg)
    optimizer = torch.optim.Adam(
        params=sae.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.99),
    )
    
    # Calculate FLOPS
    print("Start SAE FLOPS:")
    if isinstance(sae, (KronSAE)):
        print("Count theoretical Flops")
        forward_flops = count_kronsae_flops(cfg.act_size, cfg.topk, cfg.num_mkeys, cfg.num_nkeys, cfg.num_heads)
    elif isinstance(sae, (TopKSAE, BatchTopKSAE)):
        print("Count theoretical Flops")
        forward_flops = count_topk_flops(cfg.act_size, cfg.dict_size, cfg.topk)
    else:
        raise NotImplementedError

    accum_num_flops = 0

    count_parameters(sae)
    total_params = sum(p.numel() for p in sae.parameters())
    if comet_experiment is not None:
        config_parameters = {
            key: str(value) if isinstance(value, torch.dtype) else value
            for key, value in asdict(cfg).items()
        }
        comet_experiment.log_parameter("parameters_count", total_params)
        comet_experiment.log_parameters(config_parameters)

    # Calculate total number of batches
    tokens_per_batch = cfg.batch_size
    total_batches = cfg.num_tokens // tokens_per_batch
    warmup_topk_batches = int(total_batches * cfg.topk_warmup_steps_fraction)
    if cfg.warmup_fraction <= 1.0:
        warmup_lr_batches = int(total_batches * cfg.warmup_fraction)
    else:
        warmup_lr_batches = int(cfg.batch_size * cfg.warmup_fraction)


    # scheduler = get_scheduler(
    #     name=cfg.scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup_lr_batches,
    #     num_training_steps=total_batches
    # )
    # scheduler_type: str = 'cosine_with_min_lr'

    scheduler = get_scheduler(
        name=cfg.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_lr_batches,
        num_training_steps=total_batches,
        scheduler_specific_kwargs=dict(min_lr=1e-6),
    )

    print(f"tokens per iteration will be: {tokens_per_batch:,}")
    
    # Initialize CUDA events for timing if available
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    else:
        start_event = end_event = None
    
    # Training loop
    pbar = tqdm(range(total_batches), desc="Training SAE")
    assert cfg.batch_size % cfg.model_batch_size == 0, "Batch size must be a multiple of model batch size"
    num_batches_before_refresh = (cfg.seq_len * cfg.model_batch_size) // cfg.batch_size
    is_topk_variant = isinstance(sae, (TopKSAE, BatchTopKSAE, KronSAE))
    sae = torch.compile(sae)
    sae.to(cfg.device)
    
    n_tokens_processed_true = 0
    
    saving_steps = [100, 1000, 5000, 6000, 10000, 20000]
    for i in pbar:
        # Anneal topk during warmup only for TopK variants
        if is_topk_variant and i < warmup_topk_batches:
            current_topk = int(cfg.start_topk - (cfg.start_topk - cfg.topk) * (i / warmup_topk_batches))
            sae.config.topk = current_topk
        else:
            sae.config.topk = cfg.topk
        
        add_new = (i+1) % num_batches_before_refresh == 0
        batch = activations_store.get_batch(add_new=add_new)
        
        # Start timing forward pass
        if start_event is not None:
            start_event.record()
        if train_transcoder:
            loss_dict = sae(batch[0].to(cfg.sae_dtype), batch[1].to(cfg.sae_dtype))
        else:
            loss_dict = sae(batch.to(cfg.sae_dtype))
        
        # End timing forward pass
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            forward_time_ms = start_event.elapsed_time(end_event)
        else:
            forward_time_ms = 0
        
        # Start timing backward pass
        if start_event is not None:
            start_event.record()
            
        # Optimizer step
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=cfg.max_grad_norm)
        sae.make_decoder_weights_and_grad_unit_norm()
            
        # End timing backward pass
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            backward_time_ms = start_event.elapsed_time(end_event)
        else:
            backward_time_ms = 0
            
        optimizer.step()
        scheduler.step()

        del loss_dict["sae_out"]
        del loss_dict["feature_acts"]
        forward_flops_current = forward_flops
        accum_num_flops += forward_flops_current 

        # Log norm
        total_norm = 0.
        for p in sae.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Log metrics
        n_tokens_processed = (i + 1) * tokens_per_batch
        n_tokens_processed_true += batch.shape[0]
        metrics_to_log = {
            "n_tokens": n_tokens_processed,
            "n_tokens_true": n_tokens_processed_true,
            "forward_time_ms": forward_time_ms,
            "backward_time_ms": backward_time_ms,
            "forward_flops": forward_flops_current,
            "accum_num_flops": accum_num_flops,
            "grad_norm": total_norm,
            **loss_dict,
        }
        if is_topk_variant:
            metrics_to_log["current_topk"] = sae.config.topk
        
        if comet_experiment is not None:
            comet_experiment.log_metrics(
                {key: value.item() if hasattr(value, "item") else value for key, value in metrics_to_log.items()},
                step=i + 1,
            )
        
        # Update progress bar
        progress_metrics = {
            "l2_loss": f"{loss_dict['l2_loss']:.3f}",
            "explained_var": f"{loss_dict['explained_variance']:.3f}",
            "forward_ms": f"{forward_time_ms}",
            "backward_ms": f"{backward_time_ms}",
        }
        if is_topk_variant:
            progress_metrics["topk"] = sae.config.topk
            
        pbar.set_postfix(progress_metrics)
        if (i+1) % cfg.performance_log_steps == 0:
            log_model_performance(
                comet_experiment=comet_experiment,
                step=i+1,
                model=model,
                config=cfg,
                activations_store=activations_store,
                sae=sae,
            )
        if (i+1) % cfg.save_checkpoint_steps == 0:
            save_checkpoint(
                comet_experiment=comet_experiment,
                sae=sae,
                cfg=cfg,
                step=i+1)
        
        # Update progress bar
        pbar.set_postfix({
            "l2_loss": f"{loss_dict['l2_loss']:.3f}",
            "explained_var": f"{loss_dict['explained_variance']:.3f}",
            "topk": sae.config.topk,
        })
    
    sae = sae.fold_stats_into_weights()
    num_eval_iters = cfg.eval_num_batches
    log_dict_total = {
        "performance/ce_degradation": 0.,
        "performance/recovery_from_zero": 0.,
        "performance/recovery_from_mean": 0.,
    }
    for i in tqdm(range(num_eval_iters)):
        log_dict = get_model_performance(
            model=model,
            config=cfg,
            activations_store=activations_store,
            sae=sae,
        )

        log_dict_total["performance/ce_degradation"] += log_dict["performance/ce_degradation"]
        log_dict_total["performance/recovery_from_zero"] += log_dict["performance/recovery_from_zero"]
        log_dict_total["performance/recovery_from_mean"] += log_dict["performance/recovery_from_mean"]
    
    log_dict_total["performance/ce_degradation"] /= num_eval_iters
    log_dict_total["performance/recovery_from_zero"] /= num_eval_iters
    log_dict_total["performance/recovery_from_mean"] /= num_eval_iters

    if comet_experiment is not None:
        comet_experiment.log_metrics({
            "eval/performance/ce_degradation": log_dict_total["performance/ce_degradation"],
            "eval/performance/recovery_from_zero": log_dict_total["performance/recovery_from_zero"],
            "eval/performance/recovery_from_mean": log_dict_total["performance/recovery_from_mean"],
        }, step=total_batches)

    save_checkpoint(
        comet_experiment=comet_experiment,
        sae=sae,
        cfg=cfg,
        step="final"
        )

    pbar.close()
    if comet_experiment is not None:
        comet_experiment.end()


def train_sae_group(
    cfg: TrainingConfig,
    sae_types: list[str],
    topk_values: Optional[list[int]] = None,
    dict_sizes: Optional[list[int]] = None
) -> None:
    """
    Train multiple SAEs with different configurations
    """
    base_cfg = cfg
    
    for sae_type in sae_types:
        if topk_values:
            for topk in topk_values:
                cfg = TrainingConfig(**vars(base_cfg))  # Create new config instance
                cfg.sae_type = sae_type
                cfg.topk = topk
                
                model = HookedTransformer.from_pretrained(cfg.model_name).to(cfg.dtype).to(cfg.device)
                cfg.act_size = model.cfg.d_model
                activations_store = ActivationsStore(model, cfg)
                
                sae = {
                    "vanilla": VanillaSAE,
                    "topk": TopKSAE,
                    "batchtopk": BatchTopKSAE,
                    "jumprelu": JumpReLUSAE,
                }[sae_type](cfg)
                
                train_sae(sae, activations_store, model, cfg)
        
        if dict_sizes:
            for dict_size in dict_sizes:
                cfg = TrainingConfig(**vars(base_cfg))  # Create new config instance
                cfg.sae_type = sae_type
                cfg.dict_size = dict_size
                
                model = HookedTransformer.from_pretrained(cfg.model_name).to(cfg.dtype).to(cfg.device)
                cfg.act_size = model.cfg.d_model
                activations_store = ActivationsStore(model, cfg)
                
                sae = {
                    "vanilla": VanillaSAE,
                    "topk": TopKSAE,
                    "batchtopk": BatchTopKSAE,
                    "jumprelu": JumpReLUSAE,
                }[sae_type](cfg)
                
                train_sae(sae, activations_store, model, cfg)
