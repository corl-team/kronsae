import torch
from typing import Tuple
from tqdm import tqdm
from transformer_lens import HookedTransformer
import torch.nn.functional as F
from sae import BaseSAE, BatchTopKSAE, JumpReLUSAE, KronSAE, TrainingConfig, SAEConfig, VanillaSAE
from activation_store import ActivationsStore
import transformer_lens.utils as utils
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model

from sae.model import TopKSAE


def estimate_hidden_stats(
    model: HookedTransformer,
    activations_store: ActivationsStore,
    cfg: TrainingConfig,
    num_batches: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate mean and standard deviation of hidden states
    """
    print("Estimating hidden state statistics...")
    
    # Initialize running statistics on CPU for memory efficiency
    batches = []
    # Collect statistics
    for _ in tqdm(range(num_batches), desc="Collecting statistics"):
        batch = activations_store.get_batch(add_new=True).cpu()
        batches.append(batch)
    
    # Calculate mean and std on CPU
    mean = torch.stack(batches).reshape(-1, batch.size(-1)).mean(-1).mean()
    std = torch.stack(batches).reshape(-1, batch.size(-1)).std(-1).mean()
    print(f"Mean: {mean.shape}, Std: {std.shape}")
    # Move to proper device before returning
    return mean.to(cfg.device), std.to(cfg.device)


def verify_folding(
    sae_folded: BaseSAE,
    batch: torch.Tensor,
    min_explained_variance: float = 0.7
) -> bool:
    """
    Verify that folded SAE achieves good explained variance
    """
    print("Verifying folded SAE...")
    
    with torch.no_grad():
        # Verify input_unit_norm is disabled
        if sae_folded.config.input_unit_norm:
            print("Warning: input_unit_norm should be disabled after folding!")
            return False
        sae_folded.to(batch.device)
        # Forward pass with folded SAE
        output_folded = sae_folded(batch)
        
        # Check explained variance
        explained_variance = output_folded['explained_variance'].item()
        print(f"Explained variance after folding: {explained_variance:.4f}")
        variance_ok = explained_variance >= min_explained_variance
        
        if variance_ok:
            print("Verification passed: explained variance is good")
        else:
            print(f"Verification failed: explained variance {explained_variance:.4f} below threshold {min_explained_variance}")
        
        return variance_ok


def post_process_sae(
    sae: BaseSAE,
    model: HookedTransformer,
    cfg: TrainingConfig,
    num_batches: int = 100,
) -> BaseSAE:
    """
    Main function to post-process SAE by folding normalization into weights
    """
    if not sae.config.input_unit_norm:
        print("Normalization is already folded into the SAE weights.")
        return sae
    # Create activation store
    activations_store = ActivationsStore(model, cfg)
    
    # Verify original SAE performance
    test_batch = activations_store.get_batch()
    with torch.no_grad():
        sae.to(test_batch.device)
        original_output = sae(test_batch)
        original_variance = original_output['explained_variance'].item()
        print(f"Original explained variance: {original_variance:.4f}")
        if original_variance < 0.7:
            print("Warning: Original SAE has low explained variance!")
    
    # Get statistics from model
    mean = sae.input_mean
    std = sae.input_std
    print(f"Using stored statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Create a copy of the original SAE
    sae_folded = type(sae)(sae.config)
    sae_folded.load_state_dict(sae.state_dict())
    
    # Fold statistics into the encoder
    sae_folded.fold_stats_into_weights(mean, std)
    if not isinstance(sae_folded, KronSAE):
        sae_folded.fold_W_dec_norm()
    # Verify the folding
    if verify_folding(sae_folded, test_batch):
        print("Post-processing successful!")
        return sae_folded
    else:
        print("Post-processing failed!")
        assert False
        return sae


if __name__ == "__main__":
    import pyrallis
    from dataclasses import dataclass
    from typing import Optional
    from huggingface_hub import hf_hub_download
    
    @dataclass
    class PostProcessConfig:
        # HF Hub settings
        repo_id: str
        revision: Optional[str] = "final"  # Which checkpoint to load
        num_batches: int = 100  # Number of batches for statistics estimation
        
        # Output settings
        output_repo_id: Optional[str] = None  # If None, will be auto-generated
        output_revision: str = "folded"
    
    @pyrallis.wrap()
    def main(cfg: PostProcessConfig):
        print(f"Loading SAE from {cfg.repo_id}@{cfg.revision}")
        
        # Load config from hub
        sae_config = SAEConfig.from_pretrained(
            cfg.repo_id,
            revision=cfg.revision
        )
        
        # Create training config from SAE config
        train_cfg = sae_config.to_training_config()
        
        # Load the original model
        model = HookedTransformer.from_pretrained(
            train_cfg.model_name
        ).to(train_cfg.dtype).to(train_cfg.device)
        train_cfg.act_size = model.cfg.d_model
        
        # Load SAE weights from hub
        weights_path = hf_hub_download(
            repo_id=cfg.repo_id,
            filename="model.safetensors",
            revision=cfg.revision
        )
        
        # Initialize SAE with config
        sae_class = {
            "vanilla": VanillaSAE,
            "topk": TopKSAE,
            "batchtopk": BatchTopKSAE,
            "jumprelu": JumpReLUSAE,
            "kronsae": KronSAE,
        }[sae_config.sae_type]
        
        sae = sae_class(sae_config)
        
        # Load weights
        load_model(sae, weights_path)
        sae.to(train_cfg.device)
        
        # Post-process the SAE
        sae_folded = post_process_sae(
            sae=sae,
            model=model,
            cfg=train_cfg,
            num_batches=cfg.num_batches
        )
        
        # Generate output repo id if not specified
        if cfg.output_repo_id is None:
            cfg.output_repo_id = f"{cfg.repo_id}_folded"

        # Save folded model
        sae_folded.push_to_hub(
            cfg.repo_id,
            safe_serialization=True,
            revision='folded'
        )
        
        print(f"Folded model saved to {cfg.repo_id}@folded")
        print("Done!")
    
    main()
