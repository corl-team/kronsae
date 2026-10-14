from dataclasses import dataclass, field, asdict

@dataclass
class SynthConfig:

    seed: int = 42
    N: int = 256        
    S: int = 4      

    corr_intra: float = 0.8 
    corr_inter: float = 0.0 
    p_sparsity: float = 1-0.125 
    heavy_tail_scale: float = 1.0

    ae_train_steps: int = 2000 
    fe_train_steps: int = 2000
    batch_size: int = 64 
    sae_train_batch_size: int = 256 


    latent_dim: int = 16
    ae_lr: float = 1e-3


    topk: int = 4           
    fe_lr: float = 1e-3
    hidden_dim: int = 256     
    n_batches_to_dead: int = 10

    print_every: int = 100
    ckpt_dir: str = "checkpoints"
    save_checkpoint: bool = False