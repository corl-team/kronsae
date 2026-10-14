from dataclasses import dataclass, field
from typing import Optional, Union
import torch
import pyrallis
from transformers import PretrainedConfig


@dataclass(kw_only=True)
class SAEParameters:
    # SAE settings
    act_size: Optional[int] = None  # Will be set after model initialization
    sae_type: str = "topk"
    dict_size: int = 2**15
    aux_penalty: float = 1/32
    input_unit_norm: bool = True
    batch_norm_on_queries: bool = False
    affine_batch_norm: bool = False
    linear_heads: int = 0
    
    # TopK specific settings
    topk: int = 50
    topk_warmup_steps_fraction: float = 0.1
    start_topk: int = 4096
    topk_aux: int = 512

    # KronSAE specific settings
    cartesian_op: str = "mul"
    router_depth: int = 2
    router_tree_width: int = None
    # num_mkeys - dimensions of base composition 
    num_mkeys: int = None
    # num_nkeys - dimensions of extension composition 
    num_nkeys: int = None

    num_heads: int = None

    n_batches_to_dead: int = 10

    # SAE optimization settings
    bandwidth: float = 0.001
    l1_coeff: float = 0.0018
    sae_dtype: torch.dtype = field(default=torch.float32)


@dataclass(kw_only=True)
class TrainingConfig(SAEParameters):
    # Model settings
    model_name: str
    layer: int
    hook_point: Union[str, list[str, str]] = "resid_post"

    # Training settings
    lr: float = 3e-4
    num_tokens: int = int(1e9)
    seq_len: int = 1024
    model_batch_size: int = 16
    num_batches_in_buffer: int = 5
    max_grad_norm: float = 1.0
    batch_size: int = 8192
    weight_decay: float = 0.00

    # scheduler
    warmup_fraction: float = 0.1
    scheduler_type: str = 'cosine_with_min_lr' #or 'linear'
    
    # Hardware settings
    device: str = "cuda"
    dtype: torch.dtype = field(default=torch.float32)
    
    # Dataset settings
    dataset_path: str = "HuggingFaceFW/fineweb-edu"
    
    # Logging settings
    comet_project: str = "..."
    enable_comet: bool = False
    sae_name: str = "sae"

    seed: int = None

    performance_log_steps: int = 100
    save_checkpoint_steps: int = 1_000_000
    eval_num_batches: int = 150
    output_dir: str = "checkpoints"
    comet_run_suffix: str = ""
    
    sweep_pair: dict = None

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Convert string dtypes to torch.dtype if needed
        for name in ("dtype", "sae_dtype"):
            value = getattr(self, name)
            if isinstance(value, str):
                setattr(self, name, getattr(torch, value))


class SAEConfig(PretrainedConfig):
    model_type = "sae"
    
    def __init__(
        self,
        # SAE architecture
        act_size: int = None,
        dict_size: int = 2**15,
        sae_type: str = "batchtopk",
        input_unit_norm: bool = True,
        batch_norm_on_queries: bool = False,
        affine_batch_norm: bool = False,
        router_depth: int = 2,
        router_tree_width: int = None,
        cartesian_op: str = "mul",
        linear_heads: int = None,

        # Other settings
        num_heads: int = None,
        num_mkeys: int = None,
        num_nkeys: int = None,

        # TopK specific settings
        topk: int = 50,
        topk_aux: int = 512,
        n_batches_to_dead: int = 10,
        
        # Training hyperparameters
        aux_penalty: float = 1/32,
        l1_coeff: float = 0.0018,
        bandwidth: float = 0.001,
        
        # Hardware settings
        dtype: str = "float32",
        sae_dtype: str = "float32",
        
        # Optional parent model info
        parent_model_name: Optional[str] = None,
        parent_layer: Optional[int] = None,
        parent_hook_point: Optional[str] = None,
        
        # Input normalization settings
        input_mean: Optional[float] = None,
        input_std: Optional[float] = None,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        self.act_size = act_size
        self.dict_size = dict_size
        self.sae_type = sae_type
        self.input_unit_norm = input_unit_norm
        self.batch_norm_on_queries = batch_norm_on_queries
        self.affine_batch_norm = affine_batch_norm

        self.router_depth = router_depth
        self.router_tree_width = router_tree_width
        self.cartesian_op = cartesian_op
        self.linear_heads = linear_heads

        self.num_heads = num_heads
        self.num_mkeys: int = num_mkeys
        self.num_nkeys: int = num_nkeys
        
        self.topk = topk
        self.topk_aux = topk_aux
        self.n_batches_to_dead = n_batches_to_dead
        
        self.aux_penalty = aux_penalty
        self.l1_coeff = l1_coeff
        self.bandwidth = bandwidth
        
        self.dtype = dtype
        self.sae_dtype = sae_dtype
        
        self.parent_model_name = parent_model_name
        self.parent_layer = parent_layer
        self.parent_hook_point = parent_hook_point
        
        self.input_mean = input_mean
        self.input_std = input_std
    
    def get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float32)
    
    @classmethod
    def from_training_config(cls, cfg: TrainingConfig):
        """Convert TrainingConfig to SAEConfig"""
        return cls(
            act_size=cfg.act_size,
            dict_size=cfg.dict_size,
            sae_type=cfg.sae_type,
            input_unit_norm=cfg.input_unit_norm,
            affine_batch_norm=cfg.affine_batch_norm,
            topk=cfg.topk,
            topk_aux=cfg.topk_aux,
            router_depth = cfg.router_depth,
            batch_norm_on_queries = cfg.batch_norm_on_queries,
            linear_heads = cfg.linear_heads,
            router_tree_width = cfg.router_tree_width,
            cartesian_op = cfg.cartesian_op,
            num_heads = cfg.num_heads,
            num_mkeys = cfg.num_mkeys,
            num_nkeys = cfg.num_nkeys,
            n_batches_to_dead=cfg.n_batches_to_dead,
            aux_penalty=cfg.aux_penalty,
            l1_coeff=cfg.l1_coeff,
            bandwidth=cfg.bandwidth,
            dtype=str(cfg.dtype).split('.')[-1],
            sae_dtype=str(cfg.sae_dtype).split('.')[-1],
            parent_model_name=cfg.model_name,
            parent_layer=cfg.layer,
            parent_hook_point=cfg.hook_point,
            input_mean=cfg.input_mean if hasattr(cfg, 'input_mean') else None,
            input_std=cfg.input_std if hasattr(cfg, 'input_std') else None,
        )
    
    def to_training_config(self) -> TrainingConfig:
        """Convert SAEConfig back to TrainingConfig"""
        return TrainingConfig(
            dtype=self.get_torch_dtype(self.dtype),
            sae_dtype=self.get_torch_dtype(self.sae_dtype),
            model_name=self.parent_model_name,
            layer=self.parent_layer,
            hook_point=self.parent_hook_point,
            act_size=self.act_size,
            dict_size=self.dict_size,
            sae_type=self.sae_type,
            input_unit_norm=self.input_unit_norm,
            batch_norm_on_queries=self.batch_norm_on_queries,
            affine_batch_norm=self.affine_batch_norm,
            linear_heads=self.linear_heads,
            topk=self.topk,
            topk_aux=self.topk_aux,
            router_depth=self.router_depth,
            router_tree_width=self.router_tree_width,
            cartesian_op=self.cartesian_op,
            num_heads=self.num_heads,
            num_mkeys=self.num_mkeys,
            num_nkeys=self.num_nkeys,
            n_batches_to_dead=self.n_batches_to_dead,
            aux_penalty=self.aux_penalty,
            l1_coeff=self.l1_coeff,
            bandwidth=self.bandwidth,
        )


@pyrallis.wrap()
def get_config() -> TrainingConfig:
    return TrainingConfig()


# For backward compatibility
def get_default_cfg() -> TrainingConfig:
    return get_config()


def post_init_cfg(cfg: TrainingConfig, activation_store = None) -> TrainingConfig:
    """
    Any additional configuration setup that needs to happen after model initialization
    Args:
        cfg: Training configuration
        activation_store: Optional activation store to get input statistics
    """
    if activation_store is not None:
        cfg.input_mean = activation_store.mean
        cfg.input_std = activation_store.std
        print(f"Setting input statistics from activation store - Mean: {cfg.input_mean:.4f}, Std: {cfg.input_std:.4f}")
    
    return cfg
