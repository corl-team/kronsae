from prettytable import PrettyTable
import random
import numpy as np
import torch
from .config import TrainingConfig
from .model import BaseSAE


def count_topk_flops(d, features, k):
    return d * features + d * k

def count_kronsae_flops(d, k, m, n, heads):
    return d * heads * (m + n) + d * k

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def configure_optimizers(model: BaseSAE, train_config: TrainingConfig):
    # from https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L254
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()

    whitelist_weight_modules = (torch.nn.Linear)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    hc_params_static = ['static_beta',
                        'static_alpha']
    hc_params_dynamic = ['dynamic_alpha_fn',
                         'dynamic_alpha_scale',
                         'dynamic_beta_fn',
                         'dynamic_beta_scale']
    
    mask_params = ['mask_param']
    params_no_decay = mask_params
    params_decay = ['static_weights']

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith("bias") or pn in hc_params_static or pn in params_no_decay:
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules) or pn in params_decay:
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn in hc_params_dynamic:
                decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay/repeat_classifier set!" % (
        str(param_dict.keys() - union_params),
    )

    print('=' * 50)
    print("DECAY")
    print(sorted(list(decay)))
    print('=' * 50)
    print("NO DECAY")
    print(sorted(list(no_decay)))

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict and pn not in mask_params],
            "weight_decay": train_config.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict and pn not in mask_params],
            "weight_decay": 0.0,
        }
    ]
    return optim_groups


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed)
