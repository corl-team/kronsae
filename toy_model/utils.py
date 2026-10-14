
# modified code from https://github.com/zroe1/toy-models-of-superposition/

import random
import torch

import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import pandas as pd

def set_all_seeds(seed: int):
    """
    Ensures reproducibility PyTorch.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_superposition_scores(w):
    w_normed = w / torch.norm(w, keepdim=True, dim=-1)
    scores = (w.T @ w_normed) ** 2
    scores.fill_diagonal_(0.)
    scores = scores.sum(-1)
    return torch.clamp(scores, max=1)


def graph_superposition(weights):
    w = weights.T.clone().detach().to('cpu')
    categories = [str(i) for i in range(w.shape[0])]
    super_scores = get_superposition_scores(w).numpy()
    colors = [(0, (i)*(136 /255), (i) * (255/255)) for i in super_scores]
    normed_weights = torch.norm(w, dim=1).numpy()
    
    plt.barh(categories, normed_weights[::-1], color=colors[::-1], height=1.0)
    plt.axis('off')
    
    plt.show()
    

def graph_weights(weights, bias):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5)) # 1 row, 2 columns
    
    w = weights.clone().cpu().detach()
    to_graph = w.T @ w
    colors = [(.4, 0, 1), (1, 1, 1), (1, .4, 0)]  # Purple -> White -> Orange
    n_bins = 100 
    cm = LinearSegmentedColormap.from_list("", colors, N=n_bins)
    
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    axs[0].imshow(to_graph, cmap=cm, norm=norm)
    # ax = plt.gca()
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    graph_biases(bias, axs[1])
    plt.subplots_adjust(left=0.0, right=1.4)
    plt.tight_layout()
    plt.show()


def graph_biases(bias, ax_obj):
  b = bias.clone().detach().cpu()
  colors = [(.4, 0, 1), (1, 1, 1), (1, .4, 0)]  # Purple -> White -> Orange
  n_bins = 100 
  cm = LinearSegmentedColormap.from_list("", colors, N=n_bins)
  
  norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
  
  ax_obj.imshow(b, cmap=cm, norm=norm)

  ax_obj.set_xticks([])
  ax_obj.set_yticks([])

import numpy as np
from scipy.linalg import sqrtm


def plot_metrics(metrics):
    sns.set_theme()
    df = pd.DataFrame(metrics)
    fig, axs = plt.subplots(1, df.shape[1], figsize = (6 * df.shape[1], 5))
    for i, col in enumerate(df.columns):
        sns.lineplot(data=df, x="step", y=col, ax = axs[i])


def rv_coefficient(C1, C2):
    num = np.trace(C1 @ C2)
    den = np.sqrt(np.trace(C1 @ C1) * np.trace(C2 @ C2))
    return num / den

def rv_permutation_pvalue(C1, C2, num_permutations=1000, seed=None):
    """
    Permutation test for RV coefficient significance.
    Returns observed RV and p-value.
    """
    rng = np.random.default_rng(seed)
    d = C1.shape[0]
    RV_obs = rv_coefficient(C1, C2)
    count = 0

    for _ in range(num_permutations):
        perm = rng.permutation(d)
        C2_perm = C2[perm][:, perm]        # permute both rows and cols
        RV_perm = rv_coefficient(C1, C2_perm)
        if RV_perm >= RV_obs:
            count += 1

    p_value = (count + 1) / (num_permutations + 1)
    return RV_obs, p_value