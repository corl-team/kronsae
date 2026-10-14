import os
import random
import math
import torch
import einops

import torch.nn as nn
import numpy as np
import pandas as pd

import torch.nn.functional as F

from torch.utils.data import IterableDataset, DataLoader
from typing import Optional
from pprint import pprint

from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *
from config import *

import numpy as np
from numpy.linalg import eigvalsh, eigh

def nearest_correlation(A, tol=1e-8, max_iter=100):
    """
    Higham's algorithm to project A onto the nearest correlation matrix.
    """
    R = A.copy()
    for _ in range(max_iter):
        # 1) Project onto PSD cone
        eigvals, eigvecs = eigh(R)
        eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
        R_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        # 2) Enforce unit diagonal
        R_new = R_psd.copy()
        np.fill_diagonal(R_new, 1.0)
        # check convergence
        if np.max(np.abs(R_new - R)) < tol:
            break
        R = R_new
    return R_new

def make_block_cov_varying(d, k, rho_range=(0.5, 0.8), eps=1e-6, random_state=None):
    """
    Construct a block-diagonal covariance matrix with varying intra-block correlations.
    """
    rng = np.random.default_rng(random_state)
    # split sizes
    base, extra = divmod(d, k)
    sizes = [base + (1 if i < extra else 0) for i in range(k)]
    
    cov = np.zeros((d, d))
    idx = 0
    for s in sizes:
        # sample random symmetric block with ones on diag
        B = rng.uniform(rho_range[0], rho_range[1], size=(s, s))
        B = np.triu(B, 1)
        B = B + B.T
        np.fill_diagonal(B, 1.0)
        # project to nearest valid correlation block
        B_corr = nearest_correlation(B)
        cov[idx:idx+s, idx:idx+s] = B_corr
        idx += s
    
    # zero inter-block, add tiny jitter
    cov += eps * np.eye(d)
    return cov
    

def create_block_covariance(N: int, S: int, corr_intra: float, corr_inter: float) -> torch.Tensor:
    """
    Create an NxN block-wise covariance matrix using torch:
      - S groups
      - correlation ~ corr_intra within each group
      - correlation ~ corr_inter across groups
    """
    assert N % S == 0, "N must be divisible by S for block partitioning"
    block_size = N // S
    Sigma = torch.zeros((N, N), dtype=torch.float32)

    for i in range(S):
        for j in range(S):
            start_i = i * block_size
            end_i = (i + 1) * block_size
            start_j = j * block_size
            end_j = (j + 1) * block_size
            if i == j:
                Sigma[start_i:end_i, start_j:end_j] = corr_intra
            else:
                Sigma[start_i:end_i, start_j:end_j] = corr_inter

    Sigma.fill_diagonal_(1.0)
    return Sigma

def bernoulli_heavytail_sample(size: int, p: float, scale: float) -> torch.Tensor:
    """
    z_i = 0 with probability p
    z_i = heavy-tail with probability (1 - p).
    We'll do a simple Pareto-ish approach plus a random sign.
    """
    mask = (torch.rand(size) > p)  # True/False
    sign = (torch.rand(size) > 0.5).float() * 2.0 - 1.0
    u = torch.rand(size)
    alpha = 0.5
    magnitude = scale * (1 - u).pow(-alpha)
    heavy_values = sign * magnitude
    return torch.where(mask, heavy_values, torch.zeros_like(heavy_values))



def nearest_correlation(A, tol=1e-8, max_iter=100):
    """
    Higham's algorithm to project A onto the nearest correlation matrix.
    """
    R = A.copy()
    for _ in range(max_iter):
        # 1) PSD projection
        eigvals, eigvecs = eigh(R)
        eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
        R_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        # 2) unit diagonal
        np.fill_diagonal(R_psd, 1.0)
        # Check convergence
        if np.max(np.abs(R_psd - R)) < tol:
            return R_psd
        R = R_psd
    return R_psd

def make_block_cov_pattern_pd(
    sizes,
    block_mask,
    rho_range=(0.3, 0.8),
    eps=1e-6,
    random_state=None
):
    """
    Build a block-structured covariance matrix and enforce positive-definiteness.
    After projection to the nearest correlation matrix, apply shrinkage to ensure PD.
    """
    rng = np.random.default_rng(random_state)
    k = len(sizes)
    assert block_mask.shape == (k, k), "block_mask must be k×k"
    d = sum(sizes)

    cov = np.zeros((d, d))
    ptr = np.cumsum([0] + sizes)

    # Assemble blocks
    for i in range(k):
        for j in range(i, k):
            si, sj = sizes[i], sizes[j]
            ii, jj = ptr[i], ptr[j]

            if i == j:
                if block_mask[i, i]:
                    # Random symmetric block
                    B = rng.uniform(rho_range[0], rho_range[1], (si, si))
                    B = np.triu(B, 1)
                    B += B.T
                    np.fill_diagonal(B, 1.0)
                else:
                    B = np.eye(si)
                cov[ii:ii+si, ii:ii+si] = B
            else:
                if block_mask[i, j]:
                    # Random off-diagonal block
                    M = rng.uniform(rho_range[0], rho_range[1], (si, sj))
                    cov[ii:ii+si, jj:jj+sj] = M
                    cov[jj:jj+sj, ii:ii+si] = M.T

    # Ensure symmetry
    cov = (cov + cov.T) / 2

    # Project onto nearest correlation matrix
    cov_pd = nearest_correlation(cov)

    # Apply shrinkage to ensure positive definiteness
    cov_pd = (1 - eps) * cov_pd + eps * np.eye(d)

    return cov_pd


class OnTheFlySynthDataset(IterableDataset):
    """
    Generates data (z in R^N) *on the fly* with:
      1) Block correlation (via L = chol(Sigma))
      2) Bernoulli heavy-tail distribution
    """
    def __init__(
        self, 
        cfg: SynthConfig, 
        on_the_fly: bool = True, 
        size: int = None, 
        batch_size= None,
        sizes: list[int] = None,
        mask: np.ndarray = None, eps = None):

        super().__init__()
        self.local_rng = torch.Generator()
        self.local_rng.manual_seed(cfg.seed)
        self.on_the_fly = on_the_fly
        self.batch_size = batch_size

        assert not (not on_the_fly and size is None)

        # Precompute the Cholesky factor
        
        if (sizes is not None) and (mask is not None)  and (eps is not None)and isinstance(cfg.corr_intra, tuple):
            print("---")
            Sigma = make_block_cov_pattern_pd(
                sizes=sizes,
                block_mask=mask,
                rho_range=cfg.corr_intra,
                random_state=cfg.seed,eps = eps)
            self.L = torch.tensor(np.linalg.cholesky(Sigma)).float()
        elif isinstance(cfg.corr_intra, tuple):
            Sigma = make_block_cov_varying(d=cfg.N, k=cfg.S, rho_range=cfg.corr_intra, random_state=cfg.seed)
            self.L = torch.tensor(np.linalg.cholesky(Sigma)).float()
        else:
            Sigma = create_block_covariance(cfg.N, cfg.S, cfg.corr_intra, cfg.corr_inter)
            eps = 1e-4
            Sigma = Sigma + torch.eye(cfg.N, dtype=torch.float32) * eps
            self.L = torch.linalg.cholesky(Sigma)
        self.cfg = cfg
        if not on_the_fly:
            self.generate(size)

    def generate(self, size):
        self.dataset = []
        print("Generate dataset ...")
        for i in tqdm(range(size)):
            z_list = []
            for _ in range(self.batch_size):
                z = bernoulli_heavytail_sample(
                    size=self.cfg.N,
                    p=self.cfg.p_sparsity,
                    scale=self.cfg.heavy_tail_scale
                )
                # Impose correlations: z @ L
                correlated = z @ self.L
                z_list.append(correlated.unsqueeze(0))
            self.dataset.append(torch.cat(z_list, dim=0))
        self.dataset = torch.vstack(self.dataset)

    def __iter__(self):
        while True:
            z_list = []
            for _ in range(self.batch_size):
                z = bernoulli_heavytail_sample(
                    size=self.cfg.N,
                    p=self.cfg.p_sparsity,
                    scale=self.cfg.heavy_tail_scale
                )
                # Impose correlations: z @ L
                correlated = z @ self.L
                z_list.append(correlated.unsqueeze(0))
            yield torch.cat(z_list, dim=0)