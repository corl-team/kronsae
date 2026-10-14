# KronSAE

### Kronecker Factorization Improves Efficiency and Interpretability of Sparse Autoencoders

Vadim Kurochkin, Yaroslav Aksenov, Daniil Laptev, Daniil Gavrilov, Nikita Balagansky

*T-Tech · EMNLP 2026*

KronSAE factorizes the latent space of a sparse autoencoder into heads and
builds features from pairs of lower-dimensional pre-latents. This gives the
encoder a compositional structure while reducing its parameter count and
compute.

This repository contains the training code, paper configurations, and
experiment and plotting notebooks used for the experiments.

## Requirements

- Python 3.11
- Linux and a CUDA GPU for full training runs
- Enough disk space for model weights, fineweb-edu, and checkpoints
- A Hugging Face account with Gemma-2 access for the Gemma experiments

Comet is optional and disabled in the example configs.

## Repository structure

```text
.
├── configs/paper/       # Qwen and Gemma training configs
├── sae/                 # SAE and KronSAE implementations
├── notebooks/           # All experiment and plotting notebooks
├── toy_model/           # Toy-model implementation and data utilities
├── main.py              # Single-GPU training entry point
├── training.py          # Training loop
├── activation_store.py  # Activation collection and buffering
├── post_training.py     # Post-training evaluation utilities
├── eval.py              # Reconstruction evaluation
├── logs.py              # Metrics and checkpoints
├── requirements.txt     # Training and notebook dependencies
├── CITATION.bib         # BibTeX citation
├── LICENSE              # Apache License 2.0
└── NOTICE               # Copyright and third-party attribution
```

## Setup

```bash
git clone https://github.com/corl-team/kronsae.git
cd kronsae
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training

To train KronSAE on Qwen, run:

```bash
python main.py --config_path configs/paper/qwen_kronsae.yaml
```

The other example configurations are:

- `configs/paper/qwen_topk.yaml`
- `configs/paper/gemma_kronsae.yaml`
- `configs/paper/gemma_topk.yaml`

These configs use fineweb-edu activations and the main settings from the paper:
layer 12, 16,384 features, sparsity 32, 250M training tokens, global batch size
8,192, learning rate `8e-4`, 10% linear warmup, and cosine decay to `1e-6`.

Gemma-2 is gated on Hugging Face. Accept its license and log in with your own
Hugging Face account before running the Gemma configs.

Checkpoints are written to the `output_dir` set in each config. Each checkpoint
contains `model.safetensors`, the SAE's `config.json`, and the original
`training_config.json`. Compiled models are unwrapped before saving. Load the
weights with `safetensors.torch.load_model` and the configuration with
`SAEConfig.from_pretrained(checkpoint_dir)`.

Model and dataset loading retry transient network errors up to three times,
with a ten-second pause between attempts. Authentication, configuration, and
memory errors are raised immediately.

To enable Comet, set `enable_comet: true` and add your own `comet_project` name.

## Paper sweeps

For the paper sweeps, vary:

- dictionary size `F`: `16384`, `32768`, `65536`;
- sparsity `topk`: `8`, `16`, `32`, `64`;
- layer: `6`, `12`, `20`;
- KronSAE factor `m`: `1`, `2`, `4`, `8`.

For KronSAE, choose `h`, `m`, and `n` so that
`dict_size = num_heads * num_mkeys * num_nkeys`. The example configs use
`h=1024`, `m=4`, and `n=4`.

## Notebooks

All notebooks are in [`notebooks/`](notebooks/). Start Jupyter from the repository
root after installing `requirements.txt`:

```bash
jupyter lab
```

`toy_logic_analysis.ipynb` visualizes toy logic operators. `toy_model.ipynb` runs synthetic
experiments with the utilities in `toy_model/` and saves checkpoints there.
Notebooks that import repository modules initialize their working directory and
import paths. Other relative data paths resolve from the repository root.

Plotting notebooks require reproduced experiment tables and evaluation outputs.
For `ablations.ipynb`, load your experiment table into a pandas DataFrame named
`df`. `feature_correlations.ipynb` compares TopKSAE and KronSAE feature
correlations and plots the resulting tables.

## Citation

If you use this code, please cite:

```bibtex
@misc{kurochkin2026kronsae,
  title  = {Kronecker Factorization Improves Efficiency and Interpretability of Sparse Autoencoders},
  author = {Kurochkin, Vadim and Aksenov, Yaroslav and Laptev, Daniil and Gavrilov, Daniil and Balagansky, Nikita},
  year   = {2026}
}
```

The same BibTeX entry is available in [`CITATION.bib`](CITATION.bib).

## License

KronSAE is released under the [Apache License 2.0](LICENSE). Third-party
portions retain their original MIT notices; see [NOTICE](NOTICE) and the source
headers.
