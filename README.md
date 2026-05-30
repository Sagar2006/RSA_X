# RSA-X (Reinforced Sparse Attention eXperimental Framework)

RSA-X is a research-grade experimental framework designed to analyze and profile the natural attention sparsity in pretrained transformers. This implementation represents **Phase 1** of the RSA-X roadmap, establishing the core analytical machinery, data pipelines, extraction engines, and publication-quality figure generators targeting **GPT-2 Small** evaluated on **WikiText-103**.

The core objective is to empirical analyze whether natural sparsity is concentrated in specific layers/heads, providing evidence for whether transformer attention can be safely pruned or optimized from $O(n^2)$ to $O(nk)$ without quality loss.

---

## 1. Scientific Methodology & Math Formulations

This framework computes the following key scientific metrics over the attention probability tensors $A \in \mathbb{R}^{B \times L_{layer} \times L_{head} \times N \times N}$ extracted from GPT-2 Small, where $N$ is the sequence length (default 256):

### 1.1 Shannon Attention Entropy ($H$)
Attention entropy measures the concentration/dispersion of the attention probability distribution. For a query token position $i$ and head $h$, the entropy over keys $j$ is computed as:
$$H(i) = -\sum_{j=1}^{N} p_{ij} \log p_{ij}$$
where $p_{ij}$ is the attention weight from query $i$ to key $j$, such that $\sum_{j} p_{ij} = 1.0$.
- **Interpretation:** High entropy ($\approx \log N$) implies uniform/dispersed attention. Low entropy ($\approx 0.0$) implies highly localized, deterministic attention (e.g. attending strictly to the previous token, the first token, or period punctuation).

### 1.2 Top-K Cumulative Mass ($M_K$)
Top-K mass measures what percentage of the total attention budget is concentrated in the top $K$ key positions:
$$M_K(i) = \sum_{j \in \mathcal{K}_i} p_{ij}$$
where $\mathcal{K}_i$ is the set of $K$ indices with the largest attention weights $p_{ij}$ for query $i$.
- **Interpretation:** If $M_5 \approx 0.85$, it means five key tokens absorb 85% of the attention probability, indicating extreme natural sparsity.

### 1.3 Attention Density ($D$)
Attention density measures the fraction of key positions that receive an attention budget greater than a uniform baseline distribution threshold ($1/N$):
$$D(i) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{I}\left(p_{ij} > \frac{1}{N}\right)$$
where $\mathbb{I}$ is the indicator function.

### 1.4 Sparsity Percentage ($S$)
The overall sparsity is defined as the percentage of attention weights that fall below a near-zero threshold (default $\epsilon = 10^{-4}$):
$$S(i) = \frac{100}{N} \sum_{j=1}^{N} \mathbb{I}\left(p_{ij} \le 10^{-4}\right)$$

### 1.5 Power-Law Exponent ($\alpha$)
To test if attention weight distributions follow a heavy-tailed power law (Zipf-like distribution), we fit a linear regression model to the log-log ranked weights:
$$\log(\text{weight}_r) = -\alpha \log(r) + C$$
where $r \in \{1, 2, \dots, N\}$ is the descending rank of the attention weight. The tail decay exponent $\alpha$ and coefficient of determination $R^2$ are extracted via least-squares.

---

## 2. Project Architecture

The codebase follows the strict layout required by the research guidelines:

```
rsa_x/
│
├── configs/
│   └── default_config.yaml      # Centralized YAML hyperparameter config
│
├── datasets/
│   ├── __init__.py
│   └── loader.py                # HF WikiText-103 text token pack pipeline
│
├── models/
│   ├── __init__.py
│   └── extraction.py            # Hooked attention extraction using TransformerLens
│
├── analysis/
│   ├── __init__.py
│   ├── entropy.py               # Numerically-stable attention entropy analyzer
│   ├── sparsity.py              # Top-k and sparsity percentiles analyzer
│   └── statistics.py            # Summary tables generator & power-law fitter
│
├── visualization/
│   ├── __init__.py
│   └── plots.py                 # Matplotlib 300 DPI publication figure generator
│
├── experiments/
│   ├── __init__.py
│   ├── tracker.py               # W&B logger with offline/local CSV backups
│   └── runner.py                # Main orchestrator running Experiments 1, 2 & 3
│
├── results/                     # Directory for all generated outputs
│   ├── metrics/                 # Serialized summary CSVs and consolidated NPZ arrays
│   ├── raw_tensors/             # Compressed .npy and sparse .parquet sample matrices
│   └── figures/                 # 300 DPI paper-ready PNG figures
│
├── scripts/
│   └── generate_figures.py      # Independent figure regenerator script
│
├── tests/
│   └── test_framework.py        # PyTest suite verifying mathematical bounds
│
├── requirements.txt             # Pinned library dependencies
├── main.py                      # Master command-line interface entry point
└── README.md                    # Core documentation
```

---

## 3. Setup and Installation

### 3.1 Prerequisites
- Python 3.12+ (tested up to 3.14)
- PyTorch 2.0+ (handles both CPU-only and CUDA systems)

### 3.2 Setup Virtual Environment
Initialize a clean Python virtual environment and install the pinned dependencies:

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux/MacOS

# Install pinned requirements
pip install -r requirements.txt
```

---

## 4. Execution and Replication Guide

### 4.1 Run the Full Experimental Suite
Execute `main.py` to run the dataset loading, model tokenization, attention extraction, entropy calculations, power-law fitting, table aggregations, and automatic figure generation:

```bash
python main.py
```

#### Key Command-line Overrides:
You can customize the run without editing the YAML config directly:
- `--num_samples N`: Total sequences to evaluate (default 50 for quick CPU testing, increase to 100+ for dataset-wide publication papers).
- `--batch_size B`: Adjust evaluation batch size (default 4).
- `--device {cpu,cuda,auto}`: Force target device (default auto).
- `--wandb_mode {online,offline}`: Toggle tracking uploads (default offline).
- `--save_raw_samples S`: Control how many samples get full JSON/NPY/Parquet extraction dumps (default 5).

Example:
```bash
python main.py --num_samples 100 --batch_size 8 --device cuda
```

### 4.2 Run Independent Figure Regeneration
If you want to tweak plot sizes, color schemes, or styles, you do not need to re-run the heavy neural network. Simply run the figure regenerator utility, which loads the saved serialized metrics:

```bash
python scripts/generate_figures.py
```

### 4.3 Run Mathematical Verification Suite
To verify the math constraints and hook implementations, execute the pytest suite:

```bash
pytest tests/test_framework.py
```

---

## 5. Generated Visualizations Overview

The pipeline automatically outputs **seven publication-ready, 300 DPI charts** in the `results/figures/` folder:

1. **Figure 1 (`fig1_attention_heatmap_...`)**: Visualizes a 2D heat matrix of token-to-token attention weights for a specific layer and head. Includes word tokens on both axes for legibility.
2. **Figure 2 (`fig2_entropy_histogram`)**: Distribution of Shannon attention entropy across all processed tokens, with reference lines marking the dataset mean and theoretical maximum (uniform distribution).
3. **Figure 3 (`fig3_sparsity_histogram`)**: Sparsity ratio histogram showing the fraction of key values dropping below $\epsilon = 10^{-4}$.
4. **Figure 4 (`fig4_top_k_curve`)**: Concentration curves showing cumulative attention mass inside Top-1, Top-5, Top-10, and Top-50 key positions with standard error bands.
5. **Figure 5 (`fig5_layerwise_comparison`)**: Multi-panel boxplot illustrating how head entropy and sparsity behave across the 12 transformer layers, revealing depth trends.
6. **Figure 6 (`fig6_headwise_comparison`)**: Grid heatmaps of Layer (y-axis) vs. Head (x-axis) showing average entropy and sparsity, helping locate localized attention specialists.
7. **Figure 7 (`fig7_attention_density`)**: Density distribution showing the active routing factor over key tokens.

---

## 6. Weights & Biases Dashboard

By default, the framework runs in **offline mode**, caching all training runs locally in `results/metrics/local_metrics_log.csv` and W&B files. 

To sync the run with your live Weights & Biases dashboard:
1. Log in to your W&B account in the terminal: `wandb login`
2. Run the experiments with online tracking:
   ```bash
   python main.py --wandb_mode online
   ```
This will automatically upload all metrics, power-law fits, structural tables, and 300 DPI plots directly to your project page.
