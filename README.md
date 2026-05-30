# RSA-X (Reinforced Sparse Attention eXperimental Framework)

RSA-X is a research-grade experimental framework designed to analyze and profile the natural attention sparsity in pretrained transformers. This repository contains the **Phase 1.5 Infrastructure Upgrade**, enhancing environmental diagnostics, config cascade merging, results isolation, vector exports, and zero-dependency remote container support (Kaggle).

The core objective is to empirically analyze whether natural attention sparsity is concentrated in specific layers/heads, providing evidence for whether transformer attention can be safely optimized from $O(n^2)$ to $O(nk)$ without quality loss.

---

## 1. Scientific Methodology & Math Formulations

This framework computes key attention characteristics over the attention probability tensors $A \in \mathbb{R}^{B \times L_{layer} \times L_{head} \times N \times N}$ extracted from GPT-2 Small, where $N$ is the sequence length (default 256):

### 1.1 Shannon Attention Entropy ($H$)
$$H(i) = -\sum_{j=1}^{N} p_{ij} \log p_{ij}$$
where $p_{ij}$ is the attention weight from query $i$ to key $j$, such that $\sum_{j} p_{ij} = 1.0$.

### 1.2 Top-K Cumulative Mass ($M_K$)
$$M_K(i) = \sum_{j \in \mathcal{K}_i} p_{ij}$$
where $\mathcal{K}_i$ is the set of $K$ indices with the largest attention weights $p_{ij}$ for query $i$.

### 1.3 Attention Density ($D$)
$$D(i) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{I}\left(p_{ij} > \frac{1}{N}\right)$$

### 1.4 Sparsity Percentage ($S$)
$$S(i) = \frac{100}{N} \sum_{j=1}^{N} \mathbb{I}\left(p_{ij} \le 10^{-4}\right)$$

### 1.5 Power-Law Exponent ($\alpha$)
$$\log(\text{weight}_r) = -\alpha \log(r) + C$$
where $r \in \{1, 2, \dots, N\}$ is the descending rank of the attention weight.

---

## 2. Phase 1.5 Infrastructure Upgrades

The codebase features several professional scientific computing infrastructure upgrades:

### 2.1 Cascading Configuration Profiles
We implement a deep, cascading dictionary merge in `main.py`. The baseline configurations are loaded from `configs/default_config.yaml`, and environment-specific keys are merged over them dynamically.
- **[configs/local.yaml](file:///c:/Users/sagar/Desktop/RSA_X/configs/local.yaml):** CPU-only, 25 samples, sequence length 128, batch size 1.
- **[configs/kaggle.yaml](file:///c:/Users/sagar/Desktop/RSA_X/configs/kaggle.yaml):** CUDA T4 GPU optimized, 1000 samples, sequence length 512, batch size 4.
- **[configs/research.yaml](file:///c:/Users/sagar/Desktop/RSA_X/configs/research.yaml):** Baseline full-scale analysis template (100 samples, sequence length 256, batch size 2).

### 2.2 System Hardware Diagnostics
A zero-dependency profiler `rsa_x/hardware.py` inspects the operating system, processor models, physical RAM sizes, CUDA availability, GPU device names, and VRAM. A diagnostic summary block is logged and printed automatically at the start of every experiment.

### 2.3 Run Isolation & Results Directory Partitioning
To prevent overwriting prior experimental assets, every execution creates a timestamped isolated run folder:
`results/run_YYYY_MM_DD_HH_MM_SS/` containing:
- `figures/` — High-resolution plots.
- `metrics/` — Compressed binary `.npz` arrays and research CSV summary tables.
- `logs/` — Dynamic execution log captures.

### 2.4 Vector Graphics Figure Export (PNG + PDF)
Every figure is exported simultaneously in two formats:
- **PNG Format:** For rapid visual inspection and Weights & Biases dashboard logging.
- **PDF Format:** In true vector graphics format at 300 DPI, satisfying strict camera-ready academic publication standards (enabling infinite scaling without pixelation).

### 2.5 Experiment Metadata Logging
Every run compiles a detailed `experiment_metadata.json` documenting the start timestamp, fallback-resilient Git commit hash, config path utilized, model/dataset specifications, system hardware properties, and total execution duration (measured via `time.perf_counter`).

### 2.6 Kaggle Compatibility Checker
A dedicated remote-container utility `scripts/kaggle_setup.py` checks GPU/CUDA status, package imports, directory write permissions, and profiles system readiness inside Kaggle containers prior to runtime.

---

## 3. Local CPU Execution Workflow

To run the framework on a local CPU-friendly system (no GPU required):

```bash
# 1. Clone/pull the repository and enter the directory
git pull

# 2. Setup and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux/MacOS

# 3. Install scientific dependencies
pip install -r requirements.txt

# 4. Run using the local CPU override profile
python main.py --config configs/local.yaml
```

**Expected Outputs:**
- Printed hardware profile.
- A newly created timestamped directory e.g., `results/run_2026_05_30_23_10_45/` containing:
  - `experiment_metadata.json`
  - `logs/experiment.log`
  - `metrics/layerwise_summary_table.csv` and `consolidated_metrics.npz`
  - `raw_tensors/sample_0.npy` and `sample_0.parquet`
  - `figures/fig1_...png` to `fig7_...png` AND vector equivalents `fig1_...pdf` to `fig7_...pdf`.

---

## 4. Kaggle T4 GPU Execution Workflow

To run full-scale experiments inside a Kaggle Notebook container equipped with an NVIDIA T4 GPU accelerator:

```python
# --- STEP 1: IN A KAGGLE NOTEBOOK CELL ---
# Clone the repository and enter
!git clone <repository_url>
%cd RSA_X

# --- STEP 2: IN A KAGGLE NOTEBOOK CELL ---
# Install the framework requirements
!pip install -r requirements.txt

# --- STEP 3: IN A KAGGLE NOTEBOOK CELL ---
# Run environment diagnostics to verify T4 GPU and dependencies
!python -m scripts.kaggle_setup

# --- STEP 4: IN A KAGGLE NOTEBOOK CELL ---
# Execute the full-scale T4-GPU optimized experiment run
!python main.py --config configs/kaggle.yaml
```

**Expected Outputs:**
- Diagnostics summary verifying CUDA active status, "Tesla T4" GPU, and total VRAM (~15.0 GB).
- Isolated runs saved in `results/run_YYYY_MM_DD_HH_MM_SS/` containing all metric CSVs, compressed NPZ arrays, raw sparse parquet files, and high-quality vector PDF/PNG plots.

---

## 5. Standalone Figure Regeneration

To adjust plot details or regenerate figures from a prior run without re-running the neural model forward pass, call the utility. It automatically targets the most recent run subdirectory if no argument is passed:

```bash
# Target the most recent run automatically
python -m scripts.generate_figures

# Target a specific past run folder
python -m scripts.generate_figures --run_dir results/run_2026_05_30_22_19_00
```

---

## 6. Verification and Testing

Verify mathematical properties and calculations via the pytest suite:

```bash
pytest tests/test_framework.py
```
