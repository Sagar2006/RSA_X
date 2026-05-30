import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_scientific_style(config: dict):
    """
    Sets global matplotlib settings for publication-quality scientific charts.
    """
    style_name = config["visualization"]["style"]
    try:
        plt.style.use(style_name)
    except Exception as e:
        logger.warning(f"Could not apply style {style_name} ({e}). Applying baseline clean grid style...")
        plt.rcParams.update({
            'axes.grid': True,
            'grid.color': '#cccccc',
            'grid.linestyle': '--',
            'grid.alpha': 0.6,
            'font.family': 'sans-serif',
            'font.size': 11,
            'legend.fontsize': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': config["visualization"]["dpi"]
        })
        
    # Standardize figure parameters
    plt.rcParams["figure.dpi"] = config["visualization"]["dpi"]
    plt.rcParams["savefig.dpi"] = config["visualization"]["dpi"]
    plt.rcParams["savefig.bbox"] = "tight"


def safe_downsample(array: np.ndarray, max_points: int = 50000) -> np.ndarray:
    """
    Downsamples a numpy array systematically to a max number of points 
    for fast plotting while preserving true statistics.
    """
    flat = array.flatten()
    if len(flat) > max_points:
        step = len(flat) // max_points
        return flat[::step]
    return flat


class PublicationVisualizer:
    """
    Attention Visualization System.
    Generates 300 DPI, publication-ready research figures based on transformer attention weights.
    Supports dual-format export in PNG and PDF (vector graphics).
    """
    def __init__(self, config: dict):
        self.config = config
        apply_scientific_style(config)
        
        self.results_dir = Path(config["storage"]["results_dir"]).resolve()
        self.figures_dir = self.results_dir / config["storage"]["figures_subdir"]
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.cmap_attn = config["visualization"]["cmap_attention"]
        self.cmap_heatmap = config["visualization"]["cmap_heatmap"]
        self.dpi = config["visualization"]["dpi"]
        self.img_format = config["visualization"].get("format", "png")
        
        # High-precision timings accumulator
        self.figure_timings = {}

    def save_figure(self, filename_base: str, start_plot_time: float):
        """
        Helper method to save the currently active figure as both 
        high-resolution PNG and vector PDF for print-ready publications.
        Profiles creation and save times independently.
        """
        png_path = self.figures_dir / f"{filename_base}.png"
        pdf_path = self.figures_dir / f"{filename_base}.pdf"
        
        # 1. Figure creation/rendering duration
        creation_end = time.perf_counter()
        creation_time = creation_end - start_plot_time
        
        # 2. PNG write duration
        png_start = time.perf_counter()
        plt.savefig(str(png_path), dpi=self.dpi, format="png", bbox_inches='tight')
        png_end = time.perf_counter()
        png_time = png_end - png_start
        
        # 3. PDF write duration
        pdf_start = time.perf_counter()
        plt.savefig(str(pdf_path), dpi=self.dpi, format="pdf", bbox_inches='tight')
        pdf_end = time.perf_counter()
        pdf_time = pdf_end - pdf_start
        
        plt.close()
        
        total_time = creation_time + png_time + pdf_time
        self.figure_timings[filename_base] = {
            "creation_time": creation_time,
            "save_png_time": png_time,
            "save_pdf_time": pdf_time,
            "total_time": total_time
        }
        
        logger.info(
            f"Saved figure: {filename_base} | "
            f"Creation: {creation_time:.3f}s | "
            f"PNG: {png_time:.3f}s | "
            f"PDF: {pdf_time:.3f}s"
        )

    def plot_attention_heatmap(self, attention_matrix: np.ndarray, tokens: list, layer: int, head: int, sample_idx: int):
        """
        1. Attention Heatmap
        Plots a beautiful 2D heat grid showing query-to-key attention weights.
        """
        start_time = time.perf_counter()
        seq_len = len(tokens)
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        ax = sns.heatmap(
            attention_matrix,
            cmap=self.cmap_attn,
            vmin=0.0,
            vmax=1.0,
            cbar_kws={'label': 'Attention Weight (Probability)'}
        )
        
        # Apply labels only for small sequence lengths to ensure legibility
        if seq_len <= 64:
            ax.set_xticks(np.arange(seq_len) + 0.5)
            ax.set_yticks(np.arange(seq_len) + 0.5)
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
            ax.set_yticklabels(tokens, rotation=0, fontsize=8)
        else:
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            
        ax.set_title(f"Attention Map: Layer {layer}, Head {head} (Sample {sample_idx})")
        self.save_figure(f"fig1_attention_heatmap_L{layer}_H{head}_S{sample_idx}", start_time)

    def plot_entropy_histogram(self, token_entropy: np.ndarray, max_entropy: float):
        """
        2. Entropy Histogram
        Plots the distribution of token attention entropies.
        Optimized via systematic downsampling.
        """
        start_time = time.perf_counter()
        
        # Keep exact mathematical mean from the full array
        mean_entropy = np.mean(token_entropy)
        
        # Downsample data points for fast seaborn plotting
        flat_entropy = safe_downsample(token_entropy, 50000)
        
        plt.figure(figsize=(8, 5))
        sns.histplot(flat_entropy, bins=50, kde=True, color='#1f77b4', stat="density")
        
        # Add marker lines
        plt.axvline(mean_entropy, color='red', linestyle='--', label=f'Mean Entropy: {mean_entropy:.3f}')
        plt.axvline(max_entropy, color='black', linestyle=':', label=f'Theoretical Max: {max_entropy:.3f} (Uniform)')
        
        plt.xlabel("Attention Entropy (Nat)")
        plt.ylabel("Density")
        plt.title("Distribution of Attention Entropy across Query Positions")
        plt.legend(loc="upper left")
        
        self.save_figure("fig2_entropy_histogram", start_time)

    def plot_sparsity_histogram(self, sparsity_percentage: np.ndarray):
        """
        3. Sparsity Histogram
        Plots the distribution of head/token sparsity percentages.
        Optimized via systematic downsampling.
        """
        start_time = time.perf_counter()
        mean_sparsity = np.mean(sparsity_percentage)
        flat_sparsity = safe_downsample(sparsity_percentage, 50000)
        
        plt.figure(figsize=(8, 5))
        sns.histplot(flat_sparsity, bins=50, kde=True, color='#2ca02c', stat="density")
        
        plt.axvline(mean_sparsity, color='red', linestyle='--', label=f'Mean Sparsity: {mean_sparsity:.2f}%')
        
        plt.xlabel("Sparsity % (Tokens under 1e-4)")
        plt.ylabel("Density")
        plt.title("Distribution of Attention Sparsity")
        plt.legend(loc="upper left")
        
        self.save_figure("fig3_sparsity_histogram", start_time)

    def plot_top_k_curve(self, top_k_masses: dict):
        """
        4. Top-k Mass Curve
        Plots the cumulative attention mass concentration in the top K key positions.
        """
        start_time = time.perf_counter()
        k_values = self.config["analysis"]["top_k_values"]
        
        # Extract mean and std values
        means = []
        stds = []
        for k in k_values:
            mass = top_k_masses[f"top_{k}_mass"].flatten()
            means.append(np.mean(mass))
            stds.append(np.std(mass))
            
        means = np.array(means)
        stds = np.array(stds)
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, means, marker='o', color='#9467bd', linewidth=2, label="Mean Cumulative Mass")
        plt.fill_between(k_values, means - stds, means + stds, color='#9467bd', alpha=0.15, label="Standard Deviation")
        
        plt.xscale('log')
        plt.xticks(k_values, labels=[str(k) for k in k_values])
        plt.xlabel("Top-k Key Positions (Log Scale)")
        plt.ylabel("Cumulative Attention Mass")
        plt.ylim(0, 1.05)
        plt.title("Cumulative Attention Mass Concentration Curves")
        plt.legend(loc="lower right")
        
        self.save_figure("fig4_top_k_curve", start_time)

    def plot_layerwise_comparison(self, head_entropy: np.ndarray, sparsity_percentage: np.ndarray):
        """
        5. Layer-wise Comparison Plot
        Compares distributions of entropy and sparsity by layer using box plots.
        Optimized by downsampling high-volume layer metrics.
        """
        start_time = time.perf_counter()
        num_layers = head_entropy.shape[1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left Boxplot: Entropy distribution by layer (averaged per head, small volume)
        entropy_data = [safe_downsample(head_entropy[:, layer, :], 10000) for layer in range(num_layers)]
        axes[0].boxplot(entropy_data, labels=[str(i) for i in range(num_layers)], patch_artist=True,
                        boxprops=dict(facecolor='#aec7e8', color='#1f77b4'),
                        medianprops=dict(color='red'))
        axes[0].set_xlabel("Layer ID")
        axes[0].set_ylabel("Head Entropy (Nat)")
        axes[0].set_title("Head Entropy Distribution by Layer")
        
        # Right Boxplot: Sparsity distribution by layer (token-level metrics, downsampled)
        sparsity_data = [safe_downsample(sparsity_percentage[:, layer, :], 10000) for layer in range(num_layers)]
        axes[1].boxplot(sparsity_data, labels=[str(i) for i in range(num_layers)], patch_artist=True,
                        boxprops=dict(facecolor='#c7c7c7', color='#7f7f7f'),
                        medianprops=dict(color='red'))
        axes[1].set_xlabel("Layer ID")
        axes[1].set_ylabel("Head Sparsity (%)")
        axes[1].set_title("Head Sparsity Distribution by Layer")
        
        plt.suptitle("Layer-wise Attention Behavior Comparison", y=0.98, fontsize=14)
        self.save_figure("fig5_layerwise_comparison", start_time)

    def plot_headwise_comparison(self, head_entropy: np.ndarray, sparsity_percentage: np.ndarray):
        """
        6. Head-wise Comparison Plot
        Creates 12x12 grid maps of attention statistics (Mean Entropy, Mean Sparsity).
        """
        start_time = time.perf_counter()
        # Average metrics across samples to get [num_layers, num_heads]
        mean_entropy_grid = head_entropy.mean(axis=0)
        mean_sparsity_grid = sparsity_percentage.mean(axis=(0, 3))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left Heatmap: Entropy Grid
        sns.heatmap(
            mean_entropy_grid,
            ax=axes[0],
            cmap="viridis",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar_kws={'label': 'Mean Entropy (Nat)'}
        )
        axes[0].set_xlabel("Head ID")
        axes[0].set_ylabel("Layer ID")
        axes[0].set_title("Layer vs. Head Average Attention Entropy")
        
        # Right Heatmap: Sparsity Grid
        sns.heatmap(
            mean_sparsity_grid,
            ax=axes[1],
            cmap="mako",
            annot=True,
            fmt=".1f",
            annot_kws={"size": 7},
            cbar_kws={'label': 'Mean Sparsity (%)'}
        )
        axes[1].set_xlabel("Head ID")
        axes[1].set_ylabel("Layer ID")
        axes[1].set_title("Layer vs. Head Average Attention Sparsity (%)")
        
        plt.suptitle("Head-level Specialized Attention Profiling", y=0.98, fontsize=14)
        self.save_figure("fig6_headwise_comparison", start_time)

    def plot_attention_density(self, density_data: np.ndarray):
        """
        7. Attention Density Plot
        Plots density distribution curve across dataset.
        Optimized via systematic downsampling.
        """
        start_time = time.perf_counter()
        mean_density = np.mean(density_data)
        flat_density = safe_downsample(density_data, 50000)
        
        plt.figure(figsize=(8, 5))
        sns.histplot(flat_density, bins=50, kde=True, color='#ff7f0e', stat="density")
        
        plt.axvline(mean_density, color='red', linestyle='--', label=f'Mean Density: {mean_density:.4f}')
        
        plt.xlabel("Attention Density (Active keys / seq_len)")
        plt.ylabel("Density")
        plt.title("Distribution of Attention Density (Key weights > 1/L)")
        plt.legend(loc="upper right")
        
        self.save_figure("fig7_attention_density", start_time)
