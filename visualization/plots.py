import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


class PublicationVisualizer:
    """
    Attention Visualization System.
    Generates 300 DPI, publication-ready research figures based on transformer attention weights.
    Supports dual-format export in PNG and PDF (vector graphics).
    """
    def __init__(self, config: dict):
        self.config = config
        apply_scientific_style(config)
        
        self.results_dir = config["storage"]["results_dir"]
        self.figures_dir = os.path.join(self.results_dir, config["storage"]["figures_subdir"])
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.cmap_attn = config["visualization"]["cmap_attention"]
        self.cmap_heatmap = config["visualization"]["cmap_heatmap"]
        self.dpi = config["visualization"]["dpi"]
        self.img_format = config["visualization"].get("format", "png")

    def save_figure(self, filename_base: str):
        """
        Helper method to save the currently active figure as both 
        high-resolution PNG and vector PDF for print-ready publications.
        """
        png_path = os.path.join(self.figures_dir, f"{filename_base}.png")
        pdf_path = os.path.join(self.figures_dir, f"{filename_base}.pdf")
        
        # Save PNG
        plt.savefig(png_path, dpi=self.dpi, format="png", bbox_inches='tight')
        # Save vector PDF (perfect for zooming/scaling in papers)
        plt.savefig(pdf_path, dpi=self.dpi, format="pdf", bbox_inches='tight')
        
        plt.close()
        logger.info(f"Saved figure: {png_path} | {pdf_path}")

    def plot_attention_heatmap(self, attention_matrix: np.ndarray, tokens: list, layer: int, head: int, sample_idx: int):
        """
        1. Attention Heatmap
        Plots a beautiful 2D heat grid showing query-to-key attention weights.
        """
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
        self.save_figure(f"fig1_attention_heatmap_L{layer}_H{head}_S{sample_idx}")

    def plot_entropy_histogram(self, token_entropy: np.ndarray, max_entropy: float):
        """
        2. Entropy Histogram
        Plots the distribution of token attention entropies.
        """
        # Flatten token_entropy over samples, layers, heads, and positions
        flat_entropy = token_entropy.flatten()
        mean_entropy = np.mean(flat_entropy)
        
        plt.figure(figsize=(8, 5))
        sns.histplot(flat_entropy, bins=50, kde=True, color='#1f77b4', stat="density")
        
        # Add marker lines
        plt.axvline(mean_entropy, color='red', linestyle='--', label=f'Mean Entropy: {mean_entropy:.3f}')
        plt.axvline(max_entropy, color='black', linestyle=':', label=f'Theoretical Max: {max_entropy:.3f} (Uniform)')
        
        plt.xlabel("Attention Entropy (Nat)")
        plt.ylabel("Density")
        plt.title("Distribution of Attention Entropy across Query Positions")
        plt.legend(loc="upper left")
        
        self.save_figure("fig2_entropy_histogram")

    def plot_sparsity_histogram(self, sparsity_percentage: np.ndarray):
        """
        3. Sparsity Histogram
        Plots the distribution of head/token sparsity percentages.
        """
        flat_sparsity = sparsity_percentage.flatten()
        mean_sparsity = np.mean(flat_sparsity)
        
        plt.figure(figsize=(8, 5))
        sns.histplot(flat_sparsity, bins=50, kde=True, color='#2ca02c', stat="density")
        
        plt.axvline(mean_sparsity, color='red', linestyle='--', label=f'Mean Sparsity: {mean_sparsity:.2f}%')
        
        plt.xlabel("Sparsity % (Tokens under 1e-4)")
        plt.ylabel("Density")
        plt.title("Distribution of Attention Sparsity")
        plt.legend(loc="upper left")
        
        self.save_figure("fig3_sparsity_histogram")

    def plot_top_k_curve(self, top_k_masses: dict):
        """
        4. Top-k Mass Curve
        Plots the cumulative attention mass concentration in the top K key positions.
        """
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
        
        self.save_figure("fig4_top_k_curve")

    def plot_layerwise_comparison(self, head_entropy: np.ndarray, sparsity_percentage: np.ndarray):
        """
        5. Layer-wise Comparison Plot
        Compares distributions of entropy and sparsity by layer using box plots.
        """
        num_layers = head_entropy.shape[1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left Boxplot: Entropy distribution by layer
        entropy_data = [head_entropy[:, layer, :].flatten() for layer in range(num_layers)]
        axes[0].boxplot(entropy_data, labels=[str(i) for i in range(num_layers)], patch_artist=True,
                        boxprops=dict(facecolor='#aec7e8', color='#1f77b4'),
                        medianprops=dict(color='red'))
        axes[0].set_xlabel("Layer ID")
        axes[0].set_ylabel("Head Entropy (Nat)")
        axes[0].set_title("Head Entropy Distribution by Layer")
        
        # Right Boxplot: Sparsity distribution by layer
        sparsity_data = [sparsity_percentage[:, layer, :].flatten() for layer in range(num_layers)]
        axes[1].boxplot(sparsity_data, labels=[str(i) for i in range(num_layers)], patch_artist=True,
                        boxprops=dict(facecolor='#c7c7c7', color='#7f7f7f'),
                        medianprops=dict(color='red'))
        axes[1].set_xlabel("Layer ID")
        axes[1].set_ylabel("Head Sparsity (%)")
        axes[1].set_title("Head Sparsity Distribution by Layer")
        
        plt.suptitle("Layer-wise Attention Behavior Comparison", y=0.98, fontsize=14)
        self.save_figure("fig5_layerwise_comparison")

    def plot_headwise_comparison(self, head_entropy: np.ndarray, sparsity_percentage: np.ndarray):
        """
        6. Head-wise Comparison Plot
        Creates 12x12 grid maps of attention statistics (Mean Entropy, Mean Sparsity).
        """
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
        self.save_figure("fig6_headwise_comparison")

    def plot_attention_density(self, density_data: np.ndarray):
        """
        7. Attention Density Plot
        Plots density distribution curve across dataset.
        """
        flat_density = density_data.flatten()
        mean_density = np.mean(flat_density)
        
        plt.figure(figsize=(8, 5))
        sns.histplot(flat_density, bins=50, kde=True, color='#ff7f0e', stat="density")
        
        plt.axvline(mean_density, color='red', linestyle='--', label=f'Mean Density: {mean_density:.4f}')
        
        plt.xlabel("Attention Density (Active keys / seq_len)")
        plt.ylabel("Density")
        plt.title("Distribution of Attention Density (Key weights > 1/L)")
        plt.legend(loc="upper right")
        
        self.save_figure("fig7_attention_density")
