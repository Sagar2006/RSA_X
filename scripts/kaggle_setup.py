import os
import sys
import logging
import torch

# Configure simple stdout logging
logging.basicConfig(
    level=logging.INFO,
    format='[KAGGLE SETUP] %(levelname)s - %(message)s'
)
logger = logging.getLogger("kaggle_setup")

def verify_gpu() -> bool:
    """Verifies GPU state and properties in the container."""
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU Verification: SUCCESS (Found device: '{gpu_name}')")
        return True
    else:
        logger.warning("GPU Verification: FAILED (No NVIDIA GPU detected). Execution will default to CPU-only.")
        return False


def verify_cuda() -> bool:
    """Verifies PyTorch CUDA capability and versions."""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        logger.info(f"CUDA Verification: SUCCESS (CUDA Version: {cuda_version}, cuDNN Version: {cudnn_version})")
        return True
    else:
        logger.info("CUDA Verification: N/A (GPU acceleration is disabled/unsupported on this node)")
        return False


def verify_dependencies() -> bool:
    """Verifies that all primary scientific dependencies are correctly installed in the system."""
    required_packages = [
        ("torch", "PyTorch Core"),
        ("transformer_lens", "TransformerLens (Mechanistic Interpretability)"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("scipy", "SciPy Stats"),
        ("pandas", "Pandas DataFrames"),
        ("matplotlib", "Matplotlib Plotting"),
        ("seaborn", "Seaborn Visual Styling"),
        ("wandb", "Weights & Biases Tracker")
    ]
    
    missing_packages = []
    logger.info("Verifying Python packages imports...")
    
    for pkg_name, label in required_packages:
        try:
            __import__(pkg_name)
            logger.info(f" - {label} ({pkg_name}): INSTALLED")
        except ImportError:
            logger.error(f" - {label} ({pkg_name}): MISSING")
            missing_packages.append(pkg_name)
            
    if missing_packages:
        logger.error(
            f"Verification FAILED: {len(missing_packages)} essential packages are missing. "
            "Please run: pip install -r requirements.txt"
        )
        return False
        
    logger.info("Package Verification: SUCCESS (All dependencies are fully satisfied)")
    return True


def create_output_folders() -> bool:
    """Tests writing permissions by creating output folders in the workspace."""
    test_folders = [
        "results",
        "configs",
        "logs"
    ]
    try:
        for folder in test_folders:
            os.makedirs(folder, exist_ok=True)
        logger.info("Workspace Directory Write Verification: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"Workspace Directory Write Verification: FAILED ({e})")
        return False


def print_environment_summary():
    """Prints a clean environment profile diagnostics block."""
    logger.info("================================================================")
    logger.info("              KAGGLE COMPATIBILITY ENVIRONMENT REPORT           ")
    logger.info("================================================================")
    logger.info(f" Python Version      : {sys.version.split()[0]}")
    logger.info(f" PyTorch Build       : {torch.__version__}")
    
    gpu_ok = verify_gpu()
    verify_cuda()
    pkgs_ok = verify_dependencies()
    write_ok = create_output_folders()
    
    logger.info("----------------------------------------------------------------")
    if gpu_ok and pkgs_ok and write_ok:
        logger.info("STATUS: READY (Your Kaggle environment is fully prepared for RSA-X!)")
    elif pkgs_ok and write_ok:
        logger.info("STATUS: READY (Local CPU Execution mode. No GPU acceleration available)")
    else:
        logger.warning("STATUS: CONFIGURATION REQUIRED (Please resolve missing packages first)")
    logger.info("================================================================")


def main():
    print_environment_summary()


if __name__ == "__main__":
    main()
