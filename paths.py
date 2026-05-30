from pathlib import Path
import os

class PathManager:
    """
    Centralized Path Manager for RSA-X.
    Uses pathlib to absolutely resolve the single project root and all subdirectories,
    ensuring no nested path duplication (e.g. RSA_X/RSA_X/RSA_X).
    """
    # Anchor to the directory containing this file
    _root = Path(__file__).resolve().parent

    @classmethod
    def get_project_root(cls) -> Path:
        """Returns the absolute project root Path."""
        return cls._root

    @classmethod
    def get_results_dir(cls) -> Path:
        """Returns the absolute results directory Path."""
        return cls._root / "results"

    @classmethod
    def get_configs_dir(cls) -> Path:
        """Returns the absolute configs directory Path."""
        return cls._root / "configs"

    @classmethod
    def get_wandb_dir(cls) -> Path:
        """Returns the absolute wandb directory Path."""
        return cls._root / "wandb"

    @classmethod
    def resolve_path(cls, relative_path: str) -> Path:
        """Resolves any path relative to the single project root absolutely."""
        return (cls._root / relative_path).resolve()
