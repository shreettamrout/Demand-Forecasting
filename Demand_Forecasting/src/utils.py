import os
import yaml
import random
import numpy as np
import torch
from pathlib import Path


# ============================================================
# CONFIGURATION UTILITIES
# ============================================================

def load_config(path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ============================================================
# REPRODUCIBILITY UTILITIES
# ============================================================

def set_seed(seed: int = 42):
    """
    Set global random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🧩 Random seed set to: {seed}")


# ============================================================
# FILESYSTEM UTILITIES
# ============================================================

def ensure_dir(path: str):
    """
    Create directory if it doesn't exist.

    Args:
        path (str): Path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def list_files(directory: str, extension: str = None):
    """
    List all files in a directory optionally filtered by extension.

    Args:
        directory (str): Directory path.
        extension (str): File extension filter (e.g. '.csv').

    Returns:
        list: List of file paths.
    """
    if not os.path.exists(directory):
        return []
    if extension:
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    else:
        return [os.path.join(directory, f) for f in os.listdir(directory)]


# ============================================================
# MODEL SAVE / LOAD UTILITIES
# ============================================================

def save_model(model, path: str):
    """
    Save a PyTorch model state dictionary.

    Args:
        model (torch.nn.Module): Model to save.
        path (str): Path to save checkpoint.
    """
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model_state(model, path: str, map_location=None):
    """
    Load a saved model state dictionary into a given model.

    Args:
        model (torch.nn.Module): Model instance.
        path (str): Path to saved state_dict.
        map_location: Device mapping for loading.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {path}")
    return model


# ============================================================
# LOGGING UTILITIES
# ============================================================

def print_section(title: str):
    """
    Print a formatted section header for console readability.

    Args:
        title (str): Section title to print.
    """
    print("\n" + "=" * 60)
    print(f"{title.upper()}")
    print("=" * 60)


def log_metrics(metrics: dict):
    """
    Nicely print out evaluation metrics.

    Args:
        metrics (dict): Dictionary of metric_name: value
    """
    print_section("MODEL EVALUATION METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
