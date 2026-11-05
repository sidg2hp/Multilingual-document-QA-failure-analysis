"""Deterministic global seeding across Python, NumPy, and PyTorch."""

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Seeds all available RNGs and requests deterministic cuDNN behaviour."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
