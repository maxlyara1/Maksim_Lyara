import os
import random
from typing import Optional

import numpy as np

from .constants import RANDOM_SEED

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Устанавливает seed для всех генераторов случайных чисел.

    Args:
        seed: Значение seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def detect_torch_device(prefer_gpu: bool = True) -> str:
    """
    Определяет устройство для PyTorch моделей.

    Args:
        prefer_gpu: Предпочитать ли GPU если доступно.

    Returns:
        Название устройства: "cpu", "cuda" или "mps".
    """
    if torch is None:
        return "cpu"
    forced = os.environ.get("TORCH_DEVICE")
    if forced in {"cpu", "cuda", "mps"}:
        if forced == "mps":
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        return forced
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    if prefer_gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        return "mps"
    return "cpu"


def detect_catboost_task_type() -> str:
    """
    Определяет тип задачи для CatBoost (CPU или GPU).

    Returns:
        "CPU" или "GPU".
    """
    if torch is not None and torch.cuda.is_available():
        return "GPU"
    return "CPU"


def detect_device(prefer_gpu: bool = True) -> str:
    """Алиас для обратной совместимости."""
    return detect_torch_device(prefer_gpu=prefer_gpu)
