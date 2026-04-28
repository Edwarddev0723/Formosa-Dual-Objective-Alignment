"""formosa_dual.utils.seeding — reproducible random seeding."""
import random

import numpy as np
import torch

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed all randomness sources for reproducibility.

    Seeds: :mod:`random`, :mod:`numpy`, :mod:`torch` (CPU, all CUDA GPUs,
    and MPS if available).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    logger.info("Global seed set to %d", seed)
