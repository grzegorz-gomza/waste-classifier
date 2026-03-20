from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class VectorizeConfig:
    flatten: bool = True


def tensor_to_feature_vector(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW image tensor to a flat feature vector.

    Assumes input is a torch tensor (C, H, W) with numeric dtype.
    """
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims (C,H,W), got shape {tuple(image_tensor.shape)}")

    return image_tensor.detach().cpu().numpy().reshape(-1)


def batch_to_feature_matrix(images: Sequence[torch.Tensor]) -> np.ndarray:
    vectors = [tensor_to_feature_vector(img) for img in images]
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32)
