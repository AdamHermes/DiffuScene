"""JSON conversion for numeric metadata written by inference scripts."""

import numpy as np
import torch


def json_default(value):
    """Preserve array/scalar values as JSON data, including nested metadata."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))
