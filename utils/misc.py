import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

EPSILON = 1e-7


class Struct(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def copy2cpu(data: Union[torch.Tensor, Dict]) -> Union[np.ndarray, Dict]:
    if isinstance(data, dict):
        return {k: v.detach().cpu().numpy() for k, v in data.items()}
    return data.detach().cpu().numpy()


def filename(f: str) -> str:
    return os.path.splitext(os.path.basename(f))[0]


def load_tensor_dict(path: str, keys: Optional[List[str]] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    data = np.load(path)
    if keys is None or len(keys) == 0:
        keys = list(data.keys())

    result = dict()
    for key in keys:
        if device is None:
            result[key] = torch.tensor(data[key], dtype=torch.float32)
        else:
            result[key] = torch.tensor(data[key], dtype=torch.float32, device=device)

    return result


def flatten_list(l: List[Any]) -> List[Any]:
    return [item for sublist in l for item in sublist]

def build_rotation_matrix(rot_vec: torch.Tensor) -> torch.Tensor:
    """
    rot_vec: [3] axis‐angle vector
    returns R: [3×3] rotation matrix
    """
    angle = rot_vec.norm()
    if angle < 1e-8:
        return torch.eye(3, device=rot_vec.device)
    axis = rot_vec / angle
    ux, uy, uz = axis
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    omc = 1 - cos
    return torch.stack([
        torch.stack([cos + ux*ux*omc,
                      ux*uy*omc - uz*sin,
                      ux*uz*omc + uy*sin]),
        torch.stack([uy*ux*omc + uz*sin,
                      cos + uy*uy*omc,
                      uy*uz*omc - ux*sin]),
        torch.stack([uz*ux*omc - uy*sin,
                      uz*uy*omc + ux*sin,
                      cos + uz*uz*omc])
    ])