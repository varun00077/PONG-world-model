# Small helpers shared by all scripts
import random

import numpy as np
import torch
import torch.nn.functional as F

from pong_game_env import My_pong


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_episodes(path):
    """Load a dataset saved by collect_data.py.

    Returns
        frames  : float32 tensor (N, T+1, 3, 32, 32) in [0, 1]
        actions : int64 tensor   (N, T)
    """
    data = np.load(path)
    frames = torch.from_numpy(data["frames"]).float() / 255.0
    frames = frames.permute(0, 1, 4, 2, 3).contiguous()   # (N, T+1, H, W, C) -> (N, T+1, C, H, W)
    actions = torch.from_numpy(data["actions"]).long()
    return frames, actions


# The ball is only a few pixels, so a plain loss can learn to ignore it.
# Pixels that differ from the background get a larger weight.
BACKGROUND = torch.tensor(My_pong.BACKGROUND).view(1, 3, 1, 1)


def object_weight(target, weight=20.0):
    is_object = (target - BACKGROUND.to(target.device)).abs().sum(dim=1, keepdim=True) > 0.1
    return 1.0 + weight * is_object.float()


def weighted_bce(pred, target):
    """Object-weighted binary cross-entropy, summed over pixels, averaged over images.
    Used as the training loss: summing per image keeps it on the same scale as the
    latent terms (smoothness, size penalty)."""
    pred = pred.reshape(-1, *pred.shape[-3:])
    target = target.reshape(-1, *target.shape[-3:])
    bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target, reduction="none")
    return (object_weight(target) * bce).sum() / len(target)


def weighted_mse(pred, target):
    """Object-weighted MSE averaged over pixels. Used as an easy-to-read evaluation number."""
    pred = pred.reshape(-1, *pred.shape[-3:])
    target = target.reshape(-1, *target.shape[-3:])
    return (object_weight(target) * (pred - target) ** 2).mean()
