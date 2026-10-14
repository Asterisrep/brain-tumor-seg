import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, CropForegroundd,
    EnsureTyped, MapTransform,
)

CHANNELS = ["FLAIR", "T1w", "T1gd", "T2w"]   # channel order from dataset.json


class NormalizeBrainD(MapTransform):
    """Per channel: clip to brain percentiles, z-score on brain voxels, background = 0.

    Expects channel-first arrays (C, H, W, D). Statistics come from the single
    case only, so the same code works on one new scan at inference time.
    """

    def __init__(self, keys, p_low=1, p_high=99):
        super().__init__(keys)
        self.p_low, self.p_high = p_low, p_high

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            x = np.asarray(d[k], dtype=np.float32)
            brain = (x != 0).any(axis=0)
            out = np.zeros_like(x)
            for c in range(x.shape[0]):
                v = x[c][brain]
                lo, hi = np.percentile(v, [self.p_low, self.p_high])
                v = np.clip(v, lo, hi)
                out[c][brain] = (v - v.mean()) / (v.std() + 1e-8)
            d[k] = torch.from_numpy(out)
        return d


def build_preprocess(margin=4, p_low=1, p_high=99):
    """Deterministic steps only (safe to cache). Random augmentation goes after this.

    The label stays as raw integers 0-3 (uint8). It gets a channel axis and the
    same crop as the image, and nothing else.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=margin),
        NormalizeBrainD(keys=["image"], p_low=p_low, p_high=p_high),
        EnsureTyped(keys=["image"], dtype=torch.float16),
        EnsureTyped(keys=["label"], dtype=torch.uint8),
    ])