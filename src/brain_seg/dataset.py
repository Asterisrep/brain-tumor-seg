import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, EnsureTyped, SpatialPadd, RandCropByPosNegLabeld,
    SqueezeDimd, RandScaleIntensityd, RandFlipd,
)

from .data import build_dataset


def build_train_transform(patch=160, num_samples=4, pos_ratio=0.5, flip_prob=0.0):
    keys = ["image", "label"]
    tfs = [
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
        SpatialPadd(keys=keys, spatial_size=(patch, patch, -1)),
        RandCropByPosNegLabeld(
            keys=keys, label_key="label",
            spatial_size=(patch, patch, 1),
            pos=pos_ratio, neg=1 - pos_ratio, num_samples=num_samples,
        ),
        SqueezeDimd(keys=keys, dim=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    ]
    if flip_prob > 0:
        tfs.append(RandFlipd(keys=keys, spatial_axis=0, prob=flip_prob))
    return Compose(tfs)


def build_train_dataset(data_dir, split_path, cache_dir=None, limit=None, **kw):
    base = build_dataset(data_dir, split_path, "train", cache_dir=cache_dir, limit=limit)
    return Dataset(data=base, transform=build_train_transform(**kw))


def build_val_dataset(data_dir, split_path, cache_dir=None, subset="val", limit=None):
    """Whole volumes: the model predicts slice by slice and we score the full 3D volume."""
    return build_dataset(data_dir, split_path, subset, cache_dir=cache_dir, limit=limit)


def build_loaders(data_dir, split_path, cache_dir=None, batch_size=4,
                  num_workers=2, train_limit=None, val_subset="val",
                  val_limit=None, **kw):
    train_ds = build_train_dataset(data_dir, split_path, cache_dir,
                                   limit=train_limit, **kw)
    val_ds = build_val_dataset(data_dir, split_path, cache_dir,
                               subset=val_subset, limit=val_limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=num_workers)
    return train_dl, val_dl