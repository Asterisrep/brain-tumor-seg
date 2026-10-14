import argparse
import json
from pathlib import Path

from monai.data import Dataset, PersistentDataset

from .transforms import build_preprocess


def read_cases(data_dir):
    """dataset.json -> {case_name: {"image": path, "label": path}}"""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "dataset.json").read_text())
    cases = {}
    for it in meta["training"]:
        img = data_dir / it["image"].removeprefix("./")
        lbl = data_dir / it["label"].removeprefix("./")
        cases[img.name] = {"image": str(img), "label": str(lbl)}
    return cases


def load_split(split_path):
    """configs/split_seed42.json -> {"train": [...names], "val": [...], "test": [...]}"""
    return json.loads(Path(split_path).read_text())


def build_dataset(data_dir, split_path, subset, cache_dir=None, transform=None):
    """subset is 'train', 'val' or 'test'.
    cache_dir=None -> processed on the fly; otherwise cached on disk."""
    cases = read_cases(data_dir)
    data = [cases[name] for name in load_split(split_path)[subset]]
    transform = transform if transform is not None else build_preprocess()
    if cache_dir is not None:
        return PersistentDataset(data=data, transform=transform,
                                 cache_dir=Path(cache_dir) / subset)
    return Dataset(data=data, transform=transform)


def main():
    root = Path.home() / "brain-tumor-seg"
    p = argparse.ArgumentParser(description="Sanity check of the data pipeline")
    p.add_argument("--data-dir", default=str(root / "data"))
    p.add_argument("--split", default=str(root / "configs" / "split_seed42.json"))
    p.add_argument("--subset", default="train", choices=["train", "val", "test"])
    args = p.parse_args()

    # 1) the split: no overlap, nothing lost
    split = load_split(args.split)
    sizes = {k: len(v) for k, v in split.items()}
    print("split sizes:", sizes)
    names = [n for v in split.values() for n in v]
    assert len(names) == len(set(names)), "a case appears in more than one split"

    # 2) every case in the split exists in dataset.json, and its files exist
    cases = read_cases(args.data_dir)
    missing = [n for n in names if n not in cases]
    assert not missing, f"not in dataset.json: {missing[:3]}"
    for n in names:
        assert Path(cases[n]["image"]).exists() and Path(cases[n]["label"]).exists(), n

    # 3) build the dataset and check one case
    ds = build_dataset(args.data_dir, args.split, args.subset)
    print(f"{args.subset}: {len(ds)} cases")
    s = ds[0]
    img, lbl = s["image"], s["label"]
    print("image:", tuple(img.shape), img.dtype)
    print("label:", tuple(lbl.shape), lbl.dtype)

    assert img.shape[0] == 4, "expected 4 channels first"
    assert lbl.shape[0] == 1, "expected a channel axis on the label"
    assert img.shape[1:] == lbl.shape[1:], "image and label sizes differ after the crop"
    vals = set(lbl.unique().tolist())
    assert vals <= {0, 1, 2, 3}, f"unexpected label values {vals}"
    print("label values:", sorted(vals))
    print("OK")


if __name__ == "__main__":
    main()