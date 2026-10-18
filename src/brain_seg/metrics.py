import torch

CLASS_IDS = (1,2,3)
CLASS_NAMES = {1: "edema", 2: "non_enh", 3: "enh"}


def dice_per_class(pred, gt, labels = CLASS_IDS):

    pred = torch.as_tensor(pred)
    gt = torch.as_tensor(gt)
    out = []

    for c in labels:
        p = pred == c
        g = gt == c
        inter = (p & g).sum().item()
        denom = p.sum().item() + g.sum().item()
        out.append(1.0 if denom == 0 else 2.0*inter/denom)
    return torch.tensor(out)