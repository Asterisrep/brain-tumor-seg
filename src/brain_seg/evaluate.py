import torch
import torch.nn.functional as F

from .metrics import CLASS_IDS, CLASS_NAMES, dice_per_class


@torch.no_grad()
def predict_volume(model, img, batch=16, div=16):
    model.eval()
    dev = next(model.parameters()).device
    C, H, W, D = img.shape
    x = img.permute(3, 0, 1, 2)
    x = F.pad(x, (0, (-W) % div, 0, (-H) % div))   
    out = []
    for s in range(0, D, batch):
        xb = x[s:s + batch].to(dev).float()
        with torch.autocast(dev.type, dtype = torch.float16, enabled=dev.type=="cuda"):
            logits = model(xb)
        out.append(logits.argmax(1)[:, :H, :W].cpu())
    return torch.cat(out).permute(1, 2, 0)  



def validate(model, loader):
    """loader yields whole volumes (batch size 1).
    Returns (mean Dice per class as a dict, per-case scores as an (N, 3) tensor)."""
    scores = []
    for b in loader:
        img = b["image"][0].float()                      # (4, H, W, D); the cache stores float16
        gt = b["label"][0, 0].long()                     # (H, W, D)
        scores.append(dice_per_class(predict_volume(model, img), gt))
    scores = torch.stack(scores)                         # (N, 3)
    mean = {CLASS_NAMES[c]: float(scores[:, i].mean()) for i, c in enumerate(CLASS_IDS)}
    return mean, scores