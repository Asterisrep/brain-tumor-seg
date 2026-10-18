import time
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss

from .dataset import build_loaders
from .evaluate import predict_volume, validate
from .model import UNet2D

RUN_NAME = "cosine-lr"
ROOT = Path.home() / "brain-tumor-seg"
DATA_DIR = ROOT / "data"
SPLIT = ROOT / "configs" / "split_seed42.json"
OUT_DIR = ROOT / "checkpoints" / RUN_NAME

USE_WANDB = True
BATCH_SIZE = 2
ITERS = 5000
LR = 3e-4
LR_MIN = 1e-6
PRINT_EVERY = 50
VAL_EVERY = 500
VAL_CASES = 20
CLASS_LABELS = {0: "background", 1: "edema", 2: "non-enh", 3: "enh"}


def prediction_image(wandb, model, case):
    img = case["image"].float()
    gt = case["label"][0].long()
    pred = predict_volume(model, img)
    k = int((gt > 0).sum(dim=(0, 1)).argmax())

    def upright(a):
        return np.ascontiguousarray(a.T[::-1])

    t1gd = upright(img[2, :, :, k].numpy())
    t1gd = ((t1gd - t1gd.min()) / (t1gd.max() - t1gd.min() + 1e-8) * 255).astype(np.uint8)
    masks = {
        "prediction": {"mask_data": upright(pred[:, :, k].numpy()).astype(np.uint8),
                       "class_labels": CLASS_LABELS},
        "ground_truth": {"mask_data": upright(gt[:, :, k].numpy()).astype(np.uint8),
                         "class_labels": CLASS_LABELS},
    }
    return wandb.Image(t1gd, masks=masks, caption=f"slice {k}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = build_loaders(
        DATA_DIR, SPLIT, batch_size=BATCH_SIZE, num_workers=0,
        val_subset="val", val_limit=VAL_CASES,
    )

    model = UNet2D(in_ch=4, out_ch=4, base=32).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ITERS, eta_min=LR_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = dict(
        batch_size=BATCH_SIZE, crops_per_volume=4, patch=160, iters=ITERS, lr=LR,
        lr_schedule="cosine", lr_min=LR_MIN,
        optimizer="AdamW", loss="DiceCELoss", base_channels=32, amp=use_amp,
        val_cases=VAL_CASES, seed=42, split=SPLIT.name,
        n_params=sum(p.numel() for p in model.parameters()),
    )

    wandb, run, viz_case = None, None, None
    if USE_WANDB:
        import wandb
        run = wandb.init(project="brain-seg", name=RUN_NAME, config=config)
        viz_case = val_dl.dataset[0]

    step, best, running = 0, -1.0, 0.0
    start = time.time()

    while step < ITERS:
        for batch in train_dl:
            model.train()
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            running += loss.item()
            if step % PRINT_EVERY == 0:
                avg, running = running / PRINT_EVERY, 0.0
                s_per_step = (time.time() - start) / step
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {step:5d} | loss {avg:.4f} | lr {lr_now:.2e} | {s_per_step:.2f} s/step")
                if run:
                    run.log({"train/loss": avg, "train/s_per_step": s_per_step,
                             "train/lr": lr_now}, step=step)

            if step % VAL_EVERY == 0 or step == ITERS:
                mean, scores = validate(model, val_dl)
                score = sum(mean.values()) / len(mean)
                print("  val dice:", {k: round(v, 3) for k, v in mean.items()},
                      "| mean", round(score, 3))
                if run:
                    log = {f"val/dice_{k}": v for k, v in mean.items()}
                    log["val/dice_mean"] = score
                    log["val/prediction"] = prediction_image(wandb, model, viz_case)
                    run.log(log, step=step)
                ckpt = {"model": model.state_dict(), "step": step, "val": mean}
                torch.save(ckpt, OUT_DIR / "last.pt")
                if score > best:
                    best = score
                    torch.save(ckpt, OUT_DIR / "best.pt")

            if step >= ITERS:
                break

    print("best mean val dice:", round(best, 4))
    if run:
        run.summary["best_val_dice"] = best
        art = wandb.Artifact("unet2d", type="model", metadata=config)
        art.add_file(str(OUT_DIR / "best.pt"))
        run.log_artifact(art)
        table = wandb.Table(
            columns=["case", "edema", "non_enh", "enh"],
            data=[[i] + [float(v) for v in row] for i, row in enumerate(scores)])
        run.log({"val/per_case": table})
        run.finish()


if __name__ == "__main__":
    main()