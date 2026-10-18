import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from monai.data import DataLoader

from .data import load_split
from .dataset import build_val_dataset
from .evaluate import predict_volume
from .metrics import CLASS_IDS, CLASS_NAMES, dice_per_class
from .model import UNet2D

# only needed for the 3D mode; the 2D mode works without them
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    HAVE_3D = True
except ImportError:
    HAVE_3D = False

ROOT = Path.home() / "brain-tumor-seg"
DATA_DIR = ROOT / "data"
SPLIT = ROOT / "configs" / "split_seed42.json"

# label -> colour of the overlay (label 0 = background is left transparent)
COLORS = {1: "#f1c40f", 2: "#3498db", 3: "#e74c3c"}
CMAP = ListedColormap(["none"] + [COLORS[c] for c in CLASS_IDS])

OPACITY = {1: 0.45, 2: 0.85, 3: 1.0}       # 3D: labels clear and intense
BRAIN_COLOR = "#8a8f98"                     # 3D: the faint "ghost" shell of the whole brain
BRAIN_OPACITY = 0.07
VOLUME_OPACITY = 0.10                       # 3D: faint T1gd structures inside the brain


# ---------------------------------------------------------------- 2D plots
def draw(ax, base, mask, title, vmin, vmax):
    """Grey MRI slice with an optional label overlay. .T + origin='lower' = upright."""
    ax.imshow(base.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if mask is not None:
        ax.imshow(mask.T, cmap=CMAP, vmin=0, vmax=3, origin="lower",
                  alpha=0.55, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_case(path, img_sl, gt_sl, pred_sl, title):
    brain = img_sl[img_sl != 0]
    vmin, vmax = np.percentile(brain, [1, 99]) if brain.size else (0, 1)
    errors = np.where(gt_sl != pred_sl, 3, 0)          # 3 = drawn in red

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    draw(axes[0], img_sl, None, "image", vmin, vmax)
    draw(axes[1], img_sl, gt_sl, "ground truth", vmin, vmax)
    draw(axes[2], img_sl, pred_sl, "prediction", vmin, vmax)
    draw(axes[3], img_sl, errors, "errors (red = prediction differs)", vmin, vmax)
    fig.legend(handles=[Patch(color=COLORS[c], label=CLASS_NAMES[c]) for c in CLASS_IDS],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 3D plots
def mesh_of(mask, smooth=0.0, step=1, min_voxels=20):
    """Triangle mesh of the surface of a 3D boolean mask (None if it is too small).
    smooth: sigma (in voxels) of a Gaussian blur before meshing, 0 = none
    step:   marching-cubes step size, 2 = half the resolution (lighter mesh)."""
    if mask.sum() < min_voxels:
        return None
    vol = mask.astype(np.float32)
    if smooth > 0:
        vol = gaussian_filter(vol, smooth)
    padded = np.pad(vol, 1)                          # closes surfaces at the volume border
    verts, faces, _, _ = marching_cubes(padded, level=0.5, step_size=step)
    return verts - 1, faces


def brain_mesh(img):
    """Smooth, light surface of the whole brain (= everything that is not empty background)."""
    return mesh_of(img != 0, smooth=2.0, step=2)


def volume_data(img, step=4):
    """T1gd as flat x, y, z, value arrays for a volume rendering.
    Block-averaged by `step` (smaller = finer but heavier), scaled to 0..1,
    empty background = 0."""
    brain = img[img != 0]
    lo, hi = np.percentile(brain, [1, 99]) if brain.size else (0, 1)
    disp = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    disp[img == 0] = 0
    h, w, d = (s // step * step for s in img.shape)               # cut to multiples of step
    sub = disp[:h, :w, :d].reshape(h // step, step, w // step, step,
                                   d // step, step).mean(axis=(1, 3, 5))
    axes = [np.arange(n) * step + (step - 1) / 2 for n in sub.shape]   # block centres
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return dict(x=x.ravel(), y=y.ravel(), z=z.ravel(), value=sub.ravel())


def add_scene(fig, col, labels, brain, volume, show_legend):
    """One 3D panel: optional faint T1gd volume, faint brain shell, solid tumor surfaces."""
    if volume is not None:
        fig.add_trace(go.Volume(
            **volume, isomin=0.15, isomax=1.0, opacity=VOLUME_OPACITY, surface_count=14,
            colorscale="gray", showscale=False, opacityscale=[[0, 0], [1, 1]],
            caps=dict(x_show=False, y_show=False, z_show=False), hoverinfo="skip"),
            row=1, col=col)

    if brain is not None:
        v, f = brain
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=BRAIN_COLOR, opacity=BRAIN_OPACITY, name="brain (faint)",
            legendgroup="brain", showlegend=show_legend, hoverinfo="skip"),
            row=1, col=col)

    for c in CLASS_IDS:
        m = mesh_of(labels == c)
        if m is None:
            continue
        v, f = m
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=COLORS[c], opacity=OPACITY[c], name=CLASS_NAMES[c],
            legendgroup=CLASS_NAMES[c], showlegend=show_legend), row=1, col=col)


SYNC_JS = """
var gd = document.getElementById('{plot_id}');
function which(ev) {
  for (var k in ev) {
    if (k.indexOf('scene2.camera') === 0) return ['scene2', 'scene'];
    if (k.indexOf('scene.camera') === 0) return ['scene', 'scene2'];
  }
  return null;
}
function follow(ev) {
  var pair = which(ev);
  if (!pair) return;
  var src = pair[0], dst = pair[1];
  var cam = ev[src + '.camera'] || gd._fullLayout[src].camera;
  if (JSON.stringify(cam) === JSON.stringify(gd._fullLayout[dst].camera)) return;
  var upd = {};
  upd[dst + '.camera'] = cam;
  Plotly.relayout(gd, upd);
}
gd.on('plotly_relayout', follow);
gd.on('plotly_relayouting', follow);
"""


def plot_3d(path, img, gt, pred, title, brain="shell"):
    """img, gt, pred: numpy arrays (H, W, D). brain: 'shell', 'volume' or 'none'.
    Writes an interactive HTML page whose two panels (ground truth | prediction)
    always share the same camera."""
    bm = brain_mesh(img) if brain in ("shell", "volume") else None
    vol = volume_data(img) if brain == "volume" else None     # computed once for both panels

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("ground truth", "prediction"))
    add_scene(fig, 1, gt, bm, vol, show_legend=True)
    add_scene(fig, 2, pred, bm, vol, show_legend=False)

    H, W, D = img.shape                                # same axis ranges and camera
    ranges = dict(xaxis=dict(range=[0, H]), yaxis=dict(range=[0, W]),
                  zaxis=dict(range=[0, D]))
    cam = dict(eye=dict(x=1.6, y=-1.6, z=0.8))
    scene = dict(aspectmode="data", camera=cam, **ranges)
    fig.update_layout(title_text=title, scene=scene, scene2=scene,
                      margin=dict(l=0, r=0, t=70, b=0))
    fig.write_html(path, include_plotlyjs="cdn", post_script=SYNC_JS)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on whole volumes and plot them")
    p.add_argument("checkpoint")
    p.add_argument("--subset", default="test", choices=["val", "test"])
    p.add_argument("--limit", type=int, default=None, help="only the first N cases")
    p.add_argument("--mode", default="2d", choices=["2d", "3d", "both"],
                   help="2d: PNG of the slice with the most tumor | "
                        "3d: interactive HTML with 3D surfaces | both")
    p.add_argument("--brain", default="shell", choices=["shell", "volume", "none"],
                   help="3D background: faint brain shell | shell + faint T1gd volume | none")
    p.add_argument("--channel", type=int, default=2, help="0 FLAIR, 1 T1w, 2 T1gd, 3 T2w (2D plots)")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    do_2d = args.mode in ("2d", "both")
    do_3d = args.mode in ("3d", "both")
    if do_3d and not HAVE_3D:
        raise SystemExit("3D mode needs: pip install scikit-image plotly scipy")

    ckpt_path = Path(args.checkpoint)
    tag = f"{ckpt_path.parent.name}-{ckpt_path.stem}-{args.subset}"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if do_2d:
        (out_dir / "plots").mkdir(exist_ok=True)
    if do_3d:
        (out_dir / "plots3d").mkdir(exist_ok=True)

    if args.subset == "test":
        print("TEST SET: report this number once and do not tune anything on it.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNet2D(in_ch=4, out_ch=4, base=32).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = build_val_dataset(DATA_DIR, SPLIT, subset=args.subset, limit=args.limit)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    names = load_split(SPLIT)[args.subset][: len(ds)]
    print(f"checkpoint step {ckpt.get('step')} | {len(ds)} {args.subset} cases | mode {args.mode}")

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            name = names[i]
            stem = name.replace(".nii.gz", "")
            img = batch["image"][0].float()            # (4, H, W, D)
            gt = batch["label"][0, 0].long()           # (H, W, D)
            pred = predict_volume(model, img)          # (H, W, D), values 0..3

            d = dice_per_class(pred, gt)
            if isinstance(d, dict):
                d = list(d.values())
            d = [float(v) for v in d]                  # edema, non_enh, enh
            in_gt = [bool((gt == c).any()) for c in CLASS_IDS]

            k = int((gt > 0).sum(dim=(0, 1)).argmax())  # slice with the most tumor
            rows.append([name, k] + d + in_gt)

            title = (f"{name} | slice {k} | whole-volume Dice: "
                     + ", ".join(f"{CLASS_NAMES[c]} {v:.2f}" for c, v in zip(CLASS_IDS, d)))
            if do_2d:
                plot_case(out_dir / "plots" / f"{stem}_slice{k}.png",
                          img[args.channel, :, :, k].numpy(),
                          gt[:, :, k].numpy(), pred[:, :, k].numpy(), title)
            if do_3d:
                plot_3d(out_dir / "plots3d" / f"{stem}.html",
                        img[2].numpy(), gt.numpy(), pred.numpy(), title,
                        brain=args.brain)              # img[2] = T1gd
            print(f"  {i + 1:3d}/{len(ds)} {name} slice {k} "
                  + " ".join(f"{v:.3f}" for v in d))

    with open(out_dir / "per_case.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "slice"] + [CLASS_NAMES[c] for c in CLASS_IDS]
                   + [f"{CLASS_NAMES[c]}_in_gt" for c in CLASS_IDS])
        w.writerows(rows)

    scores = np.array([r[2:5] for r in rows])
    exists = np.array([r[5:8] for r in rows])
    print("\nmean Dice over ALL cases:")
    print("  ", {CLASS_NAMES[c]: round(float(scores[:, j].mean()), 3)
                 for j, c in enumerate(CLASS_IDS)},
          "| mean", round(float(scores.mean()), 3))
    print("mean Dice over cases where the class EXISTS in the ground truth:")
    for j, c in enumerate(CLASS_IDS):
        v = scores[exists[:, j], j]
        print(f"   {CLASS_NAMES[c]:8s} {v.mean():.3f}  (n={len(v)}/{len(rows)})"
              if len(v) else f"   {CLASS_NAMES[c]:8s} -")
    print(f"\nsaved in {out_dir}: per_case.csv"
          + (", plots/*.png" if do_2d else "") + (", plots3d/*.html" if do_3d else ""))


if __name__ == "__main__":
    main()