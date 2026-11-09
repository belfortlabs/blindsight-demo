"""
Train PersonNet as a near-person detector on the full Wake Vision training split
(centre-cropped images, box-derived task labels; ../../dataset/prepare_dataset.py --full).

  uv run train.py                     # 15 epochs -> runs/baseline/   (IMG_SIZE: ../preprocess.py)

Task label: 1 = person near the camera, 0 = nobody / far person; medium-distance, cropped-out
and un-boxed persons are not in the data. Best epoch by validation accuracy.

Writes <runs>/best.pt (plain state_dict, best validation accuracy -- what personnet.load() reads),
<runs>/last.pt (full checkpoint: model + optimiser + schedule + epoch, for --resume) and <runs>/train.json
(the configuration, best validation accuracy and per-epoch numbers; committed as the run's record, not read by code).
Interrupted? Re-run the same command with --resume to continue from the last completed epoch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dataset"))   # wake_vision importable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wake_vision import IMG_SIZE, MemmapDS, cache_dir, to_input
from common import accuracy_breakdown, pick_device, predict_correct, print_breakdown
from personnet import CHANNELS, RUN, STRIDES, PersonNet

WEIGHT_DECAY, LABEL_SMOOTHING, SEED, LOG_EVERY = 0.05, 0.05, 0, 250
SHIFT_FRAC = 1 / 16                              # random translation augmentation (+ horizontal flip),
SHIFT_PX = max(1, round(IMG_SIZE * SHIFT_FRAC))  # as a fraction of the image: 4 px at 64x64
WORKERS = max(1, (os.cpu_count() or 4) - 2)      # data loader processes


def main(a):
    device = pick_device()
    torch.manual_seed(SEED)
    cache = cache_dir()
    runs = Path("runs") / RUN; runs.mkdir(parents=True, exist_ok=True)
    print(f"[train] {RUN}: {IMG_SIZE}x{IMG_SIZE}, strides {STRIDES}, channels {CHANNELS}", flush=True)
    print(f"[train] device={device} cache={cache} runs={runs} epochs={a.epochs} batch={a.batch}", flush=True)

    tr = MemmapDS(cache, "train_quality", train=True, shift=SHIFT_PX)
    va = MemmapDS(cache, "validation")
    print(f"[train] train_quality: {len(tr):,} rows ({int((tr.Y == 1).sum()):,} near / {int((tr.Y == 0).sum()):,} not)  "
          f"validation: {len(va):,} rows ({int((va.Y == 1).sum()):,} / {int((va.Y == 0).sum()):,})", flush=True)
    pin = device.type == "cuda"
    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=WORKERS, pin_memory=pin,
                       drop_last=True, persistent_workers=True)
    dl_va = DataLoader(va, batch_size=512, num_workers=min(WORKERS, 2), pin_memory=pin,
                       persistent_workers=True)            # keep val workers alive across epochs

    model = PersonNet(num_classes=2, channels=CHANNELS, strides=STRIDES).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters()):,}  steps/epoch: {len(dl_tr):,}", flush=True)
    n1, n0 = float((tr.Y == 1).sum()), float((tr.Y == 0).sum())       # class-weighted loss: label 0 outnumbers 1
    cw = torch.tensor([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=WEIGHT_DECAY)
    steps, warm = a.epochs * len(dl_tr), len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    best, step, log, start_ep = 0.0, 0, [], 0
    if a.resume:
        ck = torch.load(runs / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, step, log, start_ep = ck["best"], ck["step"], ck["log"], ck["epoch"]
        print(f"[train] resumed from {runs / 'last.pt'}: {start_ep} epoch(s) done, best val {best * 100:.2f}%", flush=True)
    for ep in range(start_ep, a.epochs):
        model.train(); t0 = time.time(); loss_sum = n = correct = 0
        for xb, yb, _ in dl_tr:
            x, y = to_input(xb, device), yb.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y, weight=cw, label_smoothing=LABEL_SMOOTHING)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            loss_sum += loss.item() * y.numel(); n += y.numel(); correct += (out.argmax(1) == y).sum().item()
            if step % LOG_EVERY == 0:
                done = step - ep * len(dl_tr); rate = n / (time.time() - t0)
                print(f"  ep {ep + 1}/{a.epochs}  {done / len(dl_tr) * 100:5.1f}%  loss {loss_sum / n:.4f}  "
                      f"acc {correct / n * 100:.2f}%  lr {sched.get_last_lr()[0]:.2e}  {rate:,.0f} img/s  "
                      f"epoch eta {(len(dl_tr) - done) * a.batch / max(rate, 1) / 60:.1f} min", flush=True)
        va_acc = predict_correct(model, dl_va, device).float().mean().item()
        improved = va_acc > best
        rec = dict(epoch=ep + 1, train_loss=loss_sum / n, train_acc=correct / n, val_acc=va_acc,
                   minutes=(time.time() - t0) / 60)
        log.append(rec)
        print(f"[ep {ep + 1}/{a.epochs}] loss {rec['train_loss']:.4f} train {rec['train_acc'] * 100:.2f}% "
              f"val {va_acc * 100:.2f}%{'  *best*' if improved else ''} ({rec['minutes']:.1f} min, "
              f"total eta {(a.epochs - ep - 1) * rec['minutes']:.0f} min)", flush=True)
        if va_acc > best:
            best = va_acc; torch.save(model.state_dict(), runs / "best.pt")   # plain state_dict for load()
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=ep + 1, step=step, best=best, log=log), runs / "last.pt")
        json.dump(dict(run=RUN, img_size=IMG_SIZE, channels=CHANNELS, strides=STRIDES,
                       args=vars(a), best_val=best, epochs=log), open(runs / "train.json", "w"), indent=2)
    model.load_state_dict(torch.load(runs / "best.pt", map_location=device))
    print_breakdown(accuracy_breakdown(predict_correct(model, dl_va, device), va),
                    "\n[train] best checkpoint on validation:")
    print(f"[train] done. best val {best * 100:.2f}% -> {runs / 'best.pt'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3, help="peak learning rate (1 warm-up epoch, then cosine decay)")
    p.add_argument("--resume", action="store_true", help="continue from <runs>/last.pt")
    main(p.parse_args())
