"""
Plaintext inference with PersonNet, the near-person detector -- the reference for what the
FHE pipeline must reproduce.

  uv run inference.py photo.jpg

Uses ../training/runs/baseline/best.pt, as written by train.py.

Preprocessing is the same code the training cache was built with: ../preprocess.py's
centre_square() (RGB, CENTRE-CROPPED to a square of side min(width, height), resized to IMG_SIZE)
followed by to_input() (ImageNet mean/std normalisation), chained by ../preprocess.py's
preprocess(); use it for anything that feeds the model, encrypted or not.

Prints the predicted label (1 = near person), p(near person) and the two raw logits.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # model/

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from preprocess import preprocess  # noqa: E402
from training.personnet import load  # noqa: E402


@torch.no_grad()
def predict(net, x: torch.Tensor):
    """x: (N, 3, IMG_SIZE, IMG_SIZE) normalised. Returns (labels (N,), p_near (N,), logits (N, 2))."""
    logits = net(x)
    return logits.argmax(1), torch.softmax(logits, dim=1)[:, 1], logits


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} IMAGE")
    net = load()
    labels, p_near, logits = predict(net, preprocess(Image.open(sys.argv[1])))
    print(f"{sys.argv[1]}: {'near person' if labels[0] == 1 else 'not near'}  "
          f"p(near)={p_near[0]:.3f}  logits=({logits[0, 0]:.3f}, {logits[0, 1]:.3f})")
