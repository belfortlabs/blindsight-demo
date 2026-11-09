"""
PersonNet's input contract: what every image goes through before the model sees it.

Preprocessing lives here and ONLY here: centre_square() (PIL image -> uint8 CHW: centre crop to a
square, resize to IMG_SIZE), to_input() (uint8 batch -> ImageNet-normalised float), and
preprocess(), the two chained for one image.
../dataset/prepare_dataset.py builds the dataset caches with centre_square(); inference.py and the
Belfort ML example apply centre_square() + to_input() to a new image, so whatever is fed to the
model, in training or in the field, has been through the same code.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMG_SIZE = 64                 # THE input resolution: cache, model and inference all use this

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def centre_square(img: Image.Image) -> torch.Tensor:
    """PIL image -> uint8 CHW tensor (3, IMG_SIZE, IMG_SIZE): RGB, centre-cropped to a square
    (side = min(width, height)), resized to IMG_SIZE. The one definition of what the model sees;
    ../dataset/prepare_dataset.py applies the same crop geometry to the bounding boxes."""
    img = img.convert("RGB")
    W, H = img.size
    s = min(W, H); left, top = (W - s) // 2, (H - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((IMG_SIZE, IMG_SIZE))
    return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()


def to_input(xb: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 NCHW batch (as centre_square() makes it) -> ImageNet-normalised float on the device."""
    x = xb.to(device, non_blocking=True).float().div_(255.0)
    return (x - _MEAN.to(device)) / _STD.to(device)


def preprocess(img: Image.Image) -> torch.Tensor:
    """PIL image -> normalised float tensor (1, 3, IMG_SIZE, IMG_SIZE) on the CPU: exactly the
    training cache's path. Use it for anything that feeds the model, encrypted or not."""
    return to_input(centre_square(img).unsqueeze(0), torch.device("cpu"))
