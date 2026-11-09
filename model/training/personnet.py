"""
PersonNet -- the near-person detector, kept to layers an FHE compiler handles well.
Label 1 = a person close to the camera; label 0 = nobody, or a person far away.

Derived from MobileNetV1 with the expensive parts removed:
  * dense 3x3 Conv2d instead of depthwise-separable blocks
  * plain ReLU (not ReLU6)
  * stride-2 convs for downsampling, no MaxPool; one final AvgPool (linear)
  * 4 non-linear layers instead of 27
  * no Dropout

Layers: Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Linear.

The input is a 64x64 image (IMG_SIZE in ../preprocess.py). The weights live in
training/runs/baseline/, with best.pt, train.json (the training record) and evaluate.json (the test
record) beside each other. Strides are (2, 2, 2, 1): the 4th conv keeps its input resolution, so
the final feature map is 64 x 8 x 8.
Hand count: 60,642 params, ~5.2M MACs, ~32.8k ReLU evaluations, 4 non-linear layers, largest
activation 16x32x32 = 16,384 values, receptive field 31 px.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # model dir -> preprocess importable

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from preprocess import IMG_SIZE  # noqa: E402,F401   (the one resolution constant)
CHANNELS = (16, 32, 64, 64)
STRIDES = (2, 2, 2, 1)
RUN = "baseline"                                                               # the one run directory
WEIGHTS = Path(__file__).parent / "runs" / RUN / "best.pt"                   # what train.py writes


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class PersonNet(nn.Module):
    """3x3 conv layers with strides STRIDES -> global average pool -> linear."""

    def __init__(self, num_classes: int = 2, channels: Sequence[int] = CHANNELS,
                 strides: Sequence[int] = STRIDES, in_channels: int = 3):
        super().__init__()
        assert len(strides) == len(channels)
        layers, cin = [], in_channels
        for co, st in zip(channels, strides):
            layers.append(ConvBNReLU(cin, co, stride=st))
            cin = co
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(torch.flatten(self.pool(x), 1))


def load(weights: str | Path = WEIGHTS) -> PersonNet:
    """The trained classifier, on the CPU and in eval mode.

    Reads the state_dict train.py wrote (training/runs/baseline/best.pt)
    unless given another path. Input: float tensor (N, 3, IMG_SIZE, IMG_SIZE), ImageNet-normalised
    (see preprocess.to_input). Output: logits (N, 2), index 1 = NEAR person.
    """
    net = PersonNet()
    net.load_state_dict(torch.load(Path(weights), map_location="cpu"))
    return net.eval()
