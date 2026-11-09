"""The trained model in ../model/ and its sample images in ../dataset/, as this example
uses them:

    ../model/training/personnet.py   load(weights) -> the trained nn.Module, and WEIGHTS
    ../model/preprocess.py           IMG_SIZE, preprocess(), to_input()
    ../dataset/wake_vision.py        CLASS_NAMES, load_samples(): samples/test.npz, samples/calibration.npz

Importing this module puts ../model/ and ../dataset/ on sys.path the way their own scripts
do. load_model() loads the trained network from there, not copied here, so the deployed model is
the trained one; test_samples() and calibration_samples() read the two sample sets that
prepare_dataset.py drew, and verdict() names the class the decrypted scores predict. It needs the model's
dependencies (numpy, torch, PIL) but not belfort-ml.
"""

import sys
from pathlib import Path

import torch

# The model and the dataset sit beside this directory, in this repository.
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model"
DATASET_DIR = HERE.parent / "dataset"
MODEL = "personnet"  # the compiled model's name on the gateway

sys.path.insert(0, str(MODEL_DIR))    # preprocess, training.personnet
sys.path.insert(0, str(DATASET_DIR))  # wake_vision

from preprocess import IMG_SIZE, preprocess, to_input
from training import personnet as _personnet
from wake_vision import CLASS_NAMES, load_samples

# The trained checkpoint, as the model's train.py writes it.
WEIGHTS = Path(_personnet.WEIGHTS)


def load_model():
    """The trained network, on the CPU and in eval mode."""
    if not WEIGHTS.is_file():
        sys.exit(f"{WEIGHTS} is missing: train the model first (../model/README.md)")
    return _personnet.load(WEIGHTS)


def verdict(scores) -> tuple[int, str]:
    """The predicted class for the decrypted scores: its index and its name in CLASS_NAMES."""
    scores = list(scores)
    predicted = scores.index(max(scores))
    return predicted, CLASS_NAMES[predicted]


def samples(name: str) -> tuple[torch.Tensor, list[int]]:
    """One sample set, normalised as the model expects, (N, 3, S, S), and its labels:
    1 near person, 0 not."""
    X, y = load_samples(name)
    return to_input(torch.from_numpy(X), torch.device("cpu")), y.tolist()


def test_samples() -> tuple[torch.Tensor, list[int]]:
    """samples/test.npz, from Wake Vision's test split: client.py --samples."""
    return samples("test")


def calibration_samples() -> tuple[torch.Tensor, list[int]]:
    """samples/calibration.npz, from Wake Vision's validation split: what compile calibrates on."""
    return samples("calibration")


__all__ = [
    "CLASS_NAMES",
    "DATASET_DIR",
    "HERE",
    "IMG_SIZE",
    "MODEL",
    "MODEL_DIR",
    "WEIGHTS",
    "calibration_samples",
    "load_model",
    "preprocess",
    "test_samples",
    "to_input",
    "verdict",
]
