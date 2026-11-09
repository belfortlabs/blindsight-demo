"""
Wake Vision data access, shared by prepare_dataset.py, by ../model/training/ and by the Belfort ML
example.

Caches written by prepare_dataset.py: data_cache/full/ (--full; train_quality, validation, test),
the training cache, and data_cache/quick/ (validation and test, one parquet file each), what the
samples are drawn from. One set of files per split:
    <split>_x.npy          uint8, (N, 3, S, S)   centre-cropped to a square, resized -- memory-mapped
    <split>_y.npy          int8,  Wake Vision person label 1 / 0, -1 = unlabelled
    <split>_flags.npy      int8,  (N, 9) Wake Vision attribute flags (lighting etc.), -1 = unlabelled
    <split>_wh.npy         int32, (N, 2) original width, height (the crop geometry)
    <split>_filenames.npy  str,   original Flickr file name (join key for Open Images boxes)
    <split>_meta.json      {"N", "written", "S", "flags"}

Added from the Open Images bounding boxes, for each split:
    <split>_task.npy       int8,  the task label: 1 near person, 0 not (nobody / far),
                                  -1 dropped (medium distance, person cropped out, person without a box)
    <split>_dist.npy       int8,  why: 0 near, 1 medium, 2 far, 3 person cropped out, 4 no box,
                                  -1 no person

The preprocessing, IMG_SIZE, centre_square() and to_input(), is the model's input contract and
lives in ../model/preprocess.py; it is re-exported here. prepare_dataset.py builds the caches with
centre_square().

Also provides CLASS_NAMES and DIST_NAMES (what the task labels mean), cache_dir() (the training
cache), MemmapDS, and load_samples() for the two sample sets, samples/test.npz and
samples/calibration.npz.
Running a model over a split (device choice, scoring, the accuracy breakdown) is
../model/training/common.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))   # preprocess.py, the model's input contract
from preprocess import IMG_SIZE, centre_square, to_input  # noqa: E402,F401  (re-exported)

# macOS: the default 'file_descriptor' sharing strategy burns one fd per tensor passed
# from a DataLoader worker; with the default ulimit of 256 that fails after one epoch.
torch.multiprocessing.set_sharing_strategy("file_system")

HF_NAME = "Harvard-Edge/Wake-Vision"
SPLITS = ("train_quality", "validation", "test")
FLAGS = ("near", "medium_distance", "far",
         "bright", "normal_lighting", "dark",
         "person_depiction", "non-person_depiction", "body_part")
LIGHTING = ("bright", "normal_lighting", "dark")        # the only Wake Vision flags still used (../model/training/common.py)
DIST_NAMES = ("near", "medium", "far", "cropped out", "no box")
CLASS_NAMES = ("not near", "near person")

DATA_CACHE = Path(__file__).resolve().parent / "data_cache"
QUICK_CACHE = DATA_CACHE / "quick"                      # the two parquet files the samples come from
SAMPLES = Path(__file__).resolve().parent / "samples"   # test.npz, calibration.npz and their .jpg files


def cache_dir() -> Path:
    """The training cache: all three splits (prepare_dataset.py --full)."""
    return DATA_CACHE / "full"


def load_samples(name: str) -> tuple[np.ndarray, np.ndarray]:
    """samples/<name>.npz ("test" or "calibration") as (X uint8 (N, 3, S, S), y task labels (N,));
    exits with a hint if it is missing or was drawn at another IMG_SIZE."""
    path = SAMPLES / f"{name}.npz"
    if not path.exists():
        sys.exit(f"{path} is missing: run `uv run prepare_dataset.py` in {SAMPLES.parent}")
    samples = np.load(path)
    if int(samples["img_size"]) != IMG_SIZE:
        sys.exit(f"{path} was drawn at {samples['img_size']}px, the model takes {IMG_SIZE}px: "
                 f"run `uv run prepare_dataset.py` in {SAMPLES.parent} again")
    return samples["X"], samples["y"]


def check_size(cache: Path, split: str) -> None:
    """Exits if the cached split was not fetched at IMG_SIZE."""
    size = json.load(open(cache / f"{split}_meta.json"))["S"]
    if size != IMG_SIZE:
        sys.exit(f"{cache} holds {size}x{size} images, IMG_SIZE is {IMG_SIZE}: move or delete {cache} "
                 f"and run prepare_dataset.py again")


class MemmapDS(Dataset):
    """One cached split with its task labels. Items: (uint8 CHW, label, index). Dropped rows
    (task == -1) are skipped."""

    def __init__(self, cache: str | Path, split: str, train: bool = False, shift: int = 0):
        cache = Path(cache)
        if not (cache / f"{split}_task.npy").exists():
            raise FileNotFoundError(f"{cache / (split + '_task.npy')} missing -- run `uv run prepare_dataset.py --full` in ../dataset first")
        check_size(cache, split)
        meta = json.load(open(cache / f"{split}_meta.json"))
        self.x_path = cache / f"{split}_x.npy"
        self._X = None                                   # memmap opened lazily, per process
        T = np.load(cache / f"{split}_task.npy")
        self.rows = np.nonzero(T >= 0)[0]
        self.Y = T[self.rows]
        self.DIST = np.load(cache / f"{split}_dist.npy")[self.rows]
        self.FL = np.load(cache / f"{split}_flags.npy")[self.rows]
        self.S, self.train, self.shift = meta["S"], train, shift

    @property
    def X(self):
        # A np.memmap pickles by reading the entire array, so it must not be an attribute when
        # DataLoader workers are spawned (macOS). Each worker opens its own map.
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode="r")
        return self._X

    def __getstate__(self):
        d = self.__dict__.copy(); d["_X"] = None; return d

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.X[self.rows[i]]))
        if self.train:
            if torch.rand(()) < 0.5:
                x = x.flip(-1)                               # horizontal flip
            if self.shift:                                   # random translate by up to +-shift px
                p = self.shift
                dx, dy = (int(v) for v in torch.randint(-p, p + 1, (2,)))
                x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
        return x, int(self.Y[i]), i
