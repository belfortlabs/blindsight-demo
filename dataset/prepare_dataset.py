"""
Prepare Wake Vision for PersonNet: the sample frames for the Belfort ML example, and with --full
the training cache.

  uv run prepare_dataset.py               # the samples: ~1.1 GB of download, a few minutes
  uv run prepare_dataset.py --full        # also the training cache: ~45 GB streamed, 30-60 min

Default (quick): downloads one parquet file of Wake Vision's validation split and one of its test
split (~500 MB each, kept in data_cache/quick/), centre-crops their images, labels them from the
Open Images boxes and draws two balanced sets, half near person and half not:

  samples/test.npz          64 frames from the test split: client.py --samples sends them. The
                            test split is never trained on.
  samples/calibration.npz   512 frames from the validation split: provider.py compile calibrates
                            the encrypted model on them.

Each set is also written as .jpg files, in samples/test/ and samples/calibration/, enlarged 4x
(--upscale) to look at or to feed to ../model/inference/inference.py. The quick files are always
the same two, so the samples are the same on every machine for the same --seed.

--full: also fetches all three splits (train_quality, validation, test) into data_cache/full/ and
labels them, for ../model/training/. The dataset is streamed: parquet shards are fetched, decoded
and discarded, so only the 64x64 arrays are kept (~16 GB). Re-running skips the splits that are
complete; if a run is interrupted mid-split, delete that split's files and run again. Labelling
all three splits downloads ~3 GB of Open Images files (data_cache/open_images/), once.

Every image is centre-cropped to a square of side min(width, height) and resized to IMG_SIZE
(../model/preprocess.py's centre_square(), the model's input contract). The label comes from the
Open Images boxes, intersected with that same crop. The largest visible person box, as a fraction
of the square, sizes the image:

    side = sqrt(visible box area / crop area)
    near   side > 0.38        -> task 1
    medium 0.16 .. 0.38       -> task -1  (dropped: the ambiguous band)
    far    side < 0.16        -> task 0
    person entirely outside the crop -> task -1 (dropped: we cannot be sure nobody else is in the crop)
    person image without any box     -> task -1 (dropped)
    no-person image                  -> task 0

The two cut-offs were fitted once on Wake Vision's human-labelled validation split (83%
agreement with the human near / medium / far); they are constants here. Wake Vision's own
human distance labels are not used anywhere in this pipeline. "Person box" = classes Person, Man,
Woman, Boy, Girl, Human face, Human head, Human body, excluding boxes flagged IsDepiction or
IsGroupOf. Wake Vision's validation and test images come from the Open Images subsets of the same
name, so the quick mode only downloads their image lists and boxes (~160 MB).

| Flag | Default | What |
|---|---|---|
| `--full` | off | Also build the training cache, all three splits, in data_cache/full/ |
| `--test` | `64` | Frames in samples/test.npz, half per label |
| `--calibration` | `512` | Frames in samples/calibration.npz, half per label; fewer make wrong results under encryption more likely |
| `--seed` | `0` | Which rows the two sets are drawn from |
| `--upscale` | `4` | Enlargement of the .jpg files |
| `--workers` | cores - 2 | --full only: decode processes. Each holds a whole shard in memory: use 4 or fewer with little RAM, 2 on a slow link |
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wake_vision import (CLASS_NAMES, DATA_CACHE, DIST_NAMES, FLAGS, HF_NAME, IMG_SIZE, QUICK_CACHE,  # noqa: E402
                         SAMPLES, SPLITS, cache_dir, centre_square, check_size)

# The one parquet file per split the quick mode uses, and which set each split gives.
QUICK_FILES = {"validation": "data/validation-00000-of-00011.parquet",
               "test": "data/test-00000-of-00032.parquet"}
SAMPLE_SETS = {"test": "test", "calibration": "validation"}  # set -> split it is drawn from

OI_DIR = DATA_CACHE / "open_images"
OI_FILES = {
    "classes": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "train_images": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation_images": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test_images": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    "train_bbox": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation_bbox": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test_bbox": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
PERSON_CLASSES = {"Person", "Man", "Woman", "Boy", "Girl", "Human face", "Human head", "Human body"}
NEAR_ABOVE, FAR_BELOW = 0.38, 0.16        # on sqrt(visible box area / crop area); fitted on human labels once
NEAR, MED, FAR, CROPPED, NOBOX, NOPERSON = 0, 1, 2, 3, 4, -1


# ----------------------------------------------------------------------------- one split's cache
def save_split(out: Path, split: str, Y, FL, names, WH, written: int, source: str) -> None:
    """Everything of a cached split but its images (<split>_x.npy, written as it is filled)."""
    np.save(out / f"{split}_y.npy", Y)
    np.save(out / f"{split}_flags.npy", FL)
    np.save(out / f"{split}_filenames.npy", np.array(names))
    np.save(out / f"{split}_wh.npy", WH)
    json.dump({"N": len(Y), "written": written, "S": IMG_SIZE, "flags": list(FLAGS), "source": source},
              open(out / f"{split}_meta.json", "w"))


def quick_split(split: str) -> None:
    """Download QUICK_FILES[split] into data_cache/quick/ and cache its images, centre-cropped."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from PIL import Image

    out = QUICK_CACHE
    if (out / f"{split}_meta.json").exists():
        check_size(out, split)
        print(f"[quick] {split}: cached", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    print(f"[quick] {split}: downloading {QUICK_FILES[split]} (~500 MB) ...", flush=True)
    path = hf_hub_download(HF_NAME, QUICK_FILES[split], repo_type="dataset", local_dir=out)
    t0 = time.time()
    table = pq.ParquetFile(path)
    N = table.metadata.num_rows
    X = np.lib.format.open_memmap(out / f"{split}_x.npy", mode="w+", dtype=np.uint8, shape=(N, 3, IMG_SIZE, IMG_SIZE))
    Y = np.full(N, -1, dtype=np.int8)
    FL = np.full((N, len(FLAGS)), -1, dtype=np.int8)
    names = [""] * N
    WH = np.zeros((N, 2), dtype=np.int32)
    i = 0
    for batch in table.iter_batches(batch_size=256, columns=["image", "person", "filename", *FLAGS]):
        for row in batch.to_pylist():
            img = Image.open(io.BytesIO(row["image"]["bytes"]))
            WH[i] = img.size                              # original size: the crop geometry, for the boxes
            X[i] = centre_square(img).numpy()             # the model's preprocessing (../model/preprocess.py)
            Y[i] = row["person"] if row["person"] in (0, 1) else -1
            FL[i] = [row[f] if row[f] is not None else -1 for f in FLAGS]
            names[i] = row["filename"] or ""
            i += 1
    X.flush(); del X
    save_split(out, split, Y, FL, names, WH, i, QUICK_FILES[split])
    print(f"[quick] {split}: {i:,} images cropped to {IMG_SIZE}x{IMG_SIZE} in {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- --full: the training cache
def full_split(split: str, out: Path, workers: int) -> None:
    """Stream one whole split from Hugging Face into data_cache/full/, centre-cropped."""
    import torch
    from datasets import load_dataset, load_dataset_builder
    from torch.utils.data import DataLoader, IterableDataset

    class StreamDS(IterableDataset):
        """Yields (uint8 CHW, label, flags, filename, (W, H)). Shards are split across DataLoader
        workers by `datasets`, so the order of the rows depends on the number of workers."""

        def __init__(self, hf_iter):
            self.ds = hf_iter

        def __iter__(self):
            for item in self.ds:
                img = item["image"]
                W, H = img.size
                y = item["person"]
                flags = torch.tensor([int(item.get(f, -1)) for f in FLAGS], dtype=torch.int8)
                yield (centre_square(img), (int(y) if y in (0, 1) else -1), flags,
                       str(item.get("filename", "")), torch.tensor([W, H]))

    N = load_dataset_builder(HF_NAME).info.splits[split].num_examples
    hf = load_dataset(HF_NAME, split=split, streaming=True)
    print(f"[full] {split}: {N:,} images -> {IMG_SIZE}x{IMG_SIZE} uint8 ({N * 3 * IMG_SIZE ** 2 / 1e9:.2f} GB), "
          f"{workers} workers, streaming", flush=True)
    X = np.lib.format.open_memmap(out / f"{split}_x.npy", mode="w+", dtype=np.uint8, shape=(N, 3, IMG_SIZE, IMG_SIZE))
    Y = np.full(N, -1, dtype=np.int8)               # -1 = not written / unlabelled -> skipped by MemmapDS
    FL = np.full((N, len(FLAGS)), -1, dtype=np.int8)
    names = [""] * N                                 # Flickr file names: join key for Open Images boxes
    WH = np.zeros((N, 2), dtype=np.int32)            # original width, height (crop geometry)
    dl = DataLoader(StreamDS(hf), batch_size=256, num_workers=workers)

    i, t0 = 0, time.time()
    print("  waiting for the first batch (each worker fetches a ~500 MB shard first) ...", flush=True)
    for b, (xb, yb, fb, nb, whb) in enumerate(dl):
        if b == 0:
            print(f"  first batch after {time.time() - t0:.0f} s", flush=True)
        n = min(xb.shape[0], N - i)
        if n <= 0:
            break
        X[i:i + n] = xb[:n].numpy(); Y[i:i + n] = yb[:n].numpy(); FL[i:i + n] = fb[:n].numpy()
        names[i:i + n] = list(nb[:n]); WH[i:i + n] = whb[:n].numpy()
        i += n
        if b % 25 == 0 and b:
            rate = i / (time.time() - t0)
            print(f"  {i:,}/{N:,}  {rate:,.0f} img/s  eta {(N - i) / max(rate, 1) / 60:.1f} min", flush=True)
    X.flush(); del X
    save_split(out, split, Y, FL, names, WH, i, f"{HF_NAME} {split}, streamed")
    print(f"[full] {split}: wrote {i:,} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


# ----------------------------------------------------------------------------- Open Images downloads
def download(name: str) -> Path:
    url = OI_FILES[name]; dst = OI_DIR / url.rsplit("/", 1)[1]
    if dst.exists():
        return dst
    OI_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part"); t0 = time.time()

    def hook(blocks, bs, total):
        done = blocks * bs
        if blocks % 2000 == 0:
            print(f"\r  {dst.name}: {done / 1e6:,.0f} MB" + (f" / {total / 1e6:,.0f} MB" if total > 0 else "")
                  + f"  ({done / 1e6 / max(time.time() - t0, 1e-6):.1f} MB/s)", end="", flush=True)
    print(f"[oi] downloading {dst.name} ...", flush=True)
    urllib.request.urlretrieve(url, tmp, hook)
    tmp.rename(dst); print(f"\r  {dst.name}: done in {(time.time() - t0) / 60:.1f} min" + " " * 30, flush=True)
    return dst


# ----------------------------------------------------------------------------- Open Images boxes
def person_label_ids() -> set[str]:
    with open(download("classes"), newline="") as f:
        found = {name: lid for lid, name in csv.reader(f) if name in PERSON_CLASSES}
    missing = PERSON_CLASSES - set(found)
    if missing:
        print(f"[oi] WARNING: classes not in the Open Images list, ignored: {sorted(missing)}", flush=True)
    return set(found.values())


def filename_to_image_id(wanted: set[str]) -> dict[str, tuple[str, str]]:
    """Flickr file name -> (Open Images ImageID, Open Images subset), cached in image_ids.json;
    rescans only for new names. The small validation and test lists are scanned first, so
    labelling Wake Vision's validation or test split need not download the train list."""
    cached = OI_DIR / "image_ids.json"
    out = {n: tuple(v) for n, v in json.load(open(cached)).items()} if cached.exists() else {}
    if wanted <= set(out):
        return {n: out[n] for n in wanted}
    for subset in ("validation", "test", "train"):
        path = download(f"{subset}_images"); t0 = time.time(); n = 0
        with open(path, newline="") as f:
            r = csv.reader(f); hdr = next(r); c_id, c_url = hdr.index("ImageID"), hdr.index("OriginalURL")
            for row in r:
                n += 1
                name = row[c_url].rsplit("/", 1)[-1]
                if name in wanted and name not in out:
                    out[name] = (row[c_id], subset)
        print(f"[oi] {path.name}: {n:,} rows scanned, {len(wanted & set(out)):,} of {len(wanted):,} names found "
              f"({time.time() - t0:.0f} s)", flush=True)
        if wanted <= set(out):
            break
    json.dump(out, open(cached, "w"))
    return {n: out[n] for n in wanted if n in out}


def person_boxes(person_ids: set[str], subsets: set[str]) -> dict[str, np.ndarray]:
    """ImageID -> (k, 4) array of person boxes [xmin, xmax, ymin, ymax], normalised to the image,
    for the images of the given Open Images subsets. Each subset's bbox file is streamed once and
    its person boxes cached as person_boxes_<subset>.npz."""
    out = defaultdict(list)
    for subset in sorted(subsets):
        for i, b in subset_person_boxes(person_ids, subset):
            out[i].append(b)
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def subset_person_boxes(person_ids: set[str], subset: str):
    """(ImageID, box) pairs of one Open Images subset's person boxes, from its cache or its bbox file."""
    cached = OI_DIR / f"person_boxes_{subset}.npz"
    if cached.exists():
        z = np.load(cached)
        print(f"[oi] {subset} person boxes: {len(z['ids']):,} (from {cached.name})", flush=True)
        return zip(z["ids"].tolist(), z["boxes"])
    ids, boxes = [], []
    path = download(f"{subset}_bbox"); t0 = time.time(); n = 0
    with open(path, newline="") as f:
        r = csv.reader(f); hdr = next(r)
        c = {k: hdr.index(k) for k in ("ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction")}
        for row in r:
            n += 1
            if row[c["LabelName"]] not in person_ids or row[c["IsDepiction"]] == "1" or row[c["IsGroupOf"]] == "1":
                continue
            ids.append(row[c["ImageID"]])
            boxes.append((float(row[c["XMin"]]), float(row[c["XMax"]]), float(row[c["YMin"]]), float(row[c["YMax"]])))
            if n % 5_000_000 == 0:
                print(f"  {path.name}: {n:,} rows, {len(ids):,} person boxes", flush=True)
    print(f"[oi] {path.name}: {n:,} rows, {len(ids):,} person boxes ({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez(cached, ids=np.array(ids), boxes=np.array(boxes, dtype=np.float32))
    print(f"[oi] {subset} person boxes: {len(ids):,} -> {cached.name}", flush=True)
    return subset_person_boxes(person_ids, subset)


# ----------------------------------------------------------------------------- the crop + rule
def visible_side(boxes: np.ndarray, W: int, H: int) -> float:
    """Largest person box after intersecting with the centre square, as sqrt(area fraction of the
    square). 0.0 if no box overlaps the square."""
    s = min(W, H)
    x0, x1 = (W - s) / 2 / W, (W + s) / 2 / W          # crop window, normalised like the boxes
    y0, y1 = (H - s) / 2 / H, (H + s) / 2 / H
    ix0 = np.maximum(boxes[:, 0], x0); ix1 = np.minimum(boxes[:, 1], x1)
    iy0 = np.maximum(boxes[:, 2], y0); iy1 = np.minimum(boxes[:, 3], y1)
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    crop_area = (x1 - x0) * (y1 - y0)
    return float(np.sqrt(inter.max() / crop_area)) if len(boxes) else 0.0


def label_split(cache: Path, split: str, name2id: dict[str, tuple[str, str]], boxes: dict[str, np.ndarray]):
    Y = np.load(cache / f"{split}_y.npy"); names = np.load(cache / f"{split}_filenames.npy")
    WH = np.load(cache / f"{split}_wh.npy")
    task = np.full(len(Y), -1, dtype=np.int8); dist = np.full(len(Y), NOPERSON, dtype=np.int8)
    for i in range(len(Y)):
        if Y[i] == 0:
            task[i] = 0; continue
        if Y[i] != 1:
            continue
        b = boxes.get(name2id.get(str(names[i]), ("", ""))[0])
        if b is None:
            dist[i] = NOBOX; continue                                   # task stays -1
        side = visible_side(b, int(WH[i, 0]), int(WH[i, 1]))
        if side == 0.0:
            dist[i] = CROPPED                                            # task stays -1
        elif side > NEAR_ABOVE:
            dist[i], task[i] = NEAR, 1
        elif side < FAR_BELOW:
            dist[i], task[i] = FAR, 0
        else:
            dist[i] = MED                                                # task stays -1
    np.save(cache / f"{split}_task.npy", task); np.save(cache / f"{split}_dist.npy", dist)
    c = {n: int((dist == k).sum()) for k, n in ((NEAR, "near"), (MED, "medium"), (FAR, "far"),
                                                (CROPPED, "cropped out"), (NOBOX, "no box"), (NOPERSON, "no person"))}
    print(f"[label] {split}: " + ", ".join(f"{k} {v:,}" for k, v in c.items())
          + f"  ->  task 1: {int((task == 1).sum()):,}  task 0: {int((task == 0).sum()):,}  dropped: {int((task == -1).sum()):,}", flush=True)
    return c


def label(cache: Path, splits) -> None:
    """Task labels for the cached splits: <split>_task.npy, <split>_dist.npy and the counts in
    <split>_meta.json. The Open Images downloads, ids and boxes are cached, so this is quick to redo."""
    wanted = set()
    for split in splits:
        Y = np.load(cache / f"{split}_y.npy"); names = np.load(cache / f"{split}_filenames.npy")
        wanted |= {str(n) for n, y in zip(names, Y) if y == 1}
    print(f"[label] {len(wanted):,} person images need boxes", flush=True)
    name2id = filename_to_image_id(wanted)
    boxes = person_boxes(person_label_ids(), {subset for _, subset in name2id.values()})
    for split in splits:
        counts = label_split(cache, split, name2id, boxes)
        meta_path = cache / f"{split}_meta.json"; meta = json.load(open(meta_path))
        meta["task_labels"] = dict(source="open_images_boxes_on_centre_crop", measure="sqrt(visible_area / crop_area)",
                                   near_above=NEAR_ABOVE, far_below=FAR_BELOW, counts=counts)
        json.dump(meta, open(meta_path, "w"))


# ----------------------------------------------------------------------------- the two sample sets
def bucket(dist: int) -> str:
    """Why a sample has its label, as a file-name word: near, far, no-person..."""
    if dist < 0:
        return "no-person"
    if dist < len(DIST_NAMES):
        return DIST_NAMES[dist].replace(" ", "-")
    return f"dist{dist}"


def draw(name: str, n: int, seed: int, upscale: int) -> None:
    """Draw samples/<name>.npz from the quick cache of its split, balanced on the task label, and
    write the same images as samples/<name>/*.jpg."""
    from PIL import Image

    split = SAMPLE_SETS[name]
    task = np.load(QUICK_CACHE / f"{split}_task.npy")
    rng = np.random.default_rng(seed)
    rows = []
    for c, k in ((1, n - n // 2), (0, n // 2)):
        have = np.nonzero(task == c)[0]
        if len(have) < k:
            sys.exit(f"{QUICK_FILES[split]} has {len(have)} {CLASS_NAMES[c]} images, {name} needs {k}: "
                     f"ask for fewer with --{name}")
        rows.append(rng.choice(have, k, replace=False))
    rows = np.concatenate(rows)
    rows = rows[rng.permutation(len(rows))]          # mixed order: any prefix is balanced
    X = np.array(np.load(QUICK_CACHE / f"{split}_x.npy", mmap_mode="r")[rows])
    y, dist = task[rows].astype(np.int64), np.load(QUICK_CACHE / f"{split}_dist.npy")[rows]

    jpgs = SAMPLES / name
    jpgs.mkdir(parents=True, exist_ok=True)
    for old in jpgs.glob("*.jpg"):
        old.unlink()
    np.savez_compressed(SAMPLES / f"{name}.npz", X=X, y=y, dist=dist, rows=rows, split=split,
                        source=QUICK_FILES[split], img_size=IMG_SIZE)
    width = len(str(len(y) - 1))
    for i in range(len(y)):
        img = Image.fromarray(np.ascontiguousarray(X[i].transpose(1, 2, 0)))
        if upscale > 1:
            img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
        img.save(jpgs / f"{i:0{width}d}_{bucket(int(dist[i]))}_task{int(y[i])}_row{int(rows[i])}.jpg", quality=92)
    far = int(((y == 0) & (dist == FAR)).sum())
    print(f"[samples] {SAMPLES / name}.npz and {name}/*.jpg: {len(y)} {split} images, "
          f"{int(y.sum())} near person, {len(y) - int(y.sum())} not near ({far} of them with a far person)", flush=True)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare Wake Vision: the sample frames, and with --full the training cache.")
    p.add_argument("--full", action="store_true", help="also build the training cache: all three splits in data_cache/full/")
    p.add_argument("--test", type=int, default=64, help="frames in samples/test.npz (default 64)")
    p.add_argument("--calibration", type=int, default=512, help="frames in samples/calibration.npz (default 512)")
    p.add_argument("--seed", type=int, default=0, help="which rows the sample sets are drawn from (default 0)")
    p.add_argument("--upscale", type=int, default=4, help="enlargement of the .jpg files (default 4)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                   help="--full only: decode processes (default: cores - 2; 4 or fewer with little RAM)")
    a = p.parse_args()

    # The samples, in both modes: from the two quick files.
    for split in QUICK_FILES:
        quick_split(split)
    label(QUICK_CACHE, list(QUICK_FILES))
    draw("test", a.test, a.seed, a.upscale)
    draw("calibration", a.calibration, a.seed, a.upscale)

    if a.full:
        old = DATA_CACHE / "wake_vision"
        if old.exists() and not cache_dir().exists():
            sys.exit(f"{old} is the training cache of an earlier version: rename it with\n"
                     f"  mv {old} {cache_dir()}\nand run again, so the splits in it are reused.")
        out = cache_dir()
        out.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            if (out / f"{split}_meta.json").exists():
                check_size(out, split)
                print(f"[full] {split}: cached", flush=True)
            else:
                full_split(split, out, a.workers)
        label(out, list(SPLITS))
        print(f"[full] done: {', '.join(SPLITS)} in {out}", flush=True)
        # The streaming reader can block interpreter shutdown while it drains or closes a
        # partially-read shard. Everything is flushed to disk above, so exit hard.
        sys.stdout.flush()
        os._exit(0)
