# dataset: Wake Vision, labelled for PersonNet

[Wake Vision](https://huggingface.co/datasets/Harvard-Edge/Wake-Vision) (Harvard Edge Computing
Lab, CC BY 4.0) is a person / no-person image dataset built for tiny models: ~1.2M training
images, plus validation (18.6k) and test (55.8k) splits. Its images come from Open Images,
whose human-verified bounding boxes are public. This pipeline uses those boxes as the only
source of the "near person" label.

One script prepares everything, `prepare_dataset.py`, in one of two modes:

- **Default:** the sample frames for the [Belfort ML example](../eml/README.md). It downloads
  one parquet shard of Wake Vision's validation split and one of its test split, ~1.1 GB with the Open
  Images files, a few minutes, and draws two labelled sets from them:
  - `samples/test.npz`, 64 frames from the test split, which training never used:
    `client.py --samples` sends them.
  - `samples/calibration.npz`, 512 frames from the validation split: `provider.py compile`
    calibrates the encrypted model on them.
- **`--full`:** also the training cache, all three splits, for training
  [PersonNet](../model/README.md): ~45 GB streamed, 30-60 minutes. Training is optional, since
  the trained weights are in the repository.

| file | purpose |
|---|---|
| `prepare_dataset.py` | downloads, centre-crops and labels Wake Vision, and draws the sample sets; `--full` adds the training cache. |
| `wake_vision.py` | the cache layout, the label names, the loader (`MemmapDS`) and `load_samples()`; used by `prepare_dataset.py`, by training and by the Belfort ML example. |

The crop and resize, `centre_square()`, are the model's input contract. They live in
[`../model/preprocess.py`](../model/preprocess.py) together with `IMG_SIZE`, and
`wake_vision.py` re-exports them. Every command below runs in the `dataset/` directory.

## What you need

- **[uv](../INSTALL.md#uv):** the first `uv run` sets up the environment from
  `pyproject.toml`.
- **A Hugging Face account (optional, free):** a login raises the download rate limit (see
  *Hugging Face login* below).
- **Disk and download:** the default mode keeps ~1.2 GB in `data_cache/`. `--full` streams
  ~45 GB and keeps ~16 GB of images at 64×64, plus ~3 GB of Open Images files.

## The sample frames

```bash
uv run prepare_dataset.py
```

It downloads one parquet shard of Wake Vision's validation split and one of its test split into
`data_cache/quick/` (reused on the next run),
centre-crops their ~3,500 images, labels them from the Open Images boxes (~160 MB of Open Images
files, once), and draws the two sets, each half near person and half not:

```
samples/
  test.npz               64 frames, Wake Vision test split
  calibration.npz        512 frames, Wake Vision validation split
  test/*.jpg             the same frames as images
  calibration/*.jpg
```

Each `.jpg` is a frame as the model sees it, enlarged 4× (`--upscale`), named
`<index>_<bucket>_task<label>_row<row>.jpg`; look at them, or feed them to
[`inference.py`](../model/README.md#3-inference-seconds). The script always fetches the same two shards of
Wake Vision, one of its validation split and one of its test split, and draws from them with a
fixed seed, so the two sets are identical on every machine. `--test` and `--calibration`
change their sizes, and `--seed` picks other rows. Fewer calibration frames make wrong results under
encryption more likely, so keep the default of 512 for anything beyond a quick trial.

Nothing in `data_cache/` or `samples/` is committed. Wake Vision's images are Flickr photos under
per-image licences, and the Hugging Face release carries no attribution.

## The training cache (`--full`)

```bash
uv run prepare_dataset.py --full        # the samples, then all splits: ~45 GB streamed, 30-60 min
```

It first does everything the default mode does, then streams all three splits
(`train_quality`, `validation`, `test`) into `data_cache/full/` and labels them, which
downloads ~3 GB of Open Images files once.

* The dataset is **streamed**: parquet shards (~500 MB each) are fetched, decoded and discarded,
  so nothing accumulates in `~/.cache/huggingface`. Only the resized arrays are kept.
* The first batch takes a few minutes to appear, because each worker fetches a shard first.
  After that, progress lines show images/s and the ETA. Check real progress any time with
  `du -h data_cache/full/train_quality_x.npy`.
* `--workers N` sets the number of download-and-decode workers; the default is the number of
  cores minus 2. Each worker holds a whole shard and its decoded images in memory, so on a
  machine with many cores and little RAM, such as WSL 2 with its default memory limit, the
  default can run out of memory. A run that stops with
  `DataLoader worker (pid ...) is killed by signal: Killed` did; re-run with `--workers 4`, and
  with `--workers 2` if that is still killed. On a slow connection, 2 workers are enough too,
  since they all share the bandwidth.
* Re-running skips splits that are already complete. If a run is interrupted mid-split, delete
  that split's files from `data_cache/full/` and re-run.
* A `resource_tracker: There appear to be 1 leaked semaphore objects` warning at the end is
  harmless. The script exits hard to stop the streaming reader from hanging, and Python's
  bookkeeping process notices. The cache is complete.
* A training cache from an earlier version of this repository is in `data_cache/wake_vision/`:
  `prepare_dataset.py --full` asks you to rename it to `data_cache/full/`, so it is reused.

## Hugging Face login (recommended, free)

The dataset is public; a login only raises the download rate limit. Create a free account, make a
**read-only** token at <https://huggingface.co/settings/tokens>, then:

```bash
uv run hf auth login
```

Paste the token and answer "n" to the git-credential question. The token is stored in
`~/.cache/huggingface/token` and picked up automatically.

## The crop

Every image is **centre-cropped to a square** with side `min(width, height)`, so a 4:3 photo
loses 12.5 % on each side, and then resized to 64×64. Proportions are preserved. The crop and
resize are `centre_square()`, the same function that `inference.py` and the Belfort ML example
use. The original width and height are stored too, so that the labels can apply the same crop
to the boxes.

## The labels

For every person image, the script finds its Open Images boxes (classes Person, Man, Woman, Boy,
Girl, Human face, Human head, Human body; boxes flagged `IsDepiction` or `IsGroupOf` are
excluded), intersects each with the centre square, and sizes the image by the largest visible
box: `side = sqrt(visible area / crop area)`.

| image | `dist` | task label |
|---|---|---|
| person, side > 0.38 | near | **1** |
| person, side 0.16–0.38 | medium | −1 (dropped: the ambiguous band) |
| person, side < 0.16 | far | **0** |
| person, box entirely outside the crop | cropped out | −1 (dropped: cannot be sure nobody else is in the crop) |
| person, no box in Open Images | no box | −1 (dropped) |
| no person | — | **0** |

The two cut-offs were fitted once on Wake Vision's human-labelled validation split, with 83 %
agreement with the human near / medium / far labels, and are constants in the script. Wake
Vision's human distance labels are not used anywhere else. The same rule labels training,
validation and test, so all three measure the same definition of "near".

Wake Vision's validation and test images come from the Open Images subsets of the same name, and
its training images from Open Images' train subset. Each image is looked up in the Open Images
image lists, and the person boxes are read only from the box files of the subsets the images
come from. The downloads, the image ids (`data_cache/open_images/image_ids.json`) and the parsed
boxes (`person_boxes_<subset>.npz`) are cached, so labelling again takes seconds. The rule and
the counts land in each split's `<split>_meta.json`.

## What is in the cache

```
data_cache/
  quick/                 the default mode: one parquet shard each of validation and test
  full/                  --full: train_quality, validation, test
  open_images/           the Open Images image lists and boxes, shared by both
```

Each split in `quick/` and `full/` is the same set of files:

```
  <split>_x.npy          uint8 (N, 3, 64, 64)   centre-cropped images, CHW, memory-mapped by the loader
  <split>_y.npy          int8  Wake Vision person label 1 / 0
  <split>_flags.npy      int8  (N, 9) Wake Vision attribute flags; only the lighting ones are used (breakdown)
  <split>_wh.npy         int32 (N, 2) original width, height
  <split>_filenames.npy  original Flickr file names (join key for Open Images)
  <split>_task.npy       int8  the task label: 1 / 0 / -1 (dropped)
  <split>_dist.npy       int8  0 near, 1 medium, 2 far, 3 cropped out, 4 no box, -1 no person
  <split>_meta.json
```

Splits in `full/`: train_quality 1,196,221 rows, validation 18,582, test 55,763, each half
person / half not before labelling.

## Using it from code

```python
from wake_vision import MemmapDS, cache_dir, load_samples   # with dataset/ on sys.path
ds = MemmapDS(cache_dir(), "validation")                     # data_cache/full/: (uint8 CHW, task label, row); dropped rows skipped
X, y = load_samples("test")                                  # samples/test.npz: uint8 (N, 3, 64, 64), labels
```
