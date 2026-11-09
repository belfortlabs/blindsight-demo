# model: PersonNet, the near-person detector

PersonNet is the person detector that runs under FHE: a 4-layer convolutional net (~60k
parameters; Conv2d, BatchNorm, ReLU, AvgPool and Linear only), trained once on Wake Vision to
answer **is there a person close to the camera?** Its input is a 64×64 image.

![PersonNet at 64×64: four Conv 3×3, BatchNorm and ReLU blocks with strides 2, 2, 2 and 1 (16, 32, 64 and 64 channels), global average pooling over an 8×8 map, and a Linear 64 → 2 layer producing two logits; 60,642 parameters with BatchNorm folded, 5.16M multiply-accumulates, 32,768 ReLUs](assets/personnet.svg)

The diagram shows the network at 64×64. The ReLUs (green) are the only non-linear layers, and
the costly part under FHE, where each is approximated by a polynomial; everything else is
linear.

Every image is centre-cropped to a square and then resized, so proportions are preserved. Every
label comes from one source: Open Images' human-verified bounding boxes, intersected with that
same crop. Wake Vision's own human distance labels are not used. The data comes from
[`../dataset/`](../dataset/README.md), which the Belfort ML example shares.

```
model/
├── assets/            the network diagram above
├── preprocess.py      the input contract: IMG_SIZE, centre_square(), to_input()
├── training/          2. train PersonNet (near person = 1; nobody or far person = 0)
│   ├── personnet.py   PersonNet + load(): the only model definition in the repository; inference imports it
│   ├── train.py       the training loop -> runs/baseline/best.pt, train.json, last.pt; --resume
│   ├── evaluate.py    accuracy on the held-out test split, with a breakdown -> evaluate.json
│   ├── common.py      what the two share: pick_device() (MPS / CUDA / CPU), predict_correct() and the breakdown
│   └── runs/baseline/ the trained weights and their records (see *The runs directory*)
├── inference/         3. run it on an image in plaintext: the reference the FHE pipeline must match
│   └── inference.py   uv run inference.py IMAGE, plus predict() to reuse
├── pyproject.toml     the Python version and packages, for uv
└── README.md          this file
```

This page is the end-to-end path. Every command below runs in the `model/` directory; a step
that changes directory says so with `# from model/`.

## The model

Four `Conv2d(3×3) → BatchNorm → ReLU` blocks with channels (16, 32, 64, 64) and strides
(2, 2, 2, 1), a global average pool, and one `Linear(64, 2)`. Every choice is made for FHE:
dense convolutions rather than depthwise ones, plain ReLU rather than ReLU6, no pooling except
the final linear average, no dropout, and only 4 non-linear layers, since multiplicative depth
is what drives the cost under CKKS. BatchNorm exists only at training time; the FHE compiler
folds it into the convolutions.

At 64×64: 60,642 parameters, ~5.2M MACs, ~32.8k ReLU evaluations, a largest activation of
16×32×32 = 16,384 values, a receptive field of 31 px, and a final feature map of 64×8×8. To try
other widths or strides, edit `CHANNELS` and `STRIDES` in `training/personnet.py`. The input
resolution is `IMG_SIZE` in `preprocess.py`, shared by the cache, the model and inference, so
the three can never disagree.

The input is `(N, 3, 64, 64)`, ImageNet-normalised float after a centre crop (`preprocess()`
in `preprocess.py`); the output is logits `(N, 2)`, with index 1 = near person.

## What you need

- **[uv](../INSTALL.md#uv):** the first `uv run` in
  `model/`, or in any directory under it, creates `.venv/` (git-ignored) with a suitable Python
  and the packages in `pyproject.toml`. Later runs reuse it.
- **The dataset, for training only:** all three Wake Vision splits, fetched and labelled
  (step 1). Inference needs no dataset.
- **A GPU (optional):** the scripts pick the accelerator automatically: the Apple-silicon GPU
  (MPS) on a Mac, CUDA on a machine with an NVIDIA GPU (the Linux `torch` wheel ships CUDA
  support), and the CPU otherwise. The first line each script prints shows which one it chose.

### Platform notes

* **macOS**: raise the open-file limit once per terminal with `ulimit -n 4096`, or the data
  loader workers die after the first epoch. Training takes about an hour, so prefix the
  command with `caffeinate -i` to keep the laptop awake: `caffeinate -i uv run train.py`.
* **Linux**: neither is needed; `nohup` or `tmux` keep a long run going. PersonNet is so small
  that on a GPU, training speed is limited by how fast the data
  loader supplies images, not by the GPU itself. If the `img/s` figure in the progress lines is
  low, a larger batch (`--batch 512` or `1024`, default 256) spreads the per-batch overhead over
  more images.

## 1. Dataset

Training needs all three Wake Vision splits, fetched and labelled:

```bash
cd ../dataset                                            # from model/
uv run prepare_dataset.py --full                         # ~45 GB streamed, 30-60 min -> data_cache/full/
```

[`../dataset/README.md`](../dataset/README.md) also covers the Hugging Face login, the crop and
how the labels are derived. `--full` also draws the sample frames, so the Belfort ML example and
the sample images for step 3 are ready afterwards. Inference alone needs no dataset.

## 2. Train (optional, ≈1 h on an M-series MacBook)

The trained weights, `training/runs/baseline/best.pt`, are in the repository. This step is only
needed to retrain, for example after changing the model or the data.

One training run on the full Wake Vision training split, with task labels derived from
bounding boxes: 1 = a person close to the camera, 0 = nobody or a person far away.
Medium-distance persons, persons outside the centre crop and persons without a box are left
out.

```bash
cd training                                              # from model/
uv run train.py                                          # 15 epochs -> runs/baseline/best.pt (+ train.json)
uv run evaluate.py                                       # held-out test split: accuracy + breakdown -> evaluate.json
```

Options: `--epochs` (15), `--batch` (256), `--lr` (2e-3 peak; one warm-up epoch, then cosine
decay) and `--resume`. Fixed in the script: AdamW with weight decay 0.05, label smoothing 0.05,
a class-weighted loss (label 0 outnumbers label 1), and augmentation by a random shift of ±4 px
and a horizontal flip. A progress line every 250 steps shows the loss, the running accuracy,
the learning rate, images/s and the ETA. Each epoch ends with the validation accuracy and a
`*best*` marker when it improves. The per-epoch numbers are written to `runs/baseline/train.json`.

Reading the numbers: the loss starts near 0.69, which is chance for two classes, and should
fall fast in epoch 1. `val` is the honest figure; a growing gap between `train` and `val` means
memorisation. In the breakdown, `person far (rejected)` counts the label-0 rows that contain a
person the model must not fire on, so that line is a rejection rate (higher is better).
Training, validation and test all use the same box-derived definition of "near", so their
numbers are comparable; the test split is never trained on.

`runs/baseline/best.pt` is a plain `state_dict`. `personnet.load()` reads it from there by
default, or any other checkpoint with `load(path)`.

## 3. Inference (seconds)

`inference.py` loads the trained weights, preprocesses images exactly as the training cache
was built, and prints a prediction per image. It is the plaintext reference: the FHE pipeline
is correct when it reproduces these logits, within CKKS noise, for the same inputs.

```bash
cd inference                                             # from model/
uv run inference.py photo.jpg
```

It prints the verdict, p(near) and the two raw logits for that image, for example:

```
photo.jpg: near person  p(near)=0.912  logits=(-1.204, 1.137)
```

The weights come from `training/runs/baseline/best.pt`, as training wrote them; there is nothing
to copy. Accuracy is not scored here; that is what `training/evaluate.py` is for.

No photo at hand? The sample frames from step 1 are also there as JPEGs, in
`../dataset/samples/test/` and `../dataset/samples/calibration/`, with the label in the file
name. Or use your own photos, say a person at arm's length, a
person across the room and an empty scene; expect a few more misses than on the dataset, since
a webcam and Flickr photos differ in framing and lighting.

### Preprocessing contract

Whatever feeds the model, in the clear or encrypted, must do exactly what
`../dataset/prepare_dataset.py` did to the training images:

1. decode to RGB,
2. centre-crop to a square (side = min(width, height)), then resize to 64×64, so proportions are preserved,
3. scale to [0, 1], subtract the mean (0.485, 0.456, 0.406) and divide by the standard deviation (0.229, 0.224, 0.225), per channel,
4. shape (N, 3, 64, 64), float32.

`preprocess()` in `preprocess.py` does this in a few lines: it chains `centre_square()`, the
same code the training cache was built with, and `to_input()`. The input stage of the FHE
compiler should match it, or fold step 3 into the first convolution.

### From code

```python
from training.personnet import load
from preprocess import preprocess
from inference.inference import predict
from PIL import Image

net = load()                                           # training/runs/baseline/best.pt, CPU, eval mode
x = preprocess(Image.open("photo.jpg"))                # (1, 3, 64, 64)
labels, p_near, logits = predict(net, x)
```

## The `runs` directory

`training/runs/baseline/` holds the weights and their records:

```
training/runs/baseline/
    best.pt          plain state_dict, best validation epoch                                   (committed)
    train.json       the training record: configuration, best validation accuracy, per-epoch numbers (committed)
    evaluate.json    the test-split record: accuracy and the breakdown lines of evaluate.py       (committed)
    last.pt          full checkpoint for --resume                                               (ignored)
```

`best.pt` (~250 KB) and the two `.json` files are committed, so a clone comes with the model
and its records and can run inference straight away. The `.json` files are records for people;
nothing in the code reads them. Everything else that is generated is git-ignored: `last.pt`,
`__pycache__/` and `.venv/` here, and the dataset's cache and samples in `../dataset/`.

## Licence and attribution

The weights are released, like the code, under the repository's
[Apache License 2.0](../LICENSE). They were trained on Wake Vision (Harvard Edge Computing Lab,
CC BY 4.0), with labels derived from Open Images bounding-box annotations (Google, CC BY 4.0).
The attributions and citations are in [`../NOTICE`](../NOTICE), which should travel with the
weights when they are redistributed or built into something else.
