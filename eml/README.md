# eml: the Belfort ML example, PersonNet on encrypted camera frames

This example runs PersonNet, the near-person detector, on encrypted 64×64 camera frames. The
Belfort ML client encrypts each frame, a GPU runs the model
on the ciphertext, and the client decrypts two scores, `not near` and `near person`.

The directory is named after what Belfort ML implements, encrypted ML (eML). The example has
the shape of the SDK's quickstart: `provider.py` compiles and serves the model, and `client.py`
is the Belfort ML client: for a live webcam, in a window, or for labelled sample frames
(`--samples`) or your own images (`--images`). Unlike the quickstart, it works only inside this
repository: `trained_model.py` loads the network and its trained weights from `../model/`, and
the sample frames from `../dataset/samples/`.

Every command below runs in the `eml/` directory, unless it starts with a `cd`.

## What you need

The repository README's [*What you need*](../README.md#what-you-need) and
[`INSTALL.md`](../INSTALL.md) cover the set-up: uv, a C++20 compiler, a Belfort API
key and the sample frames for calibration and testing, drawn by `prepare_dataset.py` in
[`../dataset/`](../dataset/README.md). The trained weights,
`../model/training/runs/baseline/best.pt`, are in the repository, so training is optional. The
webcam mode of `client.py` runs on macOS, Linux and WSL 2; on WSL 2 the webcam must first be
attached with usbipd, as the [installation page](../INSTALL.md#webcam-on-wsl-2) describes.

## 1. Compile

```bash
export BELFORT_API_KEY=blf_live_...          # your API key
uv run provider.py compile
```

`provider.py` loads the trained weights and compiles the model, calibrating each ReLU on the
512 calibration frames in `../dataset/samples/calibration.npz`.
The compiler measures the range of values that reaches each ReLU and adds a margin. A frame
whose values fall outside that range decrypts to a wrong result, so more calibration frames
make the encrypted model more robust. The compiled model is uploaded as `personnet`. Running
`compile` again reuses it.

On encrypted data, each ReLU is approximated by a polynomial. `provider.py` compiles with
`use_composite_relu=True` (in `compiled_model()`), the SDK's default: three
chained polynomials per ReLU. This is accurate, at the cost of more levels and bootstraps,
slower inference and larger keys. The alternative, `False`, one degree-5 polynomial per ReLU,
needs fewer levels and smaller keys but is too coarse for PersonNet.

`provider.py` also compiles with `level_zero_encryption=True` (also in
`compiled_model()`). The Belfort ML client then encrypts each frame at the bottom of the modulus
chain. It does not have to upload the whole chain, so the ciphertext sent to the gateway is
much smaller and arrives faster. The deploy bootstraps each frame back to the
top of the chain before the network runs, which adds GPU time per frame, and the model gets a
bootstrapping parameter set. The SDK advises this setting only for a model that bootstraps
anyway; otherwise the evaluation keys grow a lot. Set it to `False` to upload the full chain
instead. Nothing changes on the client side: `client.py` encrypts at whatever level the model asks
for.

A compiled model is tied to the weights, the calibration frames, the options
`use_composite_relu` and `level_zero_encryption`, and the Belfort release (see *Belfort
release*). After any of these changes, `compile` stops with a `name_taken` error. Delete the old model and compile again:

```bash
uv run provider.py delete
uv run provider.py compile
```

## 2. Serve

In the same terminal:

```bash
uv run provider.py serve --dev   # development only
```

`serve --dev` is developer mode: the deploy opens its own key set, the gateway chooses the
secret seed and the GPU generates the evaluation keys from it, so the client has nothing to
upload. Once the deploy is ready, the provider writes the client token to `.client-token` and
the seed to that token's `.secret-seed-*` file, and prints both as `export` lines
(`BELFORT_CLIENT_TOKEN` and `BELFORT_CLIENT_SECRET_SEED`) for a client on another machine.
**Developer mode is for development only**: the gateway knows the seed, so it can decrypt
everything the deploy serves. Never use it with real data; serve without `--dev` instead, as
step 5, *Your own keys*, describes. Developer mode needs belfort-ml 0.2.5 or later (see
*Belfort release*).

Ctrl-C stops a deploy that is still starting. The deploy stops when you press Enter, or by
itself after 30 minutes. The time it ran is charged to your Belfort ML account in credits,
rounded up to 10 minutes. `.client-token` is removed when the script exits.

## 3. Classify a live webcam

Without options, `client.py` is the Belfort ML client for a webcam: it opens a window that
shows the camera picture, and classifies its frames on encrypted data. In a second terminal, in
the `eml/` directory:

```bash
uv run client.py
```

`client.py` reads the client token from `.client-token`. To run it on another machine, run the
`export BELFORT_CLIENT_TOKEN=...` line the provider printed there first.

macOS first asks once for camera access for your terminal app; if it was denied, allow it in
System Settings > Privacy & Security > Camera. On Linux, `client.py` checks before preparing
that there is a display and a `/dev/video*` camera it can open, and says what to do if not; on
WSL 2, attach the webcam with usbipd first (see the
[installation page](../INSTALL.md#webcam-on-wsl-2)).

Then `client.py` prepares the client in the terminal, as for `--samples` and `--images`, and
opens the window only once it can classify frames; pick a camera and a capture format in the
window's menus. Preparing means building the encryption code, a few minutes the first time,
and making the key pair from the seed in the client token's `.secret-seed-*` file, which
`serve --dev` wrote. The GPU already holds the evaluation keys, so nothing is uploaded. After
that, each image takes a few hundred milliseconds, not counting the upload of the encrypted
image, which depends on your connection. Without `--dev`, the client draws the seed itself and
uploads the evaluation keys, which can take several tens of minutes (see step 5, *Your own
keys*).

Then it classifies one frame at a time, always the newest; frames captured meanwhile are
skipped. Each frame is converted from the camera's format to RGB, prepared with the
model's `preprocess()` (centre crop, resize, normalisation), encrypted with the secret key,
classified on the deploy and decrypted with the secret key. The image never leaves the process
unencrypted. A frame that cannot be converted is logged and skipped; while the deploy is
unreachable, the window keeps showing the camera and the client retries.

### What you see

The window is titled **BlindSight** and shows the live camera. The status line under it gives
the number of frames classified on its left (or, while the deploy is unreachable, that) and the
last verdict on its right: its class, its scores and two timings, the round trip from sending
the encrypted frame to receiving the verdict, and the FHE evaluation, the part of that round
trip the GPU spent running the model. Along the bottom of the picture, an animation follows
each frame:

1. **Snap.** The centre square of the frame, the part the network sees, shrinks from the
   picture into the lower left.
2. **Encrypt.** Block by block, the picture turns into noise. The noise illustrates the
   ciphertext, which is sent to the deploy.
3. **To the deploy.** The encrypted picture slides to the middle, where a black dashed box
   turns around it, to show that the FHE evaluation is running, until its verdict comes back.
4. **Verdict in.** It slides to the lower right with a grey box, while the previous result
   slides out.
5. **Decrypt.** Block by block it turns back into the picture, and the box takes its colour for
   p(near), the probability that a person is near: dark green means a person is near
   (p(near) = 1), dark red means nobody is near (p(near) = 0), and orange means the model is
   unsure (p(near) = 0.5), with the colours blended in between. There is no caption: the box's
   colour is the verdict.

Each thumbnail matches the image that was sent for classification, not the live picture when
the verdict arrives, so the coloured box always belongs to the image it surrounds. p(near) is
the softmax of the two decrypted scores.

The constants at the top of `webcam.py` set the timings (`SNAP_S`, `ENCRYPT_S`,
`TO_MIDDLE_S`, `TO_RIGHT_S`, `DECRYPT_S`, `FADE_S`), the noise grid (`NOISE_BLOCKS`), the
dashed box (`BUSY_*`), the thumbnail size (`PIP_FRACTION`) and the p(near) colours
(`ANCHORS`).

Stop it by closing the window, pressing Cmd-Q (macOS) or Ctrl-Q (Linux) in it, or pressing
Ctrl-C in its terminal. The deploy keeps running until you press Enter in the first terminal.

## 4. Or: check the compiled model with sample frames or your own images

With `--samples` or `--images`, `client.py` stands in for the camera, without a window; these
modes need no camera. Run it in the second terminal instead of the webcam mode:

```bash
uv run client.py --samples                    # or: uv run client.py --samples 10
uv run client.py --images photo.jpg           # your own images instead of the samples
```

It prepares exactly like the webcam mode in step 3, with the same seed file. A client token
that one mode has used therefore works as it is for the others.

With `--samples`, `client.py` takes the 64 test frames in `../dataset/samples/test.npz`, or the
first ones if you give it a number such as `10`. It encrypts them with the secret key, has the deploy classify them,
decrypts the verdicts and prints one line per frame, such as
`sample 0 (near person): predicted near person in 412 ms (GPU 287 ms)`, followed by the number
of correct frames.

Expect about 55 of 64 correct, in plaintext too. On Wake Vision's whole test split, PersonNet
finds 78.9 % of the near persons and 93.4 % of the rest
(`../model/training/runs/baseline/evaluate.json`), and the test frames are half of each. With only
64 frames, a count anywhere from about 51 to 60 is normal.

With `--images`, it sends your own image files instead, of any size or format. Each image is
prepared with the model's `preprocess()` (centre crop to a square, resize to 64×64,
normalisation), encrypted with the secret key, classified, and decrypted with the secret key.
The script prints each verdict with its scores, such as
`photo.jpg: near person (scores -1.204, 1.137) in 418 ms (GPU 291 ms)`. There are no labels to check
against, and only the centre square of each image is seen, as in training.

Press Enter in the first terminal to stop the deploy.

## 5. Your own keys: serve without `--dev`

```bash
uv run provider.py serve
```

Without `--dev`, the seed never leaves the client. The provider mints a client token, writes it
to `.client-token`, prints it as an `export BELFORT_CLIENT_TOKEN=...` line, and waits.
`client.py` then draws the seed itself on its first run with that token, records it in the
token's `.secret-seed-*` file, derives the secret key and the evaluation keys from it and
uploads the evaluation keys. Only once the keys are in does the provider start the GPU deploy,
so the GPU does not bill while the client builds its encryption code and uploads its keys:

```
 provider.py compile ──────────▶ compiled model on the gateway
 provider.py serve   ──────────▶ client token            (.client-token)
                                       │
                                       ▼
 client.py (start-up):           client token's seed     (.secret-seed-*)
                                       │
                                       ▼
                                 secret key
                                       │
                                       ▼
                                 evaluation keys
                                       │  upload
                                       ▼
                                 gateway holds them      (one key set per client token)
                                       │
                                       ▼
 provider.py serve   ──────────▶ GPU deploy loads them
```

**That upload can take several tens of minutes**, depending on the speed of the connection
between the Belfort ML client and the gateway. It happens once per client token; later runs on
the same token reuse the key pair without a new upload, and the window opens within a minute or
so. If you stop the client during that first upload, restart `provider.py serve` for a new
client token.

Everything else is the same: steps 3 and 4 run unchanged, a client on another machine needs
only the `BELFORT_CLIENT_TOKEN` line, and the deploy stops with Enter or after 30 minutes.

## Files

| File | What |
|---|---|
| `trained_model.py` | Finds the trained model in `../model/` and the samples in `../dataset/`: image size, class names, preprocessing, the trained network (`load_model()`), the two sample sets, normalised (`test_samples()`, `calibration_samples()`), and the predicted class for decrypted scores (`verdict()`) |
| `provider.py` | The model provider: `compile`, `serve` and `delete` |
| `client.py` | The Belfort ML client: the live webcam (default), sample frames checked against their labels (`--samples`), or your own images (`--images`). It holds the client token's key pair (`prepared_client()`), with its seed kept in `.secret-seed-*`, and encrypts, classifies and decrypts one image (`classify()`) |
| `webcam.py` | The webcam mode, on every platform: the classification of the newest frame, the animation's timings and colours, and the choice of `webcam_macos.py` or `webcam_linux.py` |
| `webcam_macos.py` | The camera and window on macOS: the camera (AVFoundation), the YUV-to-RGB conversion, the window and the animation (AppKit and Core Animation) |
| `webcam_linux.py` | The camera and window on Linux and WSL 2: the cameras and their formats (V4L2), the capture (OpenCV), the window and the animation (Qt) |
| `pyproject.toml` | The Belfort ML packages, pinned to one release (see *Belfort release*), `pillow` for the model's preprocessing, the PyObjC packages for the webcam window and camera on macOS, and OpenCV for the camera and PySide6 (Qt) for the window on Linux |

`build/`, `.secret-seed-*`, `.client-token`, `.venv/` and `uv.lock` are generated in `eml/` and
git-ignored. The sample frames are generated and ignored in `../dataset/`.

## Belfort release

`pyproject.toml` pins `belfort-ml`, `belfort-ml-client` and `belfort-ml-bucket` to one release,
`0.2.5`, the first with developer mode. The gateway refuses a `belfort-ml` or
`belfort-ml-client` of another release with a `version_mismatch` error that names the release it
expects. A compiled model is tied to the release it was compiled with. To move to another
release, change the three pins together, run `uv sync`, then `uv run provider.py delete` and
`compile` again.
