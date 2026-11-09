"""The Belfort ML client: encrypt images with the secret key, have the Belfort ML deploy classify
them, and decrypt the verdicts with the secret key. Three modes:

    uv run client.py                                  # a live webcam, in a window
    uv run client.py --samples                        # the 64 test samples, checked
    uv run client.py --samples 10                     #   ... only the first 10
    uv run client.py --images photo.jpg other.png     # your own image files

Webcam (the default): once the client is prepared, a window opens with the live camera and
classifies the newest frame, one at a time, with an animation of each frame's trip (webcam.py,
with the camera and window in webcam_macos.py on macOS and webcam_linux.py on Linux and WSL 2).

--samples [COUNT]: sends the frames of ../dataset/samples/test.npz (all 64, or the first COUNT)
and checks each verdict against the frame's label. They come from Wake Vision's test split, which
neither training nor calibration used.

--images FILE...: sends those image files instead: any size or format PIL reads, prepared with
the model's preprocess() (centre crop to a square, resize to IMG_SIZE, normalisation). It prints
each verdict; there are no labels to check against.

Every mode prepares the client the same way (prepared_client()): it fetches the client code,
makes the key pair, from the client token's seed in its .secret-seed-* file if there is one,
and uploads the evaluation keys. The first time on a client token the SDK draws the seed, and
it is recorded in the seed file once the keys are made; that upload can take several tens of
minutes. Any mode can then reuse the key pair without a new upload.

The client token: client.py reads it from .client-token, which provider.py serve writes here
while it runs. On another machine, export the BELFORT_CLIENT_TOKEN line serve printed instead;
a .client-token here wins over an exported token, which may be left over from an earlier run.

After provider.py serve --dev (developer mode, testing only), the gateway chose the seed and the
GPU already holds the evaluation keys: serve wrote the seed to the token's .secret-seed-* file,
so client.py needs no option and uploads nothing. On another machine, also export the
BELFORT_CLIENT_SECRET_SEED line serve printed.
"""

import argparse
import hashlib
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import trained_model
from belfort_ml_client import Client, ClientError
from PIL import Image
from trained_model import CLASS_NAMES, preprocess, test_samples


# ---------------------------------------------------------------- the client token's key pair

def seed_file(token: str) -> Path:
    """Where the seed of this client token's secret key is kept (git-ignored): one file per client
    token, so every mode on the same client token uses the same key pair, the one whose evaluation
    keys the deploy holds."""
    return trained_model.HERE / f".secret-seed-{hashlib.sha256(token.encode()).hexdigest()[:12]}"


DEPLOY_READY = threading.Event()  # set once a deploy holds the keys: from then on it is running


def duration(seconds: float) -> str:
    seconds = round(seconds)
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60} min {seconds % 60:02d} s"


@contextmanager
def step(text: str):
    """Print what a preparation step does and how long it took. Meanwhile the Belfort ML client
    shows its own progress: a status line while it waits, a bar while it uploads keys."""
    print(text, flush=True)
    start = time.monotonic()
    yield
    print(f"      done in {duration(time.monotonic() - start)}", flush=True)


@contextmanager
def prepared_client(token: str):
    """A Belfort ML client on this client token, prepared: client code fetched and built, the key
    pair made, and the evaluation keys held by the deploy.

    The key pair comes from the seed in seed_file() when the file exists. Otherwise the SDK draws
    the seed, and it is recorded in seed_file() once the keys are made, so later runs on the same
    client token reuse the key pair."""
    print("Preparing the Belfort ML client in three steps.", flush=True)
    path = seed_file(token)
    if path.exists():
        seed, source = bytes.fromhex(path.read_text()), f"the seed in {path.name}"
    elif os.environ.get("BELFORT_CLIENT_SECRET_SEED"):
        # A developer-mode seed from provider.py serve --dev on another machine (see its docstring).
        seed, source = bytes.fromhex(os.environ["BELFORT_CLIENT_SECRET_SEED"]), "BELFORT_CLIENT_SECRET_SEED"
    else:
        seed, source = None, ""
    new = seed is None
    with Client(token, secret_seed=seed) as client:
        # client.prepare() in its three steps: each gets a heading, and the seed is recorded as soon
        # as the keys are made, not only once the deploy holds them. prepare()'s GPU warm-up is left out.
        with step("1/3 Client code: the encryption code for this compiled model. The first time it is "
                  "downloaded and compiled, which can take a few minutes; after that it is reused."):
            client.fetch_code()

        upload = client.status()["state"] in ("awaiting_keys", "awaiting_upload")
        origin = (f"a new key pair for this client token (its seed goes to {path.name})" if new else
                  f"the key pair of this client token, from {source}")
        if upload:
            text = (f"2/3 Keys: generating {origin}, then uploading the evaluation keys to the "
                    f"gateway. This happens once per client token and can take several tens of "
                    f"minutes, depending on your connection.")
        else:
            text = (f"2/3 Keys: generating {origin}. The gateway already holds its evaluation "
                    f"keys, so nothing is uploaded.")
        with step(text):
            client.publish_keys()
            if new:
                path.touch(mode=0o600)
                path.write_text(client.secret_seed.hex())

        with step("3/3 Deploy: waiting until the GPU deploy has loaded the evaluation keys."):
            client.wait_until_ready()
        DEPLOY_READY.set()
        yield client


# ---------------------------------------------------------------- classifying one image

@dataclass
class Verdict:
    predicted: int           # index into trained_model.CLASS_NAMES
    predicted_name: str
    scores: list             # the decrypted scores, one per class
    latency_ms: float        # the gateway's call to the GPU runtime


def classify(client: Client, values) -> Verdict:
    """Encrypt one prepared image (preprocess()'s (1, 3, S, S) tensor) with the secret key, have the
    deploy classify it, and decrypt the scores with the secret key."""
    answer = client.infer([values.flatten().tolist()])
    scores = list(answer["values"])
    predicted, name = trained_model.verdict(scores)
    return Verdict(predicted=predicted, predicted_name=name, scores=scores,
                   latency_ms=answer["latency_ms"] or 0.0)


# ---------------------------------------------------------------- the three modes


def classify_files(token: str, inputs) -> None:
    """Classify (name, prepared image, label or None) inputs; count the correct ones if labelled."""
    with prepared_client(token) as client:
        correct, labelled = 0, 0
        for name, image, label in inputs:
            sent = time.monotonic()
            verdict = classify(client, image)
            timing = f"in {(time.monotonic() - sent) * 1000:.0f} ms (GPU {verdict.latency_ms:.0f} ms)"
            if label is None:
                scores = ", ".join(f"{s:.3f}" for s in verdict.scores)
                print(f"{name}: {verdict.predicted_name} (scores {scores}) {timing}")
            else:
                labelled += 1
                correct += verdict.predicted == label
                print(f"{name} ({CLASS_NAMES[label]}): predicted {verdict.predicted_name} {timing}")
        if labelled:
            print(f"{correct} of {labelled} correct")


def main():
    p = argparse.ArgumentParser(description="The Belfort ML client: classify images on encrypted data.",
                                epilog="Without --samples or --images: the live webcam, in a window.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--samples", nargs="?", type=int, const=0, metavar="COUNT",
                      help="send the test samples and check them (all, or the first COUNT)")
    mode.add_argument("--images", nargs="+", metavar="FILE", help="send your own image files")
    a = p.parse_args()

    token_file = trained_model.HERE / ".client-token"  # written by provider.py serve
    if token_file.exists():
        token = token_file.read_text().strip()
    elif not (token := os.environ.get("BELFORT_CLIENT_TOKEN")):
        sys.exit("Run `uv run provider.py serve` here first, or set BELFORT_CLIENT_TOKEN.")

    try:
        if a.images:
            # Load the files before preparing the client, so a missing file fails at once.
            try:
                inputs = [(path, preprocess(Image.open(path)), None) for path in a.images]
            except OSError as exc:
                sys.exit(f"cannot read an image: {exc}")
            classify_files(token, inputs)
        elif a.samples is not None:
            images, labels = test_samples()
            inputs = [(f"sample {i}", image, label) for i, (image, label) in enumerate(zip(images, labels))]
            classify_files(token, inputs[: a.samples or None])
        else:
            import webcam  # picks the camera and window for this platform
            # Ask for camera access now, not after a long key upload.
            webcam.ensure_camera_access()
            with prepared_client(token) as client:
                webcam.run(client, classify)
    except ClientError as exc:
        sys.exit(f"Belfort ML client: {exc}")
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    # The deploy is the provider's: it runs, and costs credits, until it is stopped there.
    if DEPLOY_READY.is_set():
        print("The deploy keeps running until you press Enter in the terminal of provider.py serve "
              "(or 30 minutes after it started).", flush=True)


if __name__ == "__main__":
    main()
