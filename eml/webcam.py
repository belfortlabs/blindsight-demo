"""The webcam mode of client.py: show a webcam, encrypt its frames with the secret key, have
the Belfort ML deploy classify them, and decrypt the verdicts.

    uv run client.py                      # client.py loads this module and calls run()

This module holds what does not depend on the platform: the newest-frame hub, the link that
classifies frames, the verdict colours and the animation's timings. The camera and the window
are in a module per platform, which run() and ensure_camera_access() pick:

    webcam_macos.py   macOS: AVFoundation for the camera, AppKit and Core Animation for the window
    webcam_linux.py   Linux and WSL 2: V4L2 through OpenCV for the camera, Qt for the window

client.py prepares the Belfort ML client first, in the terminal, as for its other modes, and
opens the window only once frames can be classified. The camera and capture format are picked
from menus. Then each newest frame is converted to RGB, prepared with the model's preprocess()
(centre crop, resize, normalisation), encrypted with the secret key, classified on the deploy
and decrypted with the secret key, one at a time: frames captured while a classification runs
are skipped. The image never leaves this process unencrypted. The status line under the picture
shows the count of classified frames on its left and the last verdict on its right. A frame that cannot be converted is logged and skipped; while the deploy is
unreachable, the window keeps showing the camera and the client retries.

Along the bottom of the picture, an animation shows each frame's trip. Each thumbnail is the
centre square of the frame, the part the network sees, taken from the frame that was sent for
classification, not from the live picture when the verdict arrives, so the coloured box always
belongs to the image it surrounds.

  * snap: the centre square of the frame shrinks from the picture into the lower left
    (0.6 s).
  * encrypt (0.6 s): block by block, in random order, the picture turns into noise:
    32 x 32 blocks of random colours.
  * to the middle (0.4 s): the encrypted picture slides to the middle, where a dashed box
    turns around it until its verdict comes back.
  * verdict in (0.4 s): it takes a grey box and slides to the lower right, in front of
    the next frame being snapped and still encrypted, while the previous result slides
    out.
  * decrypt (0.6 s): block by block it turns back into the picture, and the box fades
    from grey to its colour for p(near): dark green when a person is near (1), dark red
    when nobody is (0), orange when unsure (0.5), blended in between. p(near) is the
    softmax of the decrypted scores. No caption: the box's colour is the verdict.
  * a frame that cannot be converted is skipped: its thumbnail fades out and the next
    frame is snapped.

The noise is random, drawn for each frame; it only illustrates the ciphertext, which is sent
to the deploy. If a new verdict reaches the right while the previous result is still
decrypting, that result is shown decrypted at once and slides out. The deploy usually answers
before the frame reaches the middle, so it goes straight on to the right; the frames
classified meanwhile are not shown, so the display shows only some of the frames sent.

In the top left of the picture is belfortlabs.com's header logo, the mark and "Belfort" in white
(belfort_wordmark.png beside this file), on a square-cornered plate of the site's dark teal.
"""

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

import trained_model
from belfort_ml_client import ClientError

RETRY_S = 5.0          # seconds before retrying after the deploy could not be reached

# Thumbnails
PIP_FRACTION = 0.22   # thumbnail side, as a fraction of the shown camera picture's height
PIP_MAX_WIDTH = 0.3   # ... and at most this fraction of its width, so the three fit side by side
PIP_MARGIN = 12       # points between a thumbnail and the picture's edge
PIP_MAX_PX = 480      # frames are decimated to at most this many pixels wide before cropping
NOISE_BLOCKS = 32     # the encrypted picture: NOISE_BLOCKS x NOISE_BLOCKS blocks of random colour
BOX_PX = 6            # thickness of the result's box
IDLE_BORDER = 1       # border of a frame on its way to the deploy
GREY = (128, 128, 128)  # the result's box while it is still encrypted

# Animations, in seconds
SNAP_S = 0.6          # the centre square shrinking from the picture to the lower left
ENCRYPT_S = 0.6       # the picture turning into noise
TO_MIDDLE_S = 0.4     # the encrypted picture sliding to the middle
TO_RIGHT_S = 0.4      # the encrypted result sliding to the right, and the previous one out
DECRYPT_S = 0.6       # the noise turning back into the picture, the box into its colour
FADE_S = 0.5          # a refused frame fading out

# The turning box around the frame at the deploy
BUSY_GAP = 4          # px between the thumbnail and the dashed box
BUSY_LINE = 4         # px, line width of the dashed box
BUSY_DASH = (12, 7)   # px on, px off
BUSY_PERIOD_S = 0.45  # one dash length (on + off) travelled round the box per period

# The logo in the top left corner of the picture
LOGO = Path(__file__).with_name("belfort_wordmark.png")  # 1000 x 248, white on transparent
LOGO_ASPECT = 125 / 31     # the site's SVG viewBox
LOGO_FRACTION = 0.2        # the logo's width, as a fraction of the shown camera picture's height
LOGO_MIN = 100             # points wide, at least
LOGO_PAD = 0.25            # teal around the logo, as a fraction of its height
LOGO_PLATE = (6, 62, 83)   # the site's secondary background, #063e53, as in its bookmark icon

# Stacking: the result on its way to the right passes in front of the next snapped frame
Z_FLOW = 1            # snapped, encrypting, at the deploy
Z_LEAVING = 2         # the previous result sliding out
Z_RESULT = 3          # the result sliding to the right and decrypting

# p(near) -> colour of the box, RGB.
DARK_RED, ORANGE, DARK_GREEN = (140, 0, 0), (255, 140, 0), (0, 120, 0)
ANCHORS = [(0.0, DARK_RED), (0.5, ORANGE), (1.0, DARK_GREEN)]

RNG = np.random.default_rng()
NEAR_INDEX = trained_model.CLASS_NAMES.index("near person")


# ---------------------------------------------------------------- the platform

def platform_module():
    """The module with this platform's camera and window. Imported only when needed, so that
    importing this module loads neither AppKit nor Qt."""
    if sys.platform == "darwin":
        import webcam_macos as module
    elif sys.platform == "linux":
        import webcam_linux as module  # WSL 2 included
    else:
        sys.exit(f"the webcam mode of client.py runs on macOS, Linux and WSL 2, not on {sys.platform}; "
                 "--samples and --images run anywhere")
    return module


def ensure_camera_access():
    """Exit with what to do if the camera cannot be used, before the long key upload."""
    platform_module().ensure_camera_access()


def run(client, classify):
    """Open the window and classify the webcam until the window closes."""
    platform_module().run(client, classify)


# ---------------------------------------------------------------- frames and the link

class FrameHub:
    """Holds the newest frame for the link, numbered with a counter that keeps increasing across capture sessions."""

    def __init__(self):
        self.cond = threading.Condition()
        self.number = 0
        self.frame = None
        self.last_number = 0

    def publish(self, make_frame):
        """Number a new frame and make it with make_frame(id): a platform's Frame, with an
        image() that returns it as a PIL image (ValueError if it cannot be converted)."""
        with self.cond:
            self.last_number += 1
            number = self.last_number
        frame = make_frame(str(number))
        with self.cond:
            # A frame built after a newer one was published must not replace it.
            if number > self.number:
                self.number, self.frame = number, frame
                self.cond.notify_all()

    def newer_than(self, number, timeout):
        """Return (number, frame) of the newest frame if it is newer than number, else (number, None)."""
        with self.cond:
            if self.cond.wait_for(lambda: self.number > number, timeout):
                return self.number, self.frame
            return number, None


class Link:
    """Classifies the hub's newest frame with the prepared Belfort ML client, one at a time,
    until stopped.

    Records the last frame sent and, by frame id, the verdicts with their round trips, for
    the animation: a frame can still be on its way there while later ones are classified.
    A frame that cannot be converted counts as sent only."""

    KEEP = 256  # verdicts kept

    def __init__(self, client, classify, hub):
        self.client = client            # the prepared Belfort ML client
        self.classify_image = classify  # (client, prepared image) -> Verdict
        self.hub = hub
        self.state = "ready"
        self.classified = 0
        self.last = None        # (Verdict, round trip ms)
        self.sent = None        # Frame
        self.verdicts = {}      # frame id -> (Verdict, round trip ms), oldest first
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, name="belfort-ml-client", daemon=True)
        self.thread.start()

    def run(self):
        number, last_refusal = 0, None
        while not self.stopping.is_set():
            number, frame = self.hub.newer_than(number, 0.5)
            if frame is None:
                continue
            sent = time.monotonic()
            self.sent = frame
            try:
                values = trained_model.preprocess(frame.image())
            except ValueError as exc:
                if str(exc) != last_refusal:
                    print(f"frame {frame.id} skipped: {exc}", flush=True)
                    last_refusal = str(exc)
                continue
            # Encrypted with the secret key, classified on the deploy, decrypted with the secret key.
            try:
                verdict = self.classify_image(self.client, values)
            except ClientError as exc:
                self.state = f"deploy unreachable ({exc})"
                print(f"Belfort ML client: {exc}; retrying in {RETRY_S:.0f} s", flush=True)
                self.stopping.wait(RETRY_S)
                continue
            self.state = "ready"
            self.classified += 1
            self.last = (verdict, (time.monotonic() - sent) * 1000)
            self.verdicts[frame.id] = self.last
            while len(self.verdicts) > self.KEEP:
                del self.verdicts[next(iter(self.verdicts))]

    def stop(self):
        self.stopping.set()
        # A classification can take long; the thread is a daemon, so it does not keep the process alive.
        self.thread.join(timeout=2)

    def summary(self):
        """The count, the left of the status line; while the deploy is unreachable, that instead."""
        return f"{self.classified} classified" if self.state == "ready" else self.state

    def last_result(self):
        """The last result, the right of the status line: its class, both scores, the round trip (from
        taking the frame to its verdict: preprocess, encrypt, the deploy, decrypt) and the FHE evaluation
        (the verdict's latency_ms: the gateway's call to the GPU runtime)."""
        if self.state != "ready" or self.last is None:
            return ""
        verdict, round_trip_ms = self.last
        scores = " ".join(f"{s:.2f}" for s in verdict.scores)
        return (f"last: {verdict.predicted_name} [{scores}], round trip in {round_trip_ms:.0f} ms "
                f"and FHE evaluation in {verdict.latency_ms:.0f} ms")


# ---------------------------------------------------------------- drawing helpers

def aspect(width, height):
    g = math.gcd(width, height)
    return f"{width // g}:{height // g} ({width / height:.2f})"


def centre_square(pixels, max_px=PIP_MAX_PX):
    """The centre square of an H x W x C image, S x S x C with S a multiple of NOISE_BLOCKS,
    decimated so the image is at most max_px wide."""
    step = max(1, math.ceil(pixels.shape[1] / max_px))
    pixels = pixels[::step, ::step]
    h, w = pixels.shape[:2]
    side = max(NOISE_BLOCKS, min(h, w) // NOISE_BLOCKS * NOISE_BLOCKS)
    top, left = max(0, (h - side) // 2), max(0, (w - side) // 2)
    square = pixels[top:top + side, left:left + side]
    if square.shape[0] != side or square.shape[1] != side:  # an image smaller than the noise grid
        reps = math.ceil(side / min(square.shape[:2]))
        square = np.repeat(np.repeat(square, reps, 0), reps, 1)[:side, :side]
    return np.ascontiguousarray(square)


def box_colour(p):
    """Piecewise-linear blend through ANCHORS, in RGB."""
    if p <= ANCHORS[0][0]:
        return ANCHORS[0][1]
    for (p0, c0), (p1, c1) in zip(ANCHORS, ANCHORS[1:]):
        if p <= p1:
            t = (p - p0) / (p1 - p0)
            return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))
    return ANCHORS[-1][1]


def p_near(scores, near_index=NEAR_INDEX):
    """Softmax probability of the near-person class from the decrypted scores."""
    if not 0 <= near_index < len(scores):
        return float("nan")
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    return exps[near_index] / sum(exps)
