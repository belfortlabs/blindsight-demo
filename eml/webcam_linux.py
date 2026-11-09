"""The webcam mode of client.py on Linux, WSL 2 included: the camera through V4L2 by OpenCV,
the window through Qt, which WSLg shows on the Windows desktop. webcam.py picks this module on
Linux, and holds what does not depend on the platform: the newest-frame hub, the link that
classifies frames, and the animation described there.

The window matches webcam_macos.py's: pick a camera and a format in the menus; the status bar shows
the count of classified frames on its left and the last verdict on its right. On WSL 2 the camera must first be attached to Linux with
usbipd (see the repository README).
"""

import fcntl
import glob
import itertools
import math
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import (QColor, QFontDatabase, QImage, QKeySequence, QPainter, QPainterPath,
                           QPalette, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGraphicsObject, QGraphicsPixmapItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsView, QLabel, QMainWindow, QToolBar)

from webcam import (BOX_PX, BUSY_DASH, BUSY_GAP, BUSY_LINE, BUSY_PERIOD_S, DECRYPT_S, ENCRYPT_S,
                    FADE_S, GREY, IDLE_BORDER, LOGO, LOGO_ASPECT, LOGO_FRACTION, LOGO_MIN, LOGO_PAD, LOGO_PLATE,
                    NOISE_BLOCKS, PIP_FRACTION, PIP_MARGIN, PIP_MAX_WIDTH, RNG, SNAP_S, TO_MIDDLE_S, TO_RIGHT_S,
                    Z_FLOW, Z_LEAVING, Z_RESULT, FrameHub, Link, aspect, box_colour, centre_square, p_near)

VIEW_W, VIEW_H = 960, 540
DEFAULT_MAX_WIDTH = 640  # the format picked first: the widest MJPG size up to this
CAMERA_TIMEOUT_S = 15    # a camera open or frame read that takes longer is stuck in the kernel
TICK_MS = 16
CORNER_RADIUS = 8
IDLE_GREY = (160, 160, 160)  # the border of a frame on its way to the deploy


# ---------------------------------------------------------------- V4L2

# ioctl numbers and struct layouts from linux/videodev2.h
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xC0405602
VIDIOC_ENUM_FRAMESIZES = 0xC02C564A
VIDIOC_ENUM_FRAMEINTERVALS = 0xC034564B
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_DEVICE_CAPS = 0x80000000
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FRMSIZE_TYPE_DISCRETE = 1
V4L2_FRMIVAL_TYPE_DISCRETE = 1

DECODABLE = ("MJPG", "YUYV")  # the formats OpenCV turns into BGR, in menu order


@dataclass
class Camera:
    path: str
    name: str


def fourcc(code):
    return code.to_bytes(4, "little").decode("ascii", "replace")


def fourcc_code(text):
    return int.from_bytes(text.encode("ascii"), "little")


def video_nodes():
    def number(path):
        suffix = path.removeprefix("/dev/video")
        return int(suffix) if suffix.isdigit() else math.inf
    return sorted(glob.glob("/dev/video*"), key=number)


def query_camera(path):
    """The camera at path, or None if the node does not capture video (a UVC metadata node)."""
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        cap = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
    finally:
        os.close(fd)
    card = bytes(cap[16:48]).split(b"\0")[0].decode("utf-8", "replace")
    capabilities, device_caps = struct.unpack_from("<II", cap, 84)
    caps = device_caps if capabilities & V4L2_CAP_DEVICE_CAPS else capabilities
    return Camera(path, card or path) if caps & V4L2_CAP_VIDEO_CAPTURE else None


def list_devices():
    """(cameras, errors): the capture devices, and why other video nodes could not be opened."""
    cameras, errors = [], []
    for path in video_nodes():
        try:
            camera = query_camera(path)
        except OSError as exc:
            errors.append(f"{path}: {exc.strerror}")
            continue
        if camera is not None:
            cameras.append(camera)
    names = [c.name for c in cameras]
    for c in cameras:
        if names.count(c.name) > 1:
            c.name = f"{c.name} ({c.path})"
    return cameras, errors


def max_fps(fd, code, width, height):
    best = 0.0
    for i in itertools.count():
        buf = bytearray(struct.pack("<5I32x", i, code, width, height, 0))
        try:
            fcntl.ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, buf)
        except OSError:
            break
        kind, numerator, denominator = struct.unpack_from("<3I", buf, 16)
        if numerator:
            best = max(best, denominator / numerator)
        if kind != V4L2_FRMIVAL_TYPE_DISCRETE:  # stepwise: the one entry starts with the shortest interval
            break
    return best


def device_formats(path):
    """Map (fourcc, width, height) -> max fps for every discrete size in a format OpenCV decodes."""
    formats = {}
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        for i in itertools.count():
            desc = bytearray(struct.pack("<3I32s2I12x", i, V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, b"", 0, 0))
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FMT, desc)
            except OSError:
                break
            code = struct.unpack_from("<I", desc, 44)[0]
            if fourcc(code) not in DECODABLE:
                continue
            for j in itertools.count():
                size = bytearray(struct.pack("<3I32x", j, code, 0))
                try:
                    fcntl.ioctl(fd, VIDIOC_ENUM_FRAMESIZES, size)
                except OSError:
                    break
                kind, width, height = struct.unpack_from("<3I", size, 8)
                if kind != V4L2_FRMSIZE_TYPE_DISCRETE:
                    break
                formats[(fourcc(code), width, height)] = max_fps(fd, code, width, height)
    finally:
        os.close(fd)
    order = list(DECODABLE)
    return dict(sorted(formats.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1], kv[0][2])))


def format_list(formats):
    """The (fourcc, width, height) entries of the format menu. Without a size list, the driver picks its default size."""
    return list(formats) or [(code, 0, 0) for code in DECODABLE]


def default_format(formats):
    """The index of the widest MJPG size up to DEFAULT_MAX_WIDTH: larger or uncompressed formats arrive corrupted over usbip on WSL 2."""
    candidates = [i for i, (code, w, _) in enumerate(formats) if code == "MJPG" and w <= DEFAULT_MAX_WIDTH]
    if candidates:
        return max(candidates, key=lambda i: formats[i][1])
    return 0


def open_capture(path, code, width, height):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def in_wsl():
    return "WSL_DISTRO_NAME" in os.environ or "microsoft" in os.uname().release.lower()


def stuck_text():
    text = (f"the camera did not answer in {CAMERA_TIMEOUT_S} s: it is stuck in the kernel, "
            f"and no process that uses it can be stopped.")
    if in_wsl():
        return text + " Run `wsl --shutdown` in PowerShell, then attach the webcam again."
    return text + " Unplug the camera and plug it in again."


def ensure_camera_access():
    """Exit with what to do if there is no display or no camera that delivers a frame, before the
    long key upload. The camera is tried in a child process: a camera stuck in the kernel (as
    usbip on WSL 2 can be) leaves only the child stuck, and this process can still exit."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("the webcam mode needs a display for its window; DISPLAY is not set "
                         "(on WSL 2, WSLg sets it)")
    probe = subprocess.Popen([sys.executable, __file__], stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, text=True)
    try:
        _, errors = probe.communicate(timeout=CAMERA_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        probe.kill()
        raise SystemExit(stuck_text())
    if probe.returncode:
        raise SystemExit(errors.strip())


def check_camera():
    """Exit with what to do if the first camera does not deliver a frame in its default format."""
    cameras, errors = list_devices()
    if not cameras:
        if errors:
            raise SystemExit("cannot open the video devices: " + "; ".join(errors) +
                             "\nsee INSTALL.md, 'Webcam on WSL 2', for the Permission denied fixes")
        text = "no camera found: no /dev/video* device captures video"
        if in_wsl():
            text += ("\nOn WSL 2, attach the webcam with usbipd from an administrator PowerShell on Windows:\n"
                     "    usbipd list\n"
                     "    usbipd bind --busid <BUSID>\n"
                     "    usbipd attach --wsl --busid <BUSID>\n"
                     "then check that /dev/video0 exists (see INSTALL.md, 'Webcam on WSL 2')")
        raise SystemExit(text)
    path = cameras[0].path
    formats = format_list(device_formats(path))
    try:
        cap = open_capture(path, *formats[default_format(formats)])
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{path} delivers no frames")


# ---------------------------------------------------------------- frames and the capture

@dataclass
class Frame:
    id: str
    rgb: np.ndarray        # H x W x 3, as captured

    def image(self):
        return Image.fromarray(self.rgb)


class Capture:
    """Reads one camera format in a thread and publishes every frame, as RGB, to the hub."""

    def __init__(self, path, code, width, height, hub):
        self.format = (path, code, width, height)
        self.hub = hub
        self.lock = threading.Lock()
        self.latest = None     # the newest frame, RGB
        self.error = None
        self.waiting_since = time.monotonic()  # while the camera is opened or a frame is read, else None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, name="camera", daemon=True)
        self.thread.start()

    def run(self):
        try:
            cap = open_capture(*self.format)
        except RuntimeError as exc:
            self.error = str(exc)
            self.waiting_since = None
            return
        while not self.stopping.is_set():
            self.waiting_since = time.monotonic()
            ok, bgr = cap.read()
            self.waiting_since = None
            if not ok or bgr is None:
                self.error = "the camera delivers no frames"
                self.stopping.wait(0.1)
                continue
            self.error = None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.hub.publish(lambda id: Frame(id, rgb))
            with self.lock:
                self.latest = rgb
        cap.release()

    def stuck(self):
        since = self.waiting_since
        return since is not None and time.monotonic() - since > CAMERA_TIMEOUT_S

    def frame(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=2)


# ---------------------------------------------------------------- drawing

def to_pixmap(rgb):
    h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb)
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def qcolor(rgb):
    return QColor(*(int(c) for c in rgb))


class Slot(QGraphicsObject):
    """One thumbnail: the frame it shows, how much of it is encrypted and its box. Its
    rect is the picture's; the box is drawn outside it, not over its edge, so the result's box is as
    big as the turning box in the middle."""

    def __init__(self):
        super().__init__()
        self.frame = None
        self.clear = None        # the centre square, S x S x 3
        self.noise = None        # S x S x 3: NOISE_BLOCKS x NOISE_BLOCKS blocks of random colour
        self.order = None        # the order in which blocks turn into noise
        self.pixmap = None
        self.encrypted = -1
        self.phase = None
        self.start = self.end = 0.0
        self.verdict = None      # (Verdict, round trip ms), once classified
        self.border = IDLE_BORDER
        self._rect = QRectF()
        self._colour = qcolor(IDLE_GREY)
        self.animations = {}

    @property
    def frame_id(self):
        return None if self.frame is None else self.frame.id

    def load(self, frame):
        """Take the frame's centre square; new noise only for a new frame."""
        clear = centre_square(frame.rgb)
        if self.frame is None or self.frame.id != frame.id or self.noise is None:
            k = clear.shape[0] // NOISE_BLOCKS
            blocks = RNG.integers(0, 256, (NOISE_BLOCKS, NOISE_BLOCKS, 3), dtype=np.uint8)
            self.noise = np.repeat(np.repeat(blocks, k, 0), k, 1)
            self.order = RNG.permutation(NOISE_BLOCKS * NOISE_BLOCKS)
            self.encrypted = -1
        self.frame, self.clear = frame, clear
        self.draw(max(self.encrypted, 0), force=True)

    def draw(self, blocks, force=False):
        """Show the picture with this many of its blocks turned into noise."""
        blocks = int(min(max(blocks, 0), NOISE_BLOCKS * NOISE_BLOCKS))
        if blocks == self.encrypted and not force:
            return
        self.encrypted = blocks
        if blocks == 0:
            image = self.clear
        elif blocks == NOISE_BLOCKS * NOISE_BLOCKS:
            image = self.noise
        else:
            mask = np.zeros(NOISE_BLOCKS * NOISE_BLOCKS, dtype=bool)
            mask[self.order[:blocks]] = True
            k = self.clear.shape[0] // NOISE_BLOCKS
            mask = np.repeat(np.repeat(mask.reshape(NOISE_BLOCKS, NOISE_BLOCKS), k, 0), k, 1)
            image = np.where(mask[:, :, None], self.noise, self.clear)
        self.pixmap = to_pixmap(image)
        self.update()

    def encrypt_to(self, fraction):
        self.draw(round(fraction * NOISE_BLOCKS * NOISE_BLOCKS))

    def get_rect(self):
        return self._rect

    def set_rect(self, rect):
        self.prepareGeometryChange()
        self._rect = QRectF(rect)

    def get_colour(self):
        return self._colour

    def set_colour(self, colour):
        self._colour = QColor(colour)
        self.update()

    rect = Property(QRectF, get_rect, set_rect)
    colour = Property(QColor, get_colour, set_colour)

    def animate(self, name, end, seconds=0.0):
        """Set a property now, or move it to end over seconds, eased in and out."""
        running = self.animations.pop(name, None)
        if running is not None:
            running.stop()
        if seconds <= 0:
            self.setProperty(name, end)
            return
        animation = QPropertyAnimation(self, name.encode())
        animation.setDuration(int(seconds * 1000))
        animation.setStartValue(self.property(name))
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        animation.start()
        self.animations[name] = animation

    def target(self, name):
        """Where a property is, or is moving to."""
        running = self.animations.get(name)
        if running is not None and running.state() == QPropertyAnimation.State.Running:
            return running.endValue()
        return self.property(name)

    def boundingRect(self):
        m = BOX_PX + 1  # the widest box, outside the picture
        return self._rect.adjusted(-m, -m, m, m)

    def paint(self, painter, option, widget=None):
        r = self._rect
        if self.pixmap is None or r.width() < 4:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Noise stays blocky; the clear picture is scaled smoothly.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.encrypted == 0)
        outline = QPainterPath()
        outline.addRoundedRect(r, CORNER_RADIUS, CORNER_RADIUS)
        painter.save()
        painter.setClipPath(outline)
        painter.drawPixmap(r, self.pixmap, QRectF(self.pixmap.rect()))
        painter.restore()
        b = self.border  # outside the picture: the pen's centre line half its width out
        painter.setPen(QPen(self._colour, b))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(r.adjusted(-b / 2, -b / 2, b / 2, b / 2), CORNER_RADIUS + b / 2, CORNER_RADIUS + b / 2)


class BusyBox(QGraphicsObject):
    """The turning dashed box around the frame at the deploy."""

    def __init__(self):
        super().__init__()
        self._rect = QRectF()
        self._phase = 0.0
        self.setZValue(10)
        self.setVisible(False)
        period = sum(BUSY_DASH) / BUSY_LINE  # QPen dashes are in units of the line width
        self.march = QVariantAnimation()
        self.march.setStartValue(0.0)
        self.march.setEndValue(period)
        self.march.setDuration(int(BUSY_PERIOD_S * 1000))
        self.march.setLoopCount(-1)
        self.march.valueChanged.connect(self.set_phase)
        self.march.start()

    def place(self, rect):
        if rect != self._rect:
            self.prepareGeometryChange()
            self._rect = rect

    def set_phase(self, phase):
        self._phase = phase
        if self.isVisible():
            self.update()

    def boundingRect(self):
        return self._rect.adjusted(-BUSY_LINE, -BUSY_LINE, BUSY_LINE, BUSY_LINE)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(255, 255, 255, 220), BUSY_LINE)
        pen.setDashPattern([d / BUSY_LINE for d in BUSY_DASH])
        pen.setDashOffset(-self._phase)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self._rect, CORNER_RADIUS + BUSY_GAP, CORNER_RADIUS + BUSY_GAP)


# ---------------------------------------------------------------- the window

class VideoView(QGraphicsView):
    """The camera picture and the thumbnails, scaled to the view."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor(0, 0, 0))
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMinimumSize(480, 270)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scene().setSceneRect(QRectF(0, 0, self.viewport().width(), self.viewport().height()))

    def sizeHint(self):
        return QSize(VIEW_W, VIEW_H)


class Viewer(QMainWindow):
    """The window: the live camera with its menus and status bar, and the trip of each frame sent."""

    def __init__(self, hub, link):
        super().__init__()
        self.hub = hub
        self.link = link
        self.cameras, _ = list_devices()
        self.formats = []
        self.capture = None
        self.shown = None        # the frame shown, to skip unchanged ticks
        self.leaving = []        # (slot, time it is out of sight)
        self.flow = None         # the frame on its way: snap, encrypt, to the middle, waiting
        self.right = None        # the last result: to the right, decrypt, shown
        self.last_snapped = -1
        self.build_window()
        self.fill_devices()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)

    def build_window(self):
        self.setWindowTitle("BlindSight")
        bar = QToolBar()
        bar.setMovable(False)
        self.addToolBar(bar)
        self.device_menu = QComboBox()
        self.device_menu.setMinimumContentsLength(24)
        self.device_menu.activated.connect(self.device_changed)
        self.format_menu = QComboBox()
        self.format_menu.setMinimumContentsLength(28)
        self.format_menu.activated.connect(self.format_changed)
        for label, widget in (("Camera", self.device_menu), ("Format", self.format_menu)):
            bar.addWidget(QLabel(f"  {label}  "))
            bar.addWidget(widget)

        self.scene = QGraphicsScene(self)
        self.video = QGraphicsPixmapItem()
        self.video.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.video)
        self.busy = BusyBox()
        self.scene.addItem(self.busy)
        self.plate = QGraphicsRectItem()
        self.plate.setBrush(qcolor(LOGO_PLATE))
        self.plate.setPen(QPen(Qt.PenStyle.NoPen))
        self.plate.setZValue(20)      # above the thumbnails and the turning box
        self.plate.setVisible(False)  # until there is a picture to put it on
        self.logo = QGraphicsPixmapItem(QPixmap(str(LOGO)), self.plate)
        self.logo.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.plate)
        self.view = VideoView(self.scene)
        self.setCentralWidget(self.view)

        self.status = QLabel()
        self.status.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.result_status = QLabel()
        self.result_status.setFont(self.status.font())
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.result_status)

    def fill_devices(self):
        if not self.cameras:
            self.device_menu.addItem("No cameras found")
            self.device_menu.setEnabled(False)
            self.format_menu.setEnabled(False)
            return
        self.device_menu.addItems([c.name for c in self.cameras])
        self.device_changed()

    def device_changed(self, *_):
        camera = self.cameras[self.device_menu.currentIndex()]
        try:
            formats = device_formats(camera.path)
        except OSError as exc:
            formats = {}
            self.status.setText(f"{camera.path}: {exc.strerror}")
        self.formats = format_list(formats)
        self.format_menu.clear()
        self.format_menu.addItems([f"{code}  {w}x{h}  {aspect(w, h)}  {formats[(code, w, h)]:g} fps" if w
                                   else f"{code}  default size" for code, w, h in self.formats])
        self.format_menu.setCurrentIndex(default_format(self.formats))
        self.format_changed()

    def format_changed(self, *_):
        self.stop()
        camera = self.cameras[self.device_menu.currentIndex()]
        code, w, h = self.formats[self.format_menu.currentIndex()]
        self.capture = Capture(camera.path, code, w, h, self.hub)

    def stop(self):
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        for slot in self.slots():
            self.scene.removeItem(slot)
        self.flow = self.right = None
        self.leaving = []
        self.busy.setVisible(False)
        self.shown = None

    def closeEvent(self, event):
        QApplication.instance().quit()
        event.accept()

    def slots(self):
        return [s for s in [self.flow, self.right] + [s for s, _ in self.leaving] if s is not None]

    def tick(self):
        if self.capture is not None and self.capture.stuck():
            self.status.setText(stuck_text())
            return
        rgb = self.capture.frame() if self.capture is not None else None
        if rgb is None:
            if self.capture is not None and self.capture.error:
                self.status.setText(self.capture.error)
            return
        now = time.monotonic()
        image_h, image_w = rgb.shape[:2]
        if rgb is not self.shown:
            self.shown = rgb
            self.video.setPixmap(to_pixmap(rgb))
        x, y, w, h = self.picture_rect(image_w, image_h)
        self.video.setScale(w / image_w)
        self.video.setPos(x, y)
        self.update_pips(rgb, now)
        self.update_logo(image_w, image_h)
        self.status.setText(self.link.summary())
        self.result_status.setText(self.link.last_result())

    # ------------------------------------------------------------ geometry

    def picture_rect(self, image_w, image_h):
        """Where the camera picture is shown: aspect-fit in the view, as x, y, w, h."""
        view_w, view_h = max(self.view.viewport().width(), 1), max(self.view.viewport().height(), 1)
        fit = min(view_w / image_w, view_h / image_h)
        w, h = image_w * fit, image_h * fit
        return (view_w - w) / 2, (view_h - h) / 2, w, h

    def crop_rect(self, image_w, image_h):
        """Where the frame's centre square is shown in the picture."""
        x, y, w, h = self.picture_rect(image_w, image_h)
        side = min(w, h)
        return QRectF(x + (w - side) / 2, y + (h - side) / 2, side, side)

    def geometry_for(self, image_w, image_h):
        """The left, middle, right and off-right squares along the bottom of the picture."""
        left, top, shown_w, shown_h = self.picture_rect(image_w, image_h)
        side = min(shown_h * PIP_FRACTION, shown_w * PIP_MAX_WIDTH)
        y = top + shown_h - PIP_MARGIN - side
        return {
            "left": QRectF(left + PIP_MARGIN, y, side, side),
            "middle": QRectF(left + (shown_w - side) / 2, y, side, side),
            "right": QRectF(left + shown_w - PIP_MARGIN - side, y, side, side),
            "off_right": QRectF(self.view.viewport().width(), y, side, side),
        }

    # ------------------------------------------------------------ every tick

    def update_logo(self, image_w, image_h):
        """The logo in the picture's top left corner, sized by the picture's height: the same share of it for a
        square format and for 16:9."""
        left, top, _, shown_h = self.picture_rect(image_w, image_h)
        w = max(shown_h * LOGO_FRACTION, LOGO_MIN)
        h = w / LOGO_ASPECT
        pad = h * LOGO_PAD
        self.plate.setRect(0, 0, w + 2 * pad, h + 2 * pad)
        self.plate.setPos(left + PIP_MARGIN, top + PIP_MARGIN)
        self.logo.setScale(w / max(self.logo.pixmap().width(), 1))
        self.logo.setPos(pad, pad)
        self.plate.setVisible(True)

    def update_pips(self, rgb, now):
        image_h, image_w = rgb.shape[:2]
        where = self.geometry_for(image_w, image_h)
        for slot, gone in list(self.leaving):
            if now >= gone:
                self.leaving.remove((slot, gone))
                self.scene.removeItem(slot)
        self.advance_flow(where, now)
        self.advance_right(now)
        if self.flow is None:
            sent = self.link.sent
            if sent is not None and int(sent.id) > self.last_snapped:
                self.snap(sent, self.crop_rect(image_w, image_h), where, now)
        self.follow_resize(where)
        waiting = self.flow is not None and self.flow.phase == "waiting"
        if waiting:
            g = BUSY_GAP + BUSY_LINE / 2
            self.busy.place(where["middle"].adjusted(-g, -g, g, g))
        self.busy.setVisible(waiting)

    def advance_flow(self, where, now):
        slot = self.flow
        if slot is None:
            return
        if slot.phase == "snap" and now >= slot.end:
            self.set_phase(slot, "encrypt", now, ENCRYPT_S)
        if slot.phase == "encrypt":
            slot.encrypt_to(min((now - slot.start) / ENCRYPT_S, 1.0))
            if now >= slot.end:
                self.set_phase(slot, "to_middle", now, TO_MIDDLE_S)
                slot.animate("rect", where["middle"], TO_MIDDLE_S)
        if slot.phase == "to_middle" and now >= slot.end:
            self.set_phase(slot, "waiting", now, 0.0)
        if slot.phase == "waiting":
            verdict = self.link.verdicts.get(slot.frame_id)
            sent = self.link.sent
            if verdict is not None:
                self.to_right(slot, verdict, where, now)
            elif sent is not None and int(sent.id) > int(slot.frame_id):  # no answer: the link moved on
                slot.animate("opacity", 0.0, FADE_S)
                self.leaving.append((slot, now + FADE_S))
                self.flow = None

    def advance_right(self, now):
        slot = self.right
        if slot is None:
            return
        if slot.phase == "to_right" and now >= slot.end:
            self.set_phase(slot, "decrypt", now, DECRYPT_S)
            slot.animate("colour", self.final_colour(slot), DECRYPT_S)
        if slot.phase == "decrypt":
            slot.encrypt_to(1.0 - min((now - slot.start) / DECRYPT_S, 1.0))
            if now >= slot.end:
                self.finish(slot)

    @staticmethod
    def set_phase(slot, phase, now, duration):
        slot.phase, slot.start, slot.end = phase, now, now + duration

    @staticmethod
    def final_colour(slot):
        verdict, _ = slot.verdict
        p = p_near(list(verdict.scores))
        return qcolor(box_colour(p) if p == p else GREY)  # p != p: NaN

    def finish(self, slot):
        """Show a result decrypted, with its colour, at once if need be."""
        slot.animate("colour", self.final_colour(slot))
        slot.encrypt_to(0.0)
        slot.phase = "shown"

    def follow_resize(self, where):
        """Keep resting thumbnails in place when the window changes size."""
        for slot, rest in ((self.flow, ("encrypt", "waiting")), (self.right, ("decrypt", "shown"))):
            if slot is not None and slot.phase in rest:
                spot = where["left" if slot.phase == "encrypt" else
                             "middle" if slot.phase == "waiting" else "right"]
                if slot.target("rect") != spot:
                    slot.animate("rect", spot)

    # ------------------------------------------------------------ transitions

    def snap(self, frame, crop, where, now):
        """Snap the frame's centre square from the live picture into the lower left."""
        slot = Slot()
        self.last_snapped = int(frame.id)
        slot.load(frame)
        slot.setZValue(Z_FLOW)
        slot.animate("rect", crop)
        slot.animate("rect", where["left"], SNAP_S)
        self.scene.addItem(slot)
        slot.phase, slot.start, slot.end = "snap", now, now + SNAP_S
        self.flow = slot

    def to_right(self, slot, verdict, where, now):
        """The verdict is in: the encrypted result takes a grey box and slides to the right; the
        previous result, decrypted at once if it was not yet, slides out."""
        previous = self.right
        if previous is not None:
            if previous.phase != "shown":
                self.finish(previous)
            previous.setZValue(Z_LEAVING)
            previous.animate("rect", where["off_right"], TO_RIGHT_S)
            previous.animate("opacity", 0.0, TO_RIGHT_S)
            self.leaving.append((previous, now + TO_RIGHT_S))
        slot.verdict = verdict
        slot.border = BOX_PX
        slot.animate("colour", qcolor(GREY))
        slot.setZValue(Z_RESULT)
        slot.animate("rect", where["right"], TO_RIGHT_S)
        self.set_phase(slot, "to_right", now, TO_RIGHT_S)
        self.right, self.flow = slot, None


def dark_palette():
    palette = QPalette()
    for role, rgb in ((QPalette.ColorRole.Window, (32, 33, 36)), (QPalette.ColorRole.WindowText, (232, 234, 237)),
                      (QPalette.ColorRole.Base, (24, 25, 28)), (QPalette.ColorRole.AlternateBase, (40, 41, 45)),
                      (QPalette.ColorRole.Text, (232, 234, 237)), (QPalette.ColorRole.Button, (48, 49, 54)),
                      (QPalette.ColorRole.ButtonText, (232, 234, 237)), (QPalette.ColorRole.Highlight, (66, 133, 244)),
                      (QPalette.ColorRole.HighlightedText, (255, 255, 255)), (QPalette.ColorRole.ToolTipBase, (48, 49, 54)),
                      (QPalette.ColorRole.ToolTipText, (232, 234, 237))):
        palette.setColor(role, QColor(*rgb))
    return palette


def run(client, classify):
    """Open the window and classify the webcam until the window closes. client.py passes the
    prepared Belfort ML client and how to classify one prepared image with it."""
    # Wayland first: under WSLg, X11 windows get WSLg's dated title bar instead of adwaita's.
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland;xcb" if os.environ.get("WAYLAND_DISPLAY") else "xcb")
    os.environ.setdefault("QT_WAYLAND_DECORATION", "adwaita")
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    hub = FrameHub()
    link = Link(client, classify, hub)
    viewer = Viewer(hub, link)
    viewer.resize(VIEW_W, VIEW_H + 80)
    QShortcut(QKeySequence("Ctrl+Q"), viewer, activated=app.quit)
    # Python runs signal handlers only between its own bytecodes; the timer lets it run some.
    previous = signal.signal(signal.SIGINT, lambda *_: app.quit())
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)
    viewer.show()
    print("Window open. To stop: close the window, press Ctrl-Q in it, or press Ctrl-C here.", flush=True)
    try:
        app.exec()
    finally:
        signal.signal(signal.SIGINT, previous)
        viewer.timer.stop()
        viewer.stop()
        link.stop()
    print(f"Stopped after {link.classified} classified image(s).", flush=True)


if __name__ == "__main__":  # the camera probe of ensure_camera_access()
    check_camera()
