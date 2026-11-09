"""The webcam mode of client.py on macOS: the camera through AVFoundation, the window through
AppKit and Core Animation. webcam.py picks this module on macOS, and holds what does not
depend on the platform: the newest-frame hub, the link that classifies frames, and the
animation described there.

Frames are decoded from the raw YUV buffers to RGB here rather than by CoreVideo, so what is
shown is what the format delivers. Each thumbnail is decoded from the exact YUV bytes that
were sent for classification.
"""

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import objc
import AppKit
import AVFoundation as AVF
import CoreMedia as CM
import Quartz
import libdispatch
from Foundation import NSData, NSMakeRect, NSObject, NSTimer
from PyObjCTools import AppHelper

from PIL import Image

from webcam import (BOX_PX, BUSY_DASH, BUSY_GAP, BUSY_LINE, BUSY_PERIOD_S, DECRYPT_S, ENCRYPT_S,
                    FADE_S, GREY, IDLE_BORDER, LOGO, LOGO_ASPECT, LOGO_FRACTION, LOGO_MIN, LOGO_PAD, LOGO_PLATE,
                    NOISE_BLOCKS, PIP_FRACTION, PIP_MARGIN, PIP_MAX_WIDTH, RNG, SNAP_S, TO_MIDDLE_S, TO_RIGHT_S,
                    Z_FLOW, Z_LEAVING, Z_RESULT, FrameHub, Link, aspect, box_colour, centre_square, p_near)

# Window
BAR_HEIGHT = 40
STATUS_HEIGHT = 24
BUSY_RGBA = (0, 0, 0, 1)  # the turning box around the frame at the deploy


# ---------------------------------------------------------------- YUV formats, in numpy only

# fourcc -> (description, byte offset of Y within a packed 4:2:2 pair or None for NV12 planes, full range)
PIXEL_FORMATS = {
    "yuvs": ("YUYV 4:2:2", 0, False),
    "2vuy": ("UYVY 4:2:2", 1, False),
    "420v": ("NV12 4:2:0", None, False),
    "420f": ("NV12 4:2:0 full range", None, True),
}

# Kr, Kb luma coefficients per YCbCr matrix.
MATRICES = {
    "601": (0.299, 0.114),
    "709": (0.2126, 0.0722),
}


def frame_size(code, width, height):
    """Bytes of a tightly packed frame, as YuvImage.data holds it."""
    if PIXEL_FORMATS[code][1] is None:
        return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
    return 2 * width * height


def unpack(code, width, height, data):
    """Return the planes of tightly packed frame bytes: NV12 gives (Y, CbCr), 4:2:2 one plane."""
    raw = np.frombuffer(data, dtype=np.uint8)
    if PIXEL_FORMATS[code][1] is None:
        luma = width * height
        return raw[:luma].reshape(height, width), raw[luma:].reshape((height + 1) // 2, 2 * ((width + 1) // 2))
    return (raw.reshape(height, 2 * width),)


def split_planes(code, planes):
    """Return (Y, Cb, Cr) with the chroma planes upsampled to the luma size."""
    y_offset = PIXEL_FORMATS[code][1]
    if y_offset is None:
        y, uv = planes
        uv = uv.repeat(2, axis=0)[: y.shape[0]]
    else:
        (raw,) = planes
        y = raw[:, y_offset::2]
        uv = raw[:, 1 - y_offset :: 2]
    width = y.shape[1]
    cb = uv[:, 0::2].repeat(2, axis=1)[:, :width]
    cr = uv[:, 1::2].repeat(2, axis=1)[:, :width]
    return y, cb, cr


def to_rgbx(y, cb, cr, full_range, matrix):
    """Return H x W x 4 uint8 RGB with the fourth byte 255."""
    kr, kb = MATRICES[matrix]
    kg = 1 - kr - kb
    y = y.astype(np.float32)
    cb = cb.astype(np.float32) - 128
    cr = cr.astype(np.float32) - 128
    if not full_range:
        y = (y - 16) * (255 / 219)
        cb *= 255 / 224
        cr *= 255 / 224
    # Storing into uint8 truncates, so the half offset makes it round.
    y += 0.5
    out = np.empty(y.shape + (4,), dtype=np.uint8)
    out[..., 0] = np.clip(y + 2 * (1 - kr) * cr, 0, 255)
    out[..., 1] = np.clip(y - (2 * kb * (1 - kb) * cb + 2 * kr * (1 - kr) * cr) / kg, 0, 255)
    out[..., 2] = np.clip(y + 2 * (1 - kb) * cb, 0, 255)
    out[..., 3] = 255
    return out


def to_rgb(code, width, height, full_range, matrix, data):
    """Return tightly packed frame bytes as H x W x 3 uint8 RGB."""
    rgbx = to_rgbx(*split_planes(code, unpack(code, width, height, data)), full_range, matrix)
    return np.ascontiguousarray(rgbx[..., :3])


# ---------------------------------------------------------------- the camera (AVFoundation)

def fourcc(code):
    return code.to_bytes(4, "big").decode("ascii", "replace")


def fourcc_code(text):
    return int.from_bytes(text.encode("ascii"), "big")


def list_devices():
    device_types = [AVF.AVCaptureDeviceTypeExternal, AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    if hasattr(AVF, "AVCaptureDeviceTypeContinuityCamera"):
        device_types.append(AVF.AVCaptureDeviceTypeContinuityCamera)
    session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        device_types, AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified
    )
    return sorted(session.devices(), key=lambda d: (d.localizedName().lower(), d.uniqueID()))


def device_formats(device):
    """Map (fourcc, width, height) -> (format, max fps) for every supported format the device offers."""
    formats = {}
    for fmt in device.formats():
        desc = fmt.formatDescription()
        code = fourcc(CM.CMFormatDescriptionGetMediaSubType(desc))
        if code not in PIXEL_FORMATS:
            continue
        dims = CM.CMVideoFormatDescriptionGetDimensions(desc)
        fps = max(r.maxFrameRate() for r in fmt.videoSupportedFrameRateRanges())
        key = (code, dims.width, dims.height)
        if key not in formats or fps > formats[key][1]:
            formats[key] = (fmt, fps)
    return formats


def ensure_camera_access():
    status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    if status == AVF.AVAuthorizationStatusAuthorized:
        return
    if status == AVF.AVAuthorizationStatusNotDetermined:
        done = threading.Event()
        granted = []
        def handler(ok):
            granted.append(ok)
            done.set()
        AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVF.AVMediaTypeVideo, handler)
        done.wait()
        if granted[0]:
            return
    sys.exit("Camera access denied. Allow your terminal app in System Settings > Privacy & Security > Camera.")


def start_capture(device, code, fmt, width, height, delegate, queue):
    """Start a session delivering frames of the given format to delegate; raises RuntimeError."""
    session = AVF.AVCaptureSession.alloc().init()
    device_input, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
    if device_input is None:
        raise RuntimeError(f"Cannot open {device.localizedName()}: {err}")

    output = AVF.AVCaptureVideoDataOutput.alloc().init()
    output.setVideoSettings_({
        Quartz.kCVPixelBufferPixelFormatTypeKey: fourcc_code(code),
        Quartz.kCVPixelBufferWidthKey: width,
        Quartz.kCVPixelBufferHeightKey: height,
    })
    output.setAlwaysDiscardsLateVideoFrames_(True)
    output.setSampleBufferDelegate_queue_(delegate, queue)

    session.beginConfiguration()
    session.addInput_(device_input)
    session.addOutput_(output)
    ok, err = device.lockForConfiguration_(None)
    if not ok:
        raise RuntimeError(f"Cannot configure {device.localizedName()}: {err}")
    device.setActiveFormat_(fmt)
    fastest = max(fmt.videoSupportedFrameRateRanges(), key=lambda r: r.maxFrameRate())
    device.setActiveVideoMinFrameDuration_(fastest.minFrameDuration())
    session.commitConfiguration()
    session.startRunning()
    # Holding the lock until after startRunning keeps the session from replacing activeFormat.
    device.unlockForConfiguration()
    return session


# ---------------------------------------------------------------- frames

@dataclass
class YuvImage:
    """A camera frame as captured: rows tightly packed, without the capture buffer's stride padding."""
    fourcc: str            # yuvs (YUYV 4:2:2), 2vuy (UYVY 4:2:2), 420v or 420f (NV12 4:2:0)
    width: int
    height: int
    full_range: bool       # False: video range (Y 16..235)
    matrix: str            # the YCbCr matrix: "601" or "709"
    captured_ns: int       # wall clock (Unix ns) when the frame was captured
    data: bytes


@dataclass
class Frame:
    id: str                # the capture counter
    yuv: YuvImage

    def image(self):
        return yuv_image(self.yuv)


def yuv_image(image: YuvImage) -> Image.Image:
    """The frame as an RGB PIL image; ValueError if it is not a frame to_rgb() can convert."""
    if image.fourcc not in PIXEL_FORMATS:
        raise ValueError(f"frame has fourcc {image.fourcc!r}; known: {', '.join(PIXEL_FORMATS)}")
    if image.width == 0 or image.height == 0:
        raise ValueError(f"frame is {image.width}x{image.height}")
    size = frame_size(image.fourcc, image.width, image.height)
    if len(image.data) != size:
        raise ValueError(f"frame has {len(image.data)} bytes; {image.fourcc} {image.width}x{image.height} needs {size}")
    if image.matrix not in MATRICES:
        raise ValueError(f"frame has matrix {image.matrix!r}; known: {', '.join(MATRICES)}")
    return Image.fromarray(to_rgb(image.fourcc, image.width, image.height,
                                      image.full_range, image.matrix, image.data))


def ycbcr_matrix(pb):
    value, _ = Quartz.CVBufferCopyAttachment(pb, Quartz.kCVImageBufferYCbCrMatrixKey, None)
    return "709" if value == Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2 else "601"


def packed_planes(pb):
    """Return (fourcc, planes) as views of the locked buffer without stride padding: NV12 gives (Y, CbCr), 4:2:2 one plane."""
    code = fourcc(Quartz.CVPixelBufferGetPixelFormatType(pb))
    width = Quartz.CVPixelBufferGetWidth(pb)
    height = Quartz.CVPixelBufferGetHeight(pb)
    if PIXEL_FORMATS[code][1] is None:
        stride = Quartz.CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        base = Quartz.CVPixelBufferGetBaseAddressOfPlane(pb, 0)
        y = np.frombuffer(base.as_buffer(stride * height), dtype=np.uint8).reshape(height, stride)[:, :width]
        c_height = (height + 1) // 2
        stride = Quartz.CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
        base = Quartz.CVPixelBufferGetBaseAddressOfPlane(pb, 1)
        uv = np.frombuffer(base.as_buffer(stride * c_height), dtype=np.uint8).reshape(c_height, stride)
        return code, (y, uv[:, : 2 * ((width + 1) // 2)])
    stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
    base = Quartz.CVPixelBufferGetBaseAddress(pb)
    # UVC 4:2:2 rows arrive tightly packed even when CoreVideo reports a stride
    # padded to 64 bytes, so the reported stride skews any width where 2*width isn't aligned.
    row_bytes = 2 * width
    raw = np.frombuffer(base.as_buffer(stride * height), dtype=np.uint8)
    return code, (raw[: row_bytes * height].reshape(height, row_bytes),)


def to_cgimage(rgbx):
    h, w, _ = rgbx.shape
    provider = Quartz.CGDataProviderCreateWithCFData(NSData.dataWithBytes_length_(rgbx.tobytes(), rgbx.nbytes))
    return Quartz.CGImageCreate(
        w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB),
        Quartz.kCGImageAlphaNoneSkipLast, provider, None, False, Quartz.kCGRenderingIntentDefault,
    )


def frame_rgbx(frame):
    """The frame's YUV bytes as H x W x 4 RGBX."""
    image = frame.yuv
    planes = unpack(image.fourcc, image.width, image.height, image.data)
    return to_rgbx(*split_planes(image.fourcc, planes), image.full_range, image.matrix)


def cg_rgb(rgb, alpha=1.0):
    return Quartz.CGColorCreateGenericRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha)


# ---------------------------------------------------------------- the capture

class RGBGrabber(NSObject):
    def initWithHub_(self, hub):
        self = objc.super(RGBGrabber, self).init()
        self.hub = hub
        self.lock = threading.Lock()
        self.rgbx = None
        self.frame_id = 0
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
        captured_ns = time.time_ns()
        pb = CM.CMSampleBufferGetImageBuffer(sample_buffer)
        Quartz.CVPixelBufferLockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            code, planes = packed_planes(pb)
            data = b"".join(plane.tobytes() for plane in planes)
            matrix = ycbcr_matrix(pb)
            full_range = PIXEL_FORMATS[code][2]
            rgbx = to_rgbx(*split_planes(code, planes), full_range, matrix)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
        height, width, _ = rgbx.shape
        self.hub.publish(lambda id: Frame(id, YuvImage(fourcc=code, width=width, height=height,
                                                       full_range=full_range, matrix=matrix,
                                                       captured_ns=captured_ns, data=data)))
        with self.lock:
            self.rgbx = rgbx
            self.frame_id += 1

    def latest(self):
        with self.lock:
            return self.frame_id, self.rgbx


# ---------------------------------------------------------------- Core Animation helpers

def immediate():
    """Start a Core Animation transaction without implicit animations."""
    Quartz.CATransaction.begin()
    Quartz.CATransaction.setDisableActions_(True)


def animate(duration):
    """Start an animated Core Animation transaction of this duration, ease in/out."""
    Quartz.CATransaction.begin()
    Quartz.CATransaction.setAnimationDuration_(duration)
    Quartz.CATransaction.setAnimationTimingFunction_(
        Quartz.CAMediaTimingFunction.functionWithName_(Quartz.kCAMediaTimingFunctionEaseInEaseOut))


def basic(key, from_value, to_value, duration):
    """An explicit animation with a fixed start value.

    Implicit animations start from what was last drawn on screen, not from a value
    set earlier in the same tick: Core Animation renders only the last state of a
    tick. A recycled thumbnail was last drawn off-screen, so an implicit animation
    would start from there."""
    anim = Quartz.CABasicAnimation.animationWithKeyPath_(key)
    anim.setFromValue_(from_value)
    anim.setToValue_(to_value)
    anim.setDuration_(duration)
    anim.setTimingFunction_(
        Quartz.CAMediaTimingFunction.functionWithName_(Quartz.kCAMediaTimingFunctionEaseInEaseOut))
    return anim


def centre(rect):
    (x, y), (w, h) = rect
    return (x + w / 2, y + h / 2)


def fly(layer, start, end, duration):
    """Animate a layer's frame from start to end. Its model frame must already be end."""
    for key, a, b in (
            ("position", AppKit.NSValue.valueWithPoint_(centre(start)), AppKit.NSValue.valueWithPoint_(centre(end))),
            ("bounds", AppKit.NSValue.valueWithRect_(((0, 0), start[1])), AppKit.NSValue.valueWithRect_(((0, 0), end[1])))):
        layer.addAnimation_forKey_(basic(key, a, b, duration), "snap-" + key)


def outset(rect, border):
    """The frame of a thumbnail's layer whose picture is at rect: the border goes round it."""
    (x, y), (w, h) = rect
    return ((x - border, y - border), (w + 2 * border, h + 2 * border))


def inside(rect, border):
    """Where Slot.place puts the picture inside a thumbnail's layer."""
    return ((border, border), rect[1])


# ---------------------------------------------------------------- the window

class Slot:
    """One thumbnail: a square picture, the frame it shows, and how much
    of it is encrypted (blocks of noise over the picture). Its layer is a frame of its border's
    width round the picture, a sublayer, so the border goes outside the picture, not over its
    edge: the result's box is as big as the turning box in the middle."""

    def __init__(self, parent):
        self.layer = Quartz.CALayer.layer()
        self.picture = Quartz.CALayer.layer()
        self.picture.setContentsGravity_(Quartz.kCAGravityResize)
        self.picture.setMagnificationFilter_(Quartz.kCAFilterNearest)  # crisp noise blocks
        self.picture.setMasksToBounds_(True)
        self.picture.setBackgroundColor_(Quartz.CGColorCreateGenericGray(0, 1))
        self.layer.addSublayer_(self.picture)
        parent.addSublayer_(self.layer)
        self.reset()

    def reset(self):
        """Back to an idle, hidden thumbnail with the plain border of a frame in flight."""
        self.frame = None
        self.clear = None        # the centre square, S x S x 4
        self.noise = None        # S x S x 4: NOISE_BLOCKS x NOISE_BLOCKS blocks of random colour
        self.order = None        # the order in which blocks turn into noise
        self.encrypted = -1      # how many blocks are noise now (-1: not drawn yet)
        self.phase = None
        self.start = self.end = 0.0
        self.verdict = None      # (Verdict, round trip ms), once classified
        self.layer.removeAllAnimations()
        self.picture.removeAllAnimations()
        self.layer.setHidden_(True)
        self.layer.setOpacity_(1.0)
        self.layer.setBorderWidth_(IDLE_BORDER)
        self.layer.setBorderColor_(Quartz.CGColorCreateGenericGray(1, 0.6))

    @property
    def frame_id(self):
        return None if self.frame is None else self.frame.id

    def load(self, frame):
        """Decode the frame's centre square; new noise only for a new frame."""
        clear = centre_square(frame_rgbx(frame))
        if self.frame is None or self.frame.id != frame.id or self.noise is None:
            k = clear.shape[0] // NOISE_BLOCKS
            blocks = RNG.integers(0, 256, (NOISE_BLOCKS, NOISE_BLOCKS, 4), dtype=np.uint8)
            self.noise = np.ascontiguousarray(np.repeat(np.repeat(blocks, k, 0), k, 1))
            self.order = RNG.permutation(NOISE_BLOCKS * NOISE_BLOCKS)
            self.encrypted = -1
        self.frame, self.clear = frame, clear
        self.draw(max(self.encrypted, 0), force=True)
        self.layer.setHidden_(False)

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
        self.picture.setContents_(to_cgimage(np.ascontiguousarray(image)))

    def encrypt_to(self, fraction):
        self.draw(round(fraction * NOISE_BLOCKS * NOISE_BLOCKS))

    def place(self, rect):
        """The picture at rect, the border round it."""
        b = self.layer.borderWidth()
        self.layer.setFrame_(outset(rect, b))
        self.picture.setFrame_(inside(rect, b))


class Viewer(NSObject):
    """The window: the live camera with its menus and status line, and each frame snapped
    from the live picture, encrypted, sent to the middle while the deploy works on it,
    then moved to the right and decrypted.

    PyObjC turns the methods of an NSObject subclass into Objective-C selectors; the
    animation helpers are marked @objc.python_method so they stay plain Python methods.
    """

    def initWithHub_link_(self, hub, link):
        self = objc.super(Viewer, self).init()
        self.hub = hub
        self.link = link
        self.devices = list_devices()
        self.formats = []
        self.session = None
        self.grabber = None
        self.shown_frame = 0
        self.queue = libdispatch.dispatch_queue_create(b"webcam.frames", None)
        self.pool = []         # idle slots
        self.leaving = []      # (slot, time it is out of sight)
        self.flow = None       # the frame on its way: snap, encrypt, to the middle, waiting
        self.right = None      # the last result: to the right, decrypt, shown
        self.last_snapped = -1
        self.build_window()
        self.fill_devices()
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1 / 60, self, "tick:", None, True)
        return self

    def build_window(self):
        width, height = 960, 540 + BAR_HEIGHT + STATUS_HEIGHT
        style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
                 | AppKit.NSWindowStyleMaskMiniaturizable | AppKit.NSWindowStyleMaskResizable)
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height), style, AppKit.NSBackingStoreBuffered, False)
        self.window.setTitle_("BlindSight")
        self.window.setMinSize_((480, 320))
        content = self.window.contentView()

        bar_y = height - BAR_HEIGHT + 8
        self.device_popup = self.add_popup(content, NSMakeRect(8, bar_y, 300, 26), "deviceChanged:")
        self.format_popup = self.add_popup(content, NSMakeRect(316, bar_y, 300, 26), "formatChanged:")

        self.video = AppKit.NSView.alloc().initWithFrame_(
            NSMakeRect(0, STATUS_HEIGHT, width, height - BAR_HEIGHT - STATUS_HEIGHT))
        self.video.setWantsLayer_(True)
        root = self.video.layer()
        root.setBackgroundColor_(Quartz.CGColorCreateGenericGray(0, 1))
        root.setContentsGravity_(Quartz.kCAGravityResizeAspect)
        root.setMasksToBounds_(True)  # results leave at the view's edge
        self.video.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        content.addSubview_(self.video)

        self.status = AppKit.NSTextField.labelWithString_("")
        self.status.setFrame_(NSMakeRect(8, 4, width - 16, 16))
        self.status.setAutoresizingMask_(AppKit.NSViewWidthSizable)
        self.status.setFont_(AppKit.NSFont.monospacedSystemFontOfSize_weight_(11, AppKit.NSFontWeightRegular))
        content.addSubview_(self.status)
        self.result_status = AppKit.NSTextField.labelWithString_("")
        self.result_status.setFrame_(self.status.frame())
        self.result_status.setAutoresizingMask_(AppKit.NSViewWidthSizable)
        self.result_status.setFont_(self.status.font())
        self.result_status.setAlignment_(AppKit.NSTextAlignmentRight)
        content.addSubview_(self.result_status)

        busy = Quartz.CAShapeLayer.layer()
        busy.setFillColor_(None)
        busy.setStrokeColor_(Quartz.CGColorCreateGenericRGB(*BUSY_RGBA))
        busy.setLineWidth_(BUSY_LINE)
        busy.setLineDashPattern_(list(BUSY_DASH))
        busy.setLineJoin_(Quartz.kCALineJoinRound)
        busy.setZPosition_(1e5)  # above the thumbnails
        busy.setHidden_(True)
        march = Quartz.CABasicAnimation.animationWithKeyPath_("lineDashPhase")
        march.setFromValue_(0)
        march.setToValue_(sum(BUSY_DASH))  # positive: the dashes run counter-clockwise
        march.setDuration_(BUSY_PERIOD_S)
        march.setRepeatCount_(float("inf"))
        march.setTimingFunction_(
            Quartz.CAMediaTimingFunction.functionWithName_(Quartz.kCAMediaTimingFunctionLinear))
        busy.addAnimation_forKey_(march, "march")
        root.addSublayer_(busy)
        self.busy = busy
        self.busy_rect = None

        plate = Quartz.CALayer.layer()
        plate.setBackgroundColor_(Quartz.CGColorCreateSRGB(*(c / 255 for c in LOGO_PLATE), 1.0))  # sRGB, as on the site
        plate.setZPosition_(1e5)  # above the thumbnails
        plate.setHidden_(True)    # until there is a picture to put it on
        logo = Quartz.CALayer.layer()
        logo.setContents_(AppKit.NSImage.alloc().initWithContentsOfFile_(str(LOGO)))
        logo.setContentsGravity_(Quartz.kCAGravityResizeAspect)
        plate.addSublayer_(logo)
        root.addSublayer_(plate)
        self.plate, self.logo = plate, logo

        self.window.center()
        self.window.makeKeyAndOrderFront_(None)

    def add_popup(self, parent, frame, action):
        popup = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(frame, False)
        popup.setTarget_(self)
        popup.setAction_(action)
        popup.setAutoresizingMask_(AppKit.NSViewMinYMargin)
        parent.addSubview_(popup)
        return popup

    def fill_devices(self):
        self.device_popup.removeAllItems()
        if not self.devices:
            self.device_popup.addItemWithTitle_("No cameras found")
            self.device_popup.setEnabled_(False)
            self.format_popup.setEnabled_(False)
            return
        names = [d.localizedName() for d in self.devices]
        for d, name in zip(self.devices, names):
            # Popup titles must be unique, so tell identical cameras apart by their ID.
            self.device_popup.addItemWithTitle_(f"{name} ({d.uniqueID()})" if names.count(name) > 1 else name)
        self.deviceChanged_(None)

    def deviceChanged_(self, sender):
        device = self.devices[self.device_popup.indexOfSelectedItem()]
        formats = device_formats(device)
        order = list(PIXEL_FORMATS)
        self.formats = sorted(formats.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1], kv[0][2]))
        self.format_popup.removeAllItems()
        if not self.formats:
            self.stop()
            self.format_popup.addItemWithTitle_("No supported format")
            self.format_popup.setEnabled_(False)
            self.status.setStringValue_(f"{device.localizedName()} offers none of {', '.join(PIXEL_FORMATS)}")
            return
        self.format_popup.setEnabled_(True)
        active = device.activeFormat()
        selected = 0
        for i, ((code, w, h), (fmt, fps)) in enumerate(self.formats):
            self.format_popup.addItemWithTitle_(f"{code}  {w}x{h}  {aspect(w, h)}  {fps:g} fps")
            if fmt == active:
                selected = i
        self.format_popup.selectItemAtIndex_(selected)
        self.formatChanged_(None)

    def formatChanged_(self, sender):
        self.stop()
        device = self.devices[self.device_popup.indexOfSelectedItem()]
        (code, w, h), (fmt, _) = self.formats[self.format_popup.indexOfSelectedItem()]
        self.grabber = RGBGrabber.alloc().initWithHub_(self.hub)
        try:
            self.session = start_capture(device, code, fmt, w, h, self.grabber, self.queue)
        except RuntimeError as e:
            self.status.setStringValue_(str(e))

    def stop(self):
        if self.session is not None:
            self.session.stopRunning()
            self.session = None
        self.grabber = None
        self.shown_frame = 0
        self.video.layer().setContents_(None)
        for slot in [self.flow, self.right] + [s for s, _ in self.leaving]:
            if slot is not None:
                slot.reset()
                self.pool.append(slot)
        self.flow = self.right = None
        self.leaving = []
        self.busy.setHidden_(True)

    def tick_(self, timer):
        if self.grabber is None:
            return
        frame_id, rgbx = self.grabber.latest()
        if rgbx is None:
            return
        if frame_id != self.shown_frame:
            self.shown_frame = frame_id
            Quartz.CATransaction.begin()
            Quartz.CATransaction.setDisableActions_(True)
            self.video.layer().setContents_(to_cgimage(np.ascontiguousarray(rgbx)))
            Quartz.CATransaction.commit()
        self.status.setStringValue_(self.link.summary())
        self.result_status.setStringValue_(self.link.last_result())
        self.update_pips()

    # ------------------------------------------------------------ geometry

    @objc.python_method
    def take_slot(self):
        if self.pool:
            return self.pool.pop()
        return Slot(self.video.layer())

    @objc.python_method
    def picture_rect(self, image_w, image_h):
        """Where the camera picture is shown: aspect-fit in the video view."""
        bounds = self.video.bounds()
        view_w, view_h = bounds.size.width, bounds.size.height
        fit = min(view_w / image_w, view_h / image_h)
        shown_w, shown_h = image_w * fit, image_h * fit
        return (((view_w - shown_w) / 2, (view_h - shown_h) / 2), (shown_w, shown_h))

    @objc.python_method
    def crop_rect(self, image_w, image_h):
        """Where the frame's centre square is shown in the picture."""
        (x, y), (w, h) = self.picture_rect(image_w, image_h)
        side = min(w, h)
        return ((x + (w - side) / 2, y + (h - side) / 2), (side, side))

    @objc.python_method
    def geometry(self, image_w, image_h):
        """The left, middle, right and off-right squares along the bottom of the picture."""
        (left, bottom), (shown_w, shown_h) = self.picture_rect(image_w, image_h)
        view_w = self.video.bounds().size.width
        side = min(shown_h * PIP_FRACTION, shown_w * PIP_MAX_WIDTH)
        y = bottom + PIP_MARGIN
        size = (side, side)
        return {
            "left": ((left + PIP_MARGIN, y), size),
            "middle": ((left + (shown_w - side) / 2, y), size),
            "right": ((left + shown_w - PIP_MARGIN - side, y), size),
            "off_right": ((view_w, y), size),
        }

    # ------------------------------------------------------------ every tick

    @objc.python_method
    def update_pips(self):
        if self.grabber is None:
            return
        _, rgbx = self.grabber.latest()
        if rgbx is None:
            return
        image_w, image_h = rgbx.shape[1], rgbx.shape[0]
        now = time.monotonic()
        where = self.geometry(image_w, image_h)

        immediate()
        for slot, gone in list(self.leaving):  # recycle thumbnails that slid or faded out
            if now >= gone:
                self.leaving.remove((slot, gone))
                slot.reset()
                self.pool.append(slot)
        Quartz.CATransaction.commit()

        self.advance_flow(where, now)
        self.advance_right(where, now)
        if self.flow is None:
            sent = self.link.sent
            if sent is not None and int(sent.id) > self.last_snapped:
                self.snap(sent, image_w, image_h, now)
        self.follow_resize(where)
        self.update_busy(where)
        self.update_logo(image_w, image_h)

    @objc.python_method
    def advance_flow(self, where, now):
        slot = self.flow
        if slot is None:
            return
        if slot.phase == "snap" and now >= slot.end:
            self.set_phase(slot, "encrypt", now, ENCRYPT_S)
        if slot.phase == "encrypt":
            immediate()
            slot.encrypt_to(min((now - slot.start) / ENCRYPT_S, 1.0))
            Quartz.CATransaction.commit()
            if now >= slot.end:
                self.set_phase(slot, "to_middle", now, TO_MIDDLE_S)
                animate(TO_MIDDLE_S)
                slot.place(where["middle"])
                Quartz.CATransaction.commit()
        if slot.phase == "to_middle" and now >= slot.end:
            self.set_phase(slot, "waiting", now, 0.0)
        if slot.phase == "waiting":
            verdict = self.link.verdicts.get(slot.frame_id)
            sent = self.link.sent
            if verdict is not None:
                self.to_right(slot, verdict, where, now)
            elif sent is not None and int(sent.id) > int(slot.frame_id):  # refused: the link moved on
                self.fade_out(slot, now)
                self.flow = None

    @objc.python_method
    def advance_right(self, where, now):
        slot = self.right
        if slot is None:
            return
        if slot.phase == "to_right" and now >= slot.end:
            self.set_phase(slot, "decrypt", now, DECRYPT_S)
            animate(DECRYPT_S)  # the box fades from grey to its colour
            slot.layer.setBorderColor_(self.final_colour(slot))
            Quartz.CATransaction.commit()
        if slot.phase == "decrypt":
            immediate()
            slot.encrypt_to(1.0 - min((now - slot.start) / DECRYPT_S, 1.0))
            Quartz.CATransaction.commit()
            if now >= slot.end:
                self.finish(slot)

    @objc.python_method
    def set_phase(self, slot, phase, now, duration):
        slot.phase, slot.start, slot.end = phase, now, now + duration

    @objc.python_method
    def final_colour(self, slot):
        verdict, _ = slot.verdict
        p = p_near(list(verdict.scores))
        return cg_rgb(box_colour(p) if p == p else GREY)  # p != p: NaN

    @objc.python_method
    def finish(self, slot):
        """Show a result decrypted, with its colour, at once if need be."""
        immediate()
        slot.layer.removeAnimationForKey_("borderColor")
        slot.layer.setBorderColor_(self.final_colour(slot))
        slot.encrypt_to(0.0)
        Quartz.CATransaction.commit()
        slot.phase = "shown"

    @objc.python_method
    def follow_resize(self, where):
        """Keep resting thumbnails in place when the window changes size."""
        immediate()
        for slot, rest in ((self.flow, ("encrypt", "waiting")), (self.right, ("decrypt", "shown"))):
            if slot is not None and slot.phase in rest:
                slot.place(where["left" if slot.phase == "encrypt" else
                                 "middle" if slot.phase == "waiting" else "right"])
        Quartz.CATransaction.commit()

    @objc.python_method
    def update_logo(self, image_w, image_h):
        """The logo in the picture's top left corner, sized by the picture's height: the same share of it for a
        square format and for 16:9."""
        (left, bottom), (_, shown_h) = self.picture_rect(image_w, image_h)
        w = max(shown_h * LOGO_FRACTION, LOGO_MIN)
        h = w / LOGO_ASPECT
        pad = h * LOGO_PAD
        plate_h = h + 2 * pad
        immediate()
        self.plate.setFrame_(((left + PIP_MARGIN, bottom + shown_h - PIP_MARGIN - plate_h), (w + 2 * pad, plate_h)))
        self.logo.setFrame_(((pad, pad), (w, h)))
        self.plate.setHidden_(False)
        Quartz.CATransaction.commit()

    @objc.python_method
    def update_busy(self, where):
        """The turning box around the frame at the deploy."""
        waiting = self.flow is not None and self.flow.phase == "waiting"
        immediate()
        if waiting:
            (x, y), (w, h) = where["middle"]
            g = BUSY_GAP + BUSY_LINE / 2
            rect = ((x - g, y - g), (w + 2 * g, h + 2 * g))
            if rect != self.busy_rect:  # first show, or the window was resized
                self.busy_rect = rect
                self.busy.setFrame_(rect)
                inner = ((BUSY_LINE / 2, BUSY_LINE / 2), (rect[1][0] - BUSY_LINE, rect[1][1] - BUSY_LINE))
                self.busy.setPath_(Quartz.CGPathCreateWithRoundedRect(inner, 3, 3, None))
        self.busy.setHidden_(not waiting)
        Quartz.CATransaction.commit()

    # ------------------------------------------------------------ transitions

    @objc.python_method
    def snap(self, frame, image_w, image_h, now):
        """Snap the frame's centre square from the live picture into the lower left."""
        slot = self.take_slot()
        self.last_snapped = int(frame.id)
        crop = self.crop_rect(image_w, image_h)
        left = self.geometry(image_w, image_h)["left"]
        # The end state without implicit animations, then explicit ones that start
        # from the picture (see basic()).
        immediate()
        slot.load(frame)
        slot.layer.setZPosition_(Z_FLOW)
        slot.place(left)
        border = slot.layer.borderWidth()
        fly(slot.layer, outset(crop, border), outset(left, border), SNAP_S)
        fly(slot.picture, inside(crop, border), inside(left, border), SNAP_S)
        Quartz.CATransaction.commit()
        slot.phase, slot.start, slot.end = "snap", now, now + SNAP_S
        self.flow = slot

    @objc.python_method
    def to_right(self, slot, verdict, where, now):
        """The verdict is in: the encrypted result takes a grey box and slides to the
        right; the previous result, decrypted at once if it was not yet, slides out."""
        previous = self.right
        if previous is not None:
            if previous.phase != "shown":
                self.finish(previous)
            immediate()
            previous.layer.setZPosition_(Z_LEAVING)
            Quartz.CATransaction.commit()
            animate(TO_RIGHT_S)
            previous.place(where["off_right"])
            previous.layer.setOpacity_(0.0)
            Quartz.CATransaction.commit()
            self.leaving.append((previous, now + TO_RIGHT_S))
        slot.verdict = verdict
        immediate()
        slot.layer.setBorderWidth_(BOX_PX)
        slot.layer.setBorderColor_(cg_rgb(GREY))
        slot.layer.setZPosition_(Z_RESULT)
        Quartz.CATransaction.commit()
        animate(TO_RIGHT_S)
        slot.place(where["right"])
        Quartz.CATransaction.commit()
        self.set_phase(slot, "to_right", now, TO_RIGHT_S)
        self.right, self.flow = slot, None

    @objc.python_method
    def fade_out(self, slot, now):
        animate(FADE_S)
        slot.layer.setOpacity_(0.0)
        Quartz.CATransaction.commit()
        self.leaving.append((slot, now + FADE_S))


class AppDelegate(NSObject):
    """Closing the window and Cmd-Q stop the event loop, like Ctrl-C in the terminal, instead of
    ending the process on the spot, so run() returns and the client shuts down cleanly."""

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True

    def applicationShouldTerminate_(self, app):
        # Ctrl-C reaches here too: PyObjC's interrupt handler asks the app to terminate.
        stopper = AppHelper.PyObjCAppHelperRunLoopStopper.currentRunLoopStopper()
        if stopper is not None:
            stopper.shouldStop = True  # runEventLoop does not start the app again
        app.stop_(None)
        # stop: takes effect after the next event; post one so it happens now.
        wake = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            AppKit.NSEventTypeApplicationDefined, (0, 0), 0, 0, 0, None, 0, 0, 0)
        app.postEvent_atStart_(wake, True)
        return AppKit.NSTerminateCancel


def build_menu(app):
    menu = AppKit.NSMenu.alloc().init()
    app_item = AppKit.NSMenuItem.alloc().init()
    menu.addItem_(app_item)
    app_menu = AppKit.NSMenu.alloc().init()
    app_menu.addItemWithTitle_action_keyEquivalent_("Quit", "terminate:", "q")
    app_item.setSubmenu_(app_menu)
    app.setMainMenu_(menu)


def run(client, classify):
    """Open the window and classify the webcam until the window closes. client.py passes the
    prepared Belfort ML client and how to classify one prepared image with it."""
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    build_menu(app)
    hub = FrameHub()
    link = Link(client, classify, hub)
    viewer = Viewer.alloc().initWithHub_link_(hub, link)
    app.activateIgnoringOtherApps_(True)
    print("Window open. To stop: close the window, press Cmd-Q in it, or press Ctrl-C here.", flush=True)
    # Ctrl-C asks the app to terminate, which the delegate turns into a clean stop. runEventLoop's
    # installInterrupt only installs this handler when it creates the app itself; here it exists.
    AppHelper.installMachInterrupt()
    AppHelper.runEventLoop()
    viewer.stop()
    link.stop()
    print(f"Stopped after {link.classified} classified image(s).", flush=True)
