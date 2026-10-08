import ctypes
import sys

# Windows never imports the macOS bridge; the existing native macOS behavior stays intact.
if sys.platform == "darwin":
    import objc
    from AppKit import (
        NSScreenSaverWindowLevel,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorIgnoresCycle,
        NSWindowCollectionBehaviorStationary,
    )
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

BLOB_RADIUS = 90
DOT_RADIUS = 18


class _CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


def display_width_mm(screen=None):
    """Reported physical width, or zero when the display does not expose it."""
    if screen is not None:
        width = screen.physicalSize().width()
        return float(width) if width > 0 else 0.0
    if sys.platform == "darwin":
        graphics = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        graphics.CGMainDisplayID.restype = ctypes.c_uint32
        graphics.CGDisplayScreenSize.restype = _CGSize
        graphics.CGDisplayScreenSize.argtypes = [ctypes.c_uint32]
        return graphics.CGDisplayScreenSize(graphics.CGMainDisplayID()).width
    return 0.0


def _pin_windows(widget):
    """Pointer-safe Win32 topmost, click-through window without focus stealing."""
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_style = user32.GetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user32.GetWindowLongW
    set_style = user32.SetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user32.SetWindowLongW
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t
    set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    set_style.restype = ctypes.c_ssize_t
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    user32.SetWindowPos.restype = wintypes.BOOL
    handle = int(widget.winId())
    # WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
    set_style(handle, -20, get_style(handle, -20) | 0x20 | 0x80 | 0x08000000)
    # HWND_TOPMOST; SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED
    user32.SetWindowPos(handle, wintypes.HWND(-1), 0, 0, 0, 0, 0x2 | 0x1 | 0x10 | 0x20)


def pin_above_everything(widget):
    """Keep the window over every app and Space, including full-screen video, without catching clicks."""
    if QGuiApplication.platformName() in ("offscreen", "minimal"):
        return
    if sys.platform == "win32":
        _pin_windows(widget)
        return
    if sys.platform != "darwin":
        return
    window = objc.objc_object(c_void_p=int(widget.winId())).window()
    window.setLevel_(NSScreenSaverWindowLevel)
    window.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorStationary
        | NSWindowCollectionBehaviorFullScreenAuxiliary
        | NSWindowCollectionBehaviorIgnoresCycle
    )
    window.setHidesOnDeactivate_(False)
    window.setIgnoresMouseEvents_(True)


class ScreenOverlay(QWidget):
    def __init__(self, screen):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        if sys.platform == "darwin":
            self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setGeometry(screen.geometry())
        self.blob = None
        self.target = None
        self.capturing = False
        self.message = ""
        self.radius = BLOB_RADIUS
        self.quality = 1.0
        self.progress = 0.0

    def show_calibration(self, target_fraction, capturing, message):
        self.blob = None
        self.target = (
            None
            if target_fraction is None
            else (target_fraction[0] * self.width(), target_fraction[1] * self.height())
        )
        self.capturing = capturing
        self.message = message
        self._present()

    def show_tracking(self, message=""):
        self.target = None
        self.message = message
        self._present()

    def clear_message(self):
        self.message = ""
        self.update()

    def set_blob(self, x, y, quality=1.0):
        previous = self._blob_rect()
        self.blob = (x, y)
        self.quality = quality
        self.update(previous.united(self._blob_rect()))

    def set_radius(self, radius):
        previous = self._blob_rect()
        self.radius = max(16, min(180, int(radius)))
        self.update(previous.united(self._blob_rect()))

    def set_capture_progress(self, progress):
        self.progress = max(0.0, min(1.0, progress))
        if self.target is not None:
            x, y = self.target
            self.update(QRect(int(x - 30), int(y - 30), 60, 60))

    def hide_blob(self):
        previous = self._blob_rect()
        self.blob = None
        self.update(previous)

    def _present(self):
        if not self.isVisible():
            self.show()
            pin_above_everything(self)
        self.update()

    def _blob_rect(self):
        if self.blob is None:
            return QRect()
        x, y = self.blob
        return QRect(
            int(x - self.radius) - 2, int(y - self.radius) - 2, 2 * self.radius + 4, 2 * self.radius + 4
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.target is not None:
            center = QPointF(*self.target)
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            painter.setBrush(QColor(60, 220, 120) if self.capturing else QColor(255, 190, 40))
            painter.drawEllipse(center, DOT_RADIUS, DOT_RADIUS)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(center, 4, 4)
            if self.capturing:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(90, 240, 190), 3))
                painter.drawArc(QRect(int(center.x() - 25), int(center.y() - 25), 50, 50),
                                90 * 16, int(-360 * 16 * self.progress))

        if self.blob is not None:
            center = QPointF(*self.blob)
            gradient = QRadialGradient(center, self.radius)
            opacity = 0.45 + 0.55 * max(0.0, min(1.0, self.quality))
            gradient.setColorAt(0.0, QColor(255, 60, 60, int(190 * opacity)))
            gradient.setColorAt(0.55, QColor(255, 60, 60, int(110 * opacity)))
            gradient.setColorAt(1.0, QColor(255, 60, 60, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(center, self.radius, self.radius)
            painter.setBrush(QColor(255, 245, 245, 210))
            painter.drawEllipse(center, 3, 3)

        if self.message:
            painter.setFont(QFont("Segoe UI" if sys.platform == "win32" else "Helvetica Neue", 15))
            text_width = max(160, min(self.width() - 80, 920))
            flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
            bounds = painter.boundingRect(QRect(0, 0, text_width, self.height()), flags, self.message)
            y = self.height() - bounds.height() - 72 if self.target and self.target[1] < self.height() / 2 else 40
            pill = QRect((self.width() - text_width) // 2 - 20, y, text_width + 40, bounds.height() + 24)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(pill, 18, 18)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(pill.adjusted(20, 12, -20, -12), flags, self.message)
