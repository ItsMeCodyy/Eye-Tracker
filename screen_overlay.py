import ctypes

import objc
from AppKit import (
    NSScreenSaverWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
)
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

BLOB_RADIUS = 90
DOT_RADIUS = 18


class _CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


def display_width_mm():
    """Physical width of the main display in millimetres."""
    graphics = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    graphics.CGMainDisplayID.restype = ctypes.c_uint32
    graphics.CGDisplayScreenSize.restype = _CGSize
    graphics.CGDisplayScreenSize.argtypes = [ctypes.c_uint32]
    return graphics.CGDisplayScreenSize(graphics.CGMainDisplayID()).width


def pin_above_everything(widget):
    """Keep the window over every app and Space, including full-screen video, without catching clicks."""
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
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setGeometry(screen.geometry())
        self.blob = None
        self.target = None
        self.capturing = False
        self.message = ""

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

    def set_blob(self, x, y):
        previous = self._blob_rect()
        self.blob = (x, y)
        self.update(previous.united(self._blob_rect()))

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
            int(x - BLOB_RADIUS), int(y - BLOB_RADIUS), 2 * BLOB_RADIUS, 2 * BLOB_RADIUS
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

        if self.blob is not None:
            center = QPointF(*self.blob)
            gradient = QRadialGradient(center, BLOB_RADIUS)
            gradient.setColorAt(0.0, QColor(255, 60, 60, 190))
            gradient.setColorAt(0.55, QColor(255, 60, 60, 110))
            gradient.setColorAt(1.0, QColor(255, 60, 60, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(center, BLOB_RADIUS, BLOB_RADIUS)

        if self.message:
            painter.setFont(QFont("Helvetica Neue", 20))
            metrics = painter.fontMetrics()
            lines = self.message.split("\n")
            text_width = max(metrics.horizontalAdvance(line) for line in lines)
            pill = QRect(
                (self.width() - text_width) // 2 - 24,
                40,
                text_width + 48,
                metrics.lineSpacing() * len(lines) + 24,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(pill, 18, 18)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, self.message)
