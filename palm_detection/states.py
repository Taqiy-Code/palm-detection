"""Enum status dan state aplikasi (murni, tanpa dependensi eksternal).

Nilai `StrEnum` identik dengan string lama di config.json sehingga perbandingan
`config.hand_backend == HandBackend.RTMLIB` tetap berlaku dan serialisasi JSON tidak berubah.
"""

from __future__ import annotations

from enum import Enum, auto

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        """Polyfill StrEnum untuk Python < 3.11 (seperti Python 3.10 di Nvidia Jetson)."""

        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, format_spec: str) -> str:
            return str(self.value).__format__(format_spec)


class StepStatus(Enum):
    """Status internal satu langkah (pengganti integer 0/1/2)."""

    WAITING = auto()   # Menunggu tangan masuk
    HAND_IN = auto()   # Tangan sedang berada di ROI
    DONE = auto()      # Langkah selesai


class RoiState(StrEnum):
    """State tampilan satu ROI untuk GUI dan renderer."""

    SETUP = "SETUP"          # Sedang digambar
    READY = "READY"          # Poligon sudah dibuat (mode setup)
    WAIT_REF = "WAIT_REF"    # Menunggu referensi background bersih
    LOCKED = "LOCKED"        # Belum gilirannya
    ACTIVE = "ACTIVE"        # Giliran sekarang
    HAND_IN = "HAND_IN"      # Tangan di dalam ROI aktif
    DONE = "DONE"            # Selesai
    ERROR = "ERROR"          # Urutan salah


class BannerState(StrEnum):
    SETUP = "setup"
    ACTIVE = "active"
    ERROR = "error"
    DONE = "done"


class LogLevel(StrEnum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARN = "WARN"
    PART = "PART"


class TrackingMode(StrEnum):
    FULL_HAND = "full_hand"
    PALM_CENTER = "palm_center"
    FINGERTIPS = "fingertips"


class HandBackend(StrEnum):
    RTMLIB = "rtmlib"
    MEDIAPIPE = "mediapipe"
    YOLO26N = "yolo26n"
    YOLO_DETECT = "yolo_detect"


class CropMode(StrEnum):
    UNION = "union"
    INDIVIDUAL = "individual"
