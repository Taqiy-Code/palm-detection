"""Entry point utama aplikasi Palm Detection.

Seluruh arsitektur telah dipecah secara modular menjadi:
    - HandDetector (palm_detection.detector): RTMHandDetector, MediaPipeHandDetector, preprocess_glove
    - StepSequencer (palm_detection.sequencer): State machine murni tanpa dependensi OpenCV/Qt
    - OverlayRenderer (palm_detection.renderer): Visualizer grafis OpenCV
    - CameraWorker (palm_detection.worker): Orkestrator thread kamera dan deteksi
    - GUI (palm_detection.gui): Antarmuka modern PyQt5
"""

from __future__ import annotations

import os
import sys

# PENTING (Windows DLL Load Order):
# MediaPipe (_framework_bindings) harus di-import SEBELUM PyQt5 agar inisialisasi
# DLL C++ / MSVC runtime MediaPipe tidak gagal ("DLL initialization routine failed").
try:
    import mediapipe as _mp  # noqa: F401
except Exception:
    pass

# Mencegah konflik library Qt antara OpenCV dan PyQt5 pada sistem Linux/Jetson.
# Menghapus path plugin cv2 dan memprioritaskan direktori platform plugins PyQt5.
if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ and "cv2" in os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]:
    del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]

try:
    import PyQt5
    pyqt_base = os.path.dirname(PyQt5.__file__)
    for cand in (
        os.path.join(pyqt_base, "Qt5", "plugins", "platforms"),
        os.path.join(pyqt_base, "Qt", "plugins", "platforms"),
        os.path.join(pyqt_base, "plugins", "platforms"),
    ):
        if os.path.isdir(cand):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = cand
            break
except Exception:
    pass

from palm_detection.config import AppConfig, CONFIG_FILE
from palm_detection.gui import (
    MainWindow,
    ROICardWidget,
    SettingsDialog,
    VideoCanvas,
    main,
)
from palm_detection.detector import (
    DEFAULT_YOLO_MODEL,
    MODEL_DIR,
    HandDetectionPipeline,
    HandDetector,
    MediaPipeHandDetector,
    RTMHandDetector,
    SENDI_PUNGGUNG,
    UJUNG_JARI,
    YOLOHandDetector,
    YOLOPoseHandDetector,
    ensure_model_dir,
    list_available_yolo_models,
    preprocess_glove,
    resolve_yolo_model_path,
)
from palm_detection.renderer import (
    SAMBUNGAN_TANGAN,
    WARNA_KUNCI_BGR,
    WARNA_ROI_BGR,
    OverlayRenderer,
)
from palm_detection.sequencer import (
    SequencerEvent,
    StepSequencer,
    StepTelemetryItem,
)
from palm_detection.states import (
    BannerState,
    CropMode,
    HandBackend,
    LogLevel,
    RoiState,
    StepStatus,
    TrackingMode,
)
from palm_detection.vision import (
    clean_diff_mask,
    hull_roi_overlap_ratio,
    normalize_and_validate_roi,
    normalize_brightness,
    rasio_gerak_roi,
    rasio_perubahan,
    siapkan_gray,
)
from palm_detection.worker import CameraWorker

# Kompatibilitas mundur jika fungsi/variabel lama diakses
preprocess_for_mediapipe = preprocess_glove

__all__ = [
    "AppConfig",
    "CONFIG_FILE",
    "StepStatus",
    "RoiState",
    "BannerState",
    "LogLevel",
    "TrackingMode",
    "HandBackend",
    "CropMode",
    "HandDetector",
    "RTMHandDetector",
    "MediaPipeHandDetector",
    "YOLOPoseHandDetector",
    "YOLOHandDetector",
    "HandDetectionPipeline",
    "MODEL_DIR",
    "DEFAULT_YOLO_MODEL",
    "ensure_model_dir",
    "list_available_yolo_models",
    "resolve_yolo_model_path",
    "preprocess_glove",
    "preprocess_for_mediapipe",
    "StepSequencer",
    "SequencerEvent",
    "StepTelemetryItem",
    "OverlayRenderer",
    "CameraWorker",
    "VideoCanvas",
    "ROICardWidget",
    "SettingsDialog",
    "MainWindow",
    "main",
    "WARNA_ROI_BGR",
    "WARNA_KUNCI_BGR",
    "UJUNG_JARI",
    "SENDI_PUNGGUNG",
    "SAMBUNGAN_TANGAN",
    "siapkan_gray",
    "normalize_and_validate_roi",
    "hull_roi_overlap_ratio",
    "normalize_brightness",
    "clean_diff_mask",
    "rasio_perubahan",
    "rasio_gerak_roi",
]

if __name__ == "__main__":
    main()
