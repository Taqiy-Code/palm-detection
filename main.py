"""Entry point utama aplikasi Palm Detection.

Seluruh arsitektur telah dipecah secara modular menjadi:
    - HandDetector (palm_detection.detector): RTMHandDetector, MediaPipeHandDetector, preprocess_glove
    - StepSequencer (palm_detection.sequencer): State machine murni tanpa dependensi OpenCV/Qt
    - OverlayRenderer (palm_detection.renderer): Visualizer grafis OpenCV
    - CameraWorker (palm_detection.worker): Orkestrator thread kamera dan deteksi
    - GUI (palm_detection.gui): Antarmuka modern PyQt5
"""

from __future__ import annotations

import sys

from palm_detection.config import AppConfig, CONFIG_FILE
from palm_detection.detector import (
    HandDetectionPipeline,
    HandDetector,
    MediaPipeHandDetector,
    RTMHandDetector,
    SENDI_PUNGGUNG,
    UJUNG_JARI,
    YOLOPoseHandDetector,
    preprocess_glove,
)
from palm_detection.gui import (
    MainWindow,
    ROICardWidget,
    SettingsDialog,
    VideoCanvas,
    main,
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
    "HandDetectionPipeline",
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
