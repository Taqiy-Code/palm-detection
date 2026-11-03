"""Konfigurasi aplikasi (dataclass murni, tanpa dependensi OpenCV/Qt)."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import List

CONFIG_FILE = "config.json"


@dataclass
class AppConfig:
    # Parameter Deteksi ROI & Tangan
    jumlah_roi: int = 3
    debounce_masuk: int = 3
    debounce_keluar: int = 8
    ambang_piksel: int = 30
    ambang_rasio: float = 0.08
    hand_overlap_thr: float = 0.15      # Ambang rasio luas irisan convex hull tangan dan ROI terhadap luas ROI
    motion_fallback_thr: float = 0.02   # Ambang rasio gerak antar-frame di ROI sebagai fallback deteksi tangan
    ref_clean_frames: int = 5           # Jumlah frame stabil tanpa tangan sebelum mengunci referensi
    debug_diff: bool = False

    # Optimasi Sarung Tangan (CLAHE & Pseudo-Skin)
    use_clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_invert_dark: bool = False
    show_clahe_preview: bool = False

    # Optimasi ROI Cropping (Pangkas Area Deteksi Tangan)
    use_roi_crop: bool = True
    roi_crop_mode: str = "union"       # "union" (Gabungan Semua ROI) | "individual" (Per-ROI)
    roi_crop_padding: int = 40         # Margin padding dalam piksel
    show_crop_box: bool = True         # Tampilkan garis pembatas crop di video

    # Engine Deteksi Tangan (Dual-Backend: RTMlib vs MediaPipe)
    hand_backend: str = "rtmlib"             # "rtmlib" (RTMPose Hand) | "mediapipe" (MediaPipe Hands)
    rtm_score_thr: float = 0.35              # Ambang batas confidence score per sendi RTMlib
    hand_tracking_mode: str = "full_hand"     # "full_hand" (Seluruh Tangan - 21 Sendi & Kontur) | "palm_center" | "fingertips"

    # Parameter Hardware Kamera
    camera_index: int = 1
    frame_width: int = 1280
    frame_height: int = 720
    auto_focus: bool = False
    auto_exposure: bool = False
    auto_wb: bool = False
    focus: int = 20
    exposure: int = -6
    wb_temp: int = 4500
    gain: int = 0

    # Koordinat poligon ROI tersimpan (opsional)
    saved_rois: List[List[List[int]]] = field(default_factory=list)
    # Penamaan kustom per ROI / komponen
    roi_names: List[str] = field(default_factory=list)

    def get_roi_name(self, idx: int) -> str:
        """Mengembalikan nama kustom ROI atau nama default 'ROI #X' jika belum dinamai."""
        if 0 <= idx < len(self.roi_names) and self.roi_names[idx].strip():
            return self.roi_names[idx].strip()
        return f"ROI #{idx + 1}"

    def set_roi_name(self, idx: int, name: str) -> None:
        """Menyimpan nama kustom komponen untuk ROI ke-idx."""
        while len(self.roi_names) <= idx:
            self.roi_names.append("")
        self.roi_names[idx] = name.strip()

    @classmethod
    def has_field(cls, name: str) -> bool:
        return name in {f.name for f in fields(cls)}

    def update_from(self, other: "AppConfig") -> None:
        """Menyalin seluruh nilai dari `other` ke instance ini (in-place, menjaga satu sumber kebenaran)."""
        for f in fields(self):
            setattr(self, f.name, copy.deepcopy(getattr(other, f.name)))

    def save(self, filepath: str = CONFIG_FILE) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, filepath: str = CONFIG_FILE) -> "AppConfig":
        if not os.path.exists(filepath):
            return cls()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid_keys = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)
        except Exception as e:
            print(f"[Peringatan] Gagal memuat {filepath}: {e}")
            return cls()
