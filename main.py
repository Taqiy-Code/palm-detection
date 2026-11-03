from __future__ import annotations

import datetime
import json
import os
import queue
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from rtmlib import Hand as RTMHand
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QKeySequence, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

CONFIG_FILE = "config.json"

# Warna BGR untuk OpenCV overlay
WARNA_ROI_BGR = [
    (255, 128, 0),    # Biru-Oranye
    (0, 200, 255),    # Kuning-Emas
    (200, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (128, 255, 0),    # Hijau Muda
    (255, 0, 128),    # Ungu
]
WARNA_KUNCI_BGR = (130, 130, 130)
UJUNG_JARI = [4, 8, 12, 16, 20]      # 5 titik ujung jari (Thumb, Index, Middle, Ring, Pinky)
SENDI_PUNGGUNG = [0, 5, 9, 13, 17]   # Pergelangan + 4 buku pangkal jari (Area Punggung/Telapak Tangan)
SAMBUNGAN_TANGAN = [
    # Ibu Jari
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Telunjuk
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Jari Tengah
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Jari Manis
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Kelingking
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Telapak & Pergelangan
    (0, 17),
]


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
            valid_keys = {k for k in cls.__dataclass_fields__.keys()}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)
        except Exception as e:
            print(f"[Peringatan] Gagal memuat {filepath}: {e}")
            return cls()


class CameraWorker(QThread):
    """Thread pekerja untuk akuisisi webcam, kalibrasi hardware, MediaPipe Hands, dan logika ROI."""

    frame_ready = pyqtSignal(QImage, int, int)
    telemetry_updated = pyqtSignal(dict)
    camera_info_ready = pyqtSignal(str)
    log_event = pyqtSignal(str, str)  # level ("SUCCESS", "PART", "WARN", "INFO"), message

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.running = False

        # State kamera
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_w = config.frame_width
        self.frame_h = config.frame_height

        # State ROI & Deteksi
        self.rois: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.titik_klik: List[Tuple[int, int]] = []
        self.kursor: Tuple[int, int] = (0, 0)
        self.referensi: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._no_hand_frames: int = 0
        self._stable_ref_buffer: List[np.ndarray] = []
        self._morph_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self.status: List[int] = []          # 0 = menunggu, 1 = tangan di dalam, 2 = DONE
        self.masuk: List[int] = []
        self.keluar: List[int] = []
        self.error_count: List[int] = []
        self.rasio_terakhir: List[float] = []
        self._post_hand_frames: List[List[np.ndarray]] = []

        # State Produksi & Auto-Cycle Part
        self.parts_completed: int = 0
        self._cycle_cooldown_until: Optional[float] = None
        self._last_error_log_time: float = 0.0

        # Antrian perintah thread-safe dari GUI thread ke Worker thread
        self._cmd_queue: queue.Queue = queue.Queue()
        self._redo_roi_idx: Optional[int] = None

        # Look-Up Tables (LUT) 256 nilai untuk sintesis warna kulit (Pseudo Skin-Tone)
        self.lut_r = np.clip(np.arange(256) * 1.15, 0, 255).astype(np.uint8)
        self.lut_g = np.clip(np.arange(256) * 0.95, 0, 255).astype(np.uint8)
        self.lut_b = np.clip(np.arange(256) * 0.80, 0, 255).astype(np.uint8)
        self._cached_clahe_limit = -1.0
        self._clahe: Optional[cv2.CLAHE] = None

        # Dual-Backend Trackers (RTMlib vs MediaPipe)
        self.rtm_hand: Optional[RTMHand] = None
        self.mp_hands: Optional[mp.solutions.hands.Hands] = None
        self.mp_hands_regions: Dict[int, mp.solutions.hands.Hands] = {}

        self._init_roi_arrays()

    def _init_roi_arrays(self) -> None:
        n = max(self.config.jumlah_roi, len(self.rois), 1)
        self.status = [0] * n
        self.masuk = [0] * n
        self.keluar = [0] * n
        self.error_count = [0] * n
        self.rasio_terakhir = [0.0] * n
        self._post_hand_frames = [[] for _ in range(n)]

    def _ensure_arrays_size(self) -> None:
        needed = max(self.config.jumlah_roi, len(self.rois), 1)
        curr = len(self.status)
        if curr < needed:
            diff = needed - curr
            self.status.extend([0] * diff)
            self.masuk.extend([0] * diff)
            self.keluar.extend([0] * diff)
            self.error_count.extend([0] * diff)
            self.rasio_terakhir.extend([0.0] * diff)
            self._post_hand_frames.extend([[] for _ in range(diff)])

    def _close_mp_hands_regions(self) -> None:
        for tracker in self.mp_hands_regions.values():
            try:
                tracker.close()
            except Exception as e:
                print(f"[Peringatan] Gagal menutup MediaPipe region: {e}")
        self.mp_hands_regions.clear()

    def _cleanup_resources(self) -> None:
        """Memastikan semua resource kamera & AI backend dilepas dari memori."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[Peringatan] Gagal merilis VideoCapture: {e}")
            finally:
                self.cap = None

        if self.mp_hands is not None:
            try:
                self.mp_hands.close()
            except Exception as e:
                print(f"[Peringatan] Gagal menutup MediaPipe: {e}")
            finally:
                self.mp_hands = None

        self._close_mp_hands_regions()
        self.rtm_hand = None

    def _preload_selected_backend(self) -> None:
        """Memuat model deteksi tangan di awal agar tidak terjadi frame stutter saat mulai."""
        try:
            if self.config.hand_backend == "rtmlib":
                self.get_rtm_hand()
            else:
                self.get_mp_hands()
        except Exception as e:
            print(f"[Peringatan] Gagal memuat backend {self.config.hand_backend}: {e}")

    def _open_camera(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        def _try_open(idx: int):
            # Coba backend CAP_DSHOW terlebih dahulu untuk kontrol manual
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not c.isOpened():
                c.release()
                c = cv2.VideoCapture(idx)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                success, f = c.read()
                if success and f is not None:
                    return c, f
                c.release()
            return None, None

        # 1. Coba buka kamera yang dikonfigurasi
        cap_obj, frame = _try_open(self.config.camera_index)

        # 2. Jika gagal atau tidak menghasilkan gambar, coba cari kamera lain yang aktif
        if cap_obj is None:
            for alt_idx in [1, 0, 2]:
                if alt_idx == self.config.camera_index:
                    continue
                cap_obj, frame = _try_open(alt_idx)
                if cap_obj is not None:
                    print(f"[Info] Kamera #{self.config.camera_index} tidak aktif, beralih ke Kamera #{alt_idx}")
                    self.config.camera_index = alt_idx
                    break

        if cap_obj is None or not cap_obj.isOpened():
            self.cap = None
            self.camera_info_ready.emit(f"Gagal membuka Kamera Index {self.config.camera_index}")
            return False

        self.cap = cap_obj
        if frame is not None:
            self.frame_h, self.frame_w = frame.shape[:2]

        self._apply_all_camera_settings()

        af = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
        ae = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        awb = self.cap.get(cv2.CAP_PROP_AUTO_WB)
        self.camera_info_ready.emit(
            f"Kamera #{self.config.camera_index} ({self.frame_w}x{self.frame_h}) | "
            f"AF:{af:.0f} AE:{ae:.2f} AWB:{awb:.0f}"
        )
        return True

    def _apply_all_camera_settings(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            return

        # 1. Atur mode Auto / Manual
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.config.auto_focus else 0)
        if self.config.auto_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if self.config.auto_wb else 0)

        # 2. Kunci nilai manual
        if not self.config.auto_focus:
            self.cap.set(cv2.CAP_PROP_FOCUS, self.config.focus)
        if not self.config.auto_exposure:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
        if not self.config.auto_wb:
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.config.wb_temp)
        self.cap.set(cv2.CAP_PROP_GAIN, self.config.gain)

    # ---- Slot / Method Komunikasi dari GUI (via Command Queue) ----
    def request_camera_switch(self, new_index: int) -> None:
        self._cmd_queue.put(("switch_camera", int(new_index)))

    def update_camera_hardware(self, prop_name: str, value) -> None:
        setattr(self.config, prop_name, value)
        self._cmd_queue.put(("update_cam_prop", (prop_name, value)))

    def add_roi_point(self, x: int, y: int) -> None:
        self._cmd_queue.put(("add_point", (int(x), int(y))))

    def undo_roi_point(self) -> None:
        self._cmd_queue.put(("undo_point", None))

    def update_cursor(self, x: int, y: int) -> None:
        self.kursor = (int(x), int(y))

    def trigger_reset_roi(self) -> None:
        self._cmd_queue.put(("reset_roi", None))

    def trigger_recapture_ref(self) -> None:
        self._cmd_queue.put(("recapture_ref", None))

    def trigger_clear_status(self) -> None:
        self._cmd_queue.put(("clear_status", None))

    def trigger_reset_parts(self) -> None:
        self._cmd_queue.put(("reset_parts", None))

    def update_roi_vertex(self, roi_idx: int, v_idx: int, nx: int, ny: int) -> None:
        self._cmd_queue.put(("update_vertex", (int(roi_idx), int(v_idx), int(nx), int(ny))))

    def finish_roi_vertex_drag(self) -> None:
        self._cmd_queue.put(("rebuild_masks", None))

    def trigger_redo_roi(self, idx: int) -> None:
        self._cmd_queue.put(("redo_roi", int(idx)))

    def load_saved_rois(self, saved_rois: List[List[List[int]]]) -> None:
        self._cmd_queue.put(("load_rois", saved_rois))

    def set_jumlah_roi(self, count: int) -> None:
        self.config.jumlah_roi = int(count)
        self._cmd_queue.put(("set_jumlah_roi", int(count)))

    # ---- Helper Pemrosesan Gambar & ROI ----
    def preprocess_for_mediapipe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Mempertajam siluet dan lipatan jari sarung tangan dengan CLAHE dan Pseudo-Skin Tone."""
        # 1. Ubah ke Grayscale untuk membuang warna sarung tangan (biru/putih/hitam)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Invert warna jika sarung tangan berwarna hitam pekat
        if self.config.clahe_invert_dark:
            gray = cv2.bitwise_not(gray)

        # 3. Tajamkan kontras lipatan jari dengan CLAHE
        limit = max(0.1, float(self.config.clahe_clip_limit))
        if self._clahe is None or abs(self._cached_clahe_limit - limit) > 1e-3:
            self._clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
            self._cached_clahe_limit = limit

        enhanced = self._clahe.apply(gray)

        # 4. Berikan tint warna kulit (Skin-Tone / Warm Sepia) dalam format RGB via OpenCV LUT
        r = cv2.LUT(enhanced, self.lut_r)
        g = cv2.LUT(enhanced, self.lut_g)
        b = cv2.LUT(enhanced, self.lut_b)
        return cv2.merge([r, g, b])

    def get_rtm_hand(self) -> RTMHand:
        if self.rtm_hand is None:
            self.rtm_hand = RTMHand(device="cpu", backend="onnxruntime")
        return self.rtm_hand

    def _create_mp_hands_instance(self) -> mp.solutions.hands.Hands:
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise RuntimeError("MediaPipe solutions.hands tidak tersedia pada versi mediapipe ini.")
        return mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

    def get_mp_hands(self, region_idx: Optional[int] = None) -> mp.solutions.hands.Hands:
        if region_idx is None:
            if self.mp_hands is None:
                self.mp_hands = self._create_mp_hands_instance()
            return self.mp_hands
        if region_idx not in self.mp_hands_regions:
            self.mp_hands_regions[region_idx] = self._create_mp_hands_instance()
        return self.mp_hands_regions[region_idx]

    @staticmethod
    def _is_duplicate_hand(cand: dict, existing_list: List[dict], dist_thr: float = 40.0) -> bool:
        """Mencegah deteksi ganda saat beberapa region crop saling tumpang tindih."""
        cx, cy = cand["palm_center"]
        for h in existing_list:
            hx, hy = h["palm_center"]
            if ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5 < dist_thr:
                return True
        return False

    def detect_hands_with_crop(
        self, frame: np.ndarray
    ) -> Tuple[List[dict], List[Tuple[int, int, int, int]]]:
        """Mendeteksi tangan dengan cropping area ROI menggunakan backend aktif (RTMlib / MediaPipe)."""
        tangan: List[dict] = []
        crop_boxes: List[Tuple[int, int, int, int]] = []
        is_rtm = (self.config.hand_backend == "rtmlib")

        # Jika fitur crop dimatikan atau belum ada poligon ROI yang dibuat: gunakan full frame
        if not self.config.use_roi_crop or len(self.rois) == 0:
            regions = [(0, 0, self.frame_w, self.frame_h)]
        else:
            padding = max(0, int(self.config.roi_crop_padding))
            regions = []

            if self.config.roi_crop_mode == "individual":
                # Mode individual: crop tiap poligon ROI secara terpisah
                for poly in self.rois:
                    bx, by, bw, bh = cv2.boundingRect(poly)
                    x1 = max(0, bx - padding)
                    y1 = max(0, by - padding)
                    x2 = min(self.frame_w, bx + bw + padding)
                    y2 = min(self.frame_h, by + bh + padding)
                    if x2 > x1 and y2 > y1:
                        regions.append((x1, y1, x2, y2))
            else:
                # Mode "union": crop kotak gabungan yang melingkupi seluruh ROI
                min_x = min(int(poly[:, 0].min()) for poly in self.rois)
                min_y = min(int(poly[:, 1].min()) for poly in self.rois)
                max_x = max(int(poly[:, 0].max()) for poly in self.rois)
                max_y = max(int(poly[:, 1].max()) for poly in self.rois)
                x1 = max(0, min_x - padding)
                y1 = max(0, min_y - padding)
                x2 = min(self.frame_w, max_x + padding)
                y2 = min(self.frame_h, max_y + padding)
                if x2 > x1 and y2 > y1:
                    regions.append((x1, y1, x2, y2))
            crop_boxes = regions

        use_per_region_mp = (
            self.config.use_roi_crop
            and self.config.roi_crop_mode == "individual"
            and len(regions) > 1
        )
        preview_patches: List[Tuple[int, int, int, int, np.ndarray]] = []

        for idx_reg, (x1, y1, x2, y2) in enumerate(regions):
            crop = frame[y1:y2, x1:x2]
            cw = x2 - x1
            ch = y2 - y1
            if cw <= 10 or ch <= 10:
                continue

            if is_rtm:
                # ==========================
                # Backend 1: RTMlib (RTMPose)
                # ==========================
                tracker = self.get_rtm_hand()
                if self.config.use_clahe:
                    crop_rgb = self.preprocess_for_mediapipe(crop)
                    crop_feed = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    if self.config.show_clahe_preview:
                        preview_patches.append((x1, y1, x2, y2, crop_feed))
                else:
                    crop_feed = crop

                keypoints, scores = tracker(crop_feed)
                thr = float(self.config.rtm_score_thr)

                if len(keypoints) > 0:
                    for kpts, scs in zip(keypoints, scores):
                        valid_mask = [float(s) >= thr for s in scs]
                        valid_count = sum(valid_mask)
                        mean_score = float(np.mean(scs)) if len(scs) > 0 else 0.0

                        # Validasi tangan berdasarkan rata-rata skor dan jumlah keypoint valid
                        if valid_count < 5 or mean_score < thr:
                            continue

                        raw_points = [
                            (int(x1 + kpts[i][0]), int(y1 + kpts[i][1]))
                            for i in range(len(kpts))
                        ]
                        keypoints_21: List[Optional[Tuple[int, int]]] = [
                            raw_points[i] if valid_mask[i] else None
                            for i in range(len(raw_points))
                        ]

                        # Filter per keypoint: hanya titik dengan skor >= thr yang dipakai
                        fingertips = [
                            raw_points[i]
                            for i in UJUNG_JARI
                            if i < len(raw_points) and valid_mask[i]
                        ]
                        dorsum_points = [
                            raw_points[i]
                            for i in SENDI_PUNGGUNG
                            if i < len(raw_points) and valid_mask[i]
                        ]
                        all_points = [
                            raw_points[i]
                            for i in range(len(raw_points))
                            if valid_mask[i]
                        ]
                        if not all_points:
                            continue

                        centroid_pts = dorsum_points if dorsum_points else all_points
                        pc_x = int(sum(p[0] for p in centroid_pts) / len(centroid_pts))
                        pc_y = int(sum(p[1] for p in centroid_pts) / len(centroid_pts))

                        hull = (
                            cv2.convexHull(np.array(all_points, dtype=np.int32))
                            if len(all_points) >= 3
                            else np.empty((0, 1, 2), dtype=np.int32)
                        )

                        hand_entry = {
                            "fingertips": fingertips,
                            "palm_center": (pc_x, pc_y),
                            "dorsum_points": dorsum_points,
                            "all_points": all_points,
                            "keypoints_21": keypoints_21,
                            "hull": hull,
                        }
                        if not self._is_duplicate_hand(hand_entry, tangan):
                            tangan.append(hand_entry)

            else:
                # ==========================
                # Backend 2: Google MediaPipe Hands
                # ==========================
                mp_tracker = self.get_mp_hands(region_idx=idx_reg if use_per_region_mp else None)
                if self.config.use_clahe:
                    crop_input = self.preprocess_for_mediapipe(crop)
                else:
                    crop_input = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                res = mp_tracker.process(crop_input)
                if res.multi_hand_landmarks:
                    for hand in res.multi_hand_landmarks:
                        fingertips = [
                            (int(x1 + hand.landmark[i].x * cw), int(y1 + hand.landmark[i].y * ch))
                            for i in UJUNG_JARI
                        ]
                        dorsum_points = [
                            (int(x1 + hand.landmark[i].x * cw), int(y1 + hand.landmark[i].y * ch))
                            for i in SENDI_PUNGGUNG
                        ]
                        pc_x = int(sum(p[0] for p in dorsum_points) / len(dorsum_points))
                        pc_y = int(sum(p[1] for p in dorsum_points) / len(dorsum_points))
                        all_points = [
                            (int(x1 + hand.landmark[i].x * cw), int(y1 + hand.landmark[i].y * ch))
                            for i in range(len(hand.landmark))
                        ]
                        hull = cv2.convexHull(np.array(all_points, dtype=np.int32))

                        hand_entry = {
                            "fingertips": fingertips,
                            "palm_center": (pc_x, pc_y),
                            "dorsum_points": dorsum_points,
                            "all_points": all_points,
                            "keypoints_21": all_points,
                            "hull": hull,
                        }
                        if not self._is_duplicate_hand(hand_entry, tangan):
                            tangan.append(hand_entry)

                if self.config.use_clahe and self.config.show_clahe_preview:
                    preview_patches.append((x1, y1, x2, y2, cv2.cvtColor(crop_input, cv2.COLOR_RGB2BGR)))

        # Terapkan preview CLAHE setelah seluruh crop selesai diproses agar tidak mengontaminasi crop berikutnya
        for px1, py1, px2, py2, patch_bgr in preview_patches:
            frame[py1:py2, px1:px2] = patch_bgr

        return tangan, crop_boxes

    @staticmethod
    def siapkan_gray(img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 0)

    @staticmethod
    def _normalize_and_validate_roi(pts, min_area: float = 100.0) -> Optional[np.ndarray]:
        """Mengurutkan 4 titik sudut secara konveks (cv2.convexHull) dan menolak bentuk bersilang/degeneratif."""
        try:
            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 2)
        except Exception:
            return None
        if len(pts_arr) != 4:
            return None
        hull = cv2.convexHull(pts_arr, clockwise=True).reshape(-1, 2)
        if len(hull) != 4:
            return None
        if not cv2.isContourConvex(hull):
            return None
        if float(cv2.contourArea(hull.astype(np.float32))) < min_area:
            return None
        return hull.astype(np.int32)

    @staticmethod
    def hull_roi_overlap_ratio(poly: np.ndarray, hull: np.ndarray) -> float:
        """Menghitung rasio luas irisan convex hull tangan dan poligon ROI konveks terhadap luas ROI."""
        if poly is None or hull is None or len(poly) < 3 or len(hull) < 3:
            return 0.0
        poly_f = poly.reshape(-1, 2).astype(np.float32)
        hull_f = hull.reshape(-1, 2).astype(np.float32)
        roi_area = float(cv2.contourArea(poly_f))
        hull_area = float(cv2.contourArea(hull_f))
        if roi_area <= 1.0 or hull_area <= 1.0:
            return 0.0
        try:
            inter_area, _ = cv2.intersectConvexConvex(poly_f, hull_f)
        except cv2.error:
            return 0.0
        return float(max(0.0, inter_area) / roi_area)

    @staticmethod
    def _normalize_brightness(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Menormalisasi perubahan cahaya global sebelum frame differencing."""
        delta = float(np.mean(ref)) - float(np.mean(img))
        if abs(delta) < 0.5:
            return img
        return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    def _clean_diff_mask(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        thr: Optional[int] = None,
    ) -> np.ndarray:
        """Menghitung mask piksel berubah dengan normalisasi brightness dan morphological opening."""
        pixel_thr = self.config.ambang_piksel if thr is None else thr
        norm_a = self._normalize_brightness(img_a, img_b)
        diff = cv2.absdiff(norm_a, img_b)
        if roi_mask is not None:
            binary = ((diff > pixel_thr) & (roi_mask > 0)).astype(np.uint8) * 255
        else:
            binary = (diff > pixel_thr).astype(np.uint8) * 255
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._morph_kernel)

    def rasio_perubahan(self, gray_eval: np.ndarray, idx: int) -> float:
        if self.referensi is None or idx >= len(self.masks):
            return 0.0
        cleaned = self._clean_diff_mask(gray_eval, self.referensi, self.masks[idx], self.config.ambang_piksel)
        berubah = np.count_nonzero(cleaned)
        luas = max(np.count_nonzero(self.masks[idx]), 1)
        return float(berubah / luas)

    def rasio_gerak_roi(self, gray_now: np.ndarray, idx: int) -> float:
        """Mendeteksi objek bergerak di dalam ROI melalui frame differencing antar-frame berturut-turut."""
        if self._prev_gray is None or self._prev_gray.shape != gray_now.shape or idx >= len(self.masks):
            return 0.0
        motion_pix_thr = max(10, self.config.ambang_piksel // 2)
        cleaned = self._clean_diff_mask(gray_now, self._prev_gray, self.masks[idx], motion_pix_thr)
        bergerak = np.count_nonzero(cleaned)
        luas = max(np.count_nonzero(self.masks[idx]), 1)
        return float(bergerak / luas)

    @staticmethod
    def isi_transparan(img: np.ndarray, poly: np.ndarray, warna: Tuple[int, int, int], alpha: float = 0.28) -> None:
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], warna)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def roi_aktif(self) -> Optional[int]:
        total = min(len(self.rois), self.config.jumlah_roi)
        for i in range(total):
            if i < len(self.status) and self.status[i] != 2:
                return i
        return None

    def _rebuild_masks(self) -> None:
        self.masks.clear()
        for poly in self.rois:
            m = np.zeros((self.frame_h, self.frame_w), np.uint8)
            cv2.fillPoly(m, [poly], 255)
            self.masks.append(m)

    def _invalidate_reference(self) -> None:
        self.referensi = None
        self._no_hand_frames = 0
        self._stable_ref_buffer.clear()

    def _do_reset_roi(self) -> None:
        self.rois.clear()
        self.masks.clear()
        self.titik_klik.clear()
        self._redo_roi_idx = None
        self._invalidate_reference()
        self._close_mp_hands_regions()
        self._init_roi_arrays()

    def _process_pending_actions(self, gray_now: Optional[np.ndarray]) -> None:
        cam_props_to_apply: dict = {}

        while True:
            try:
                cmd, arg = self._cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "switch_camera":
                self.config.camera_index = int(arg)
                self._open_camera()

            elif cmd == "update_cam_prop":
                prop_name, val = arg
                cam_props_to_apply[prop_name] = val

            elif cmd == "reset_roi":
                self._do_reset_roi()

            elif cmd == "set_jumlah_roi":
                self.config.jumlah_roi = int(arg)
                self._do_reset_roi()

            elif cmd == "load_rois":
                loaded = arg
                self._do_reset_roi()
                for pts in loaded[: self.config.jumlah_roi]:
                    poly = self._normalize_and_validate_roi(pts)
                    if poly is not None:
                        self.rois.append(poly)
                    else:
                        self.log_event.emit("WARN", "Satu ROI tersimpan diabaikan karena bentuk poligon tidak valid.")
                self._rebuild_masks()
                self._invalidate_reference()

            elif cmd == "redo_roi":
                idx = int(arg)
                self._redo_roi_idx = idx
                self.titik_klik.clear()
                name = self.config.get_roi_name(idx)
                self.log_event.emit(
                    "INFO",
                    f"Mode gambar ulang untuk {name} (ROI #{idx + 1}) aktif. Klik 4 sudut baru di video.",
                )

            elif cmd == "update_vertex":
                roi_idx, v_idx, nx, ny = arg
                if 0 <= roi_idx < len(self.rois) and 0 <= v_idx < 4:
                    nx = max(0, min(nx, self.frame_w - 1))
                    ny = max(0, min(ny, self.frame_h - 1))
                    cand = self.rois[roi_idx].copy()
                    cand[v_idx] = [nx, ny]
                    if cv2.isContourConvex(cand) and float(cv2.contourArea(cand.astype(np.float32))) >= 100.0:
                        self.rois[roi_idx] = cand
                        m = np.zeros((self.frame_h, self.frame_w), np.uint8)
                        cv2.fillPoly(m, [self.rois[roi_idx]], 255)
                        if roi_idx < len(self.masks):
                            self.masks[roi_idx] = m
                        self.config.saved_rois = [r.tolist() for r in self.rois]

            elif cmd == "rebuild_masks":
                for i in range(len(self.rois)):
                    norm_poly = self._normalize_and_validate_roi(self.rois[i])
                    if norm_poly is not None:
                        self.rois[i] = norm_poly
                self._rebuild_masks()
                self.config.saved_rois = [r.tolist() for r in self.rois]
                self.config.save(CONFIG_FILE)
                self._invalidate_reference()
                self.log_event.emit("INFO", "Bentuk sudut ROI diperbarui. Menunggu area bersih untuk referensi.")

            elif cmd == "undo_point":
                if self.titik_klik:
                    self.titik_klik.pop()
                elif self._redo_roi_idx is not None:
                    self._redo_roi_idx = None
                    self.log_event.emit("INFO", "Batal menggambar ulang ROI.")

            elif cmd == "add_point":
                px, py = arg
                px = max(0, min(px, self.frame_w - 1))
                py = max(0, min(py, self.frame_h - 1))

                if self._redo_roi_idx is not None:
                    # Mode gambar ulang ROI tertentu
                    self.titik_klik.append((px, py))
                    if len(self.titik_klik) == 4:
                        poly = self._normalize_and_validate_roi(self.titik_klik)
                        self.titik_klik.clear()
                        if poly is None:
                            self.log_event.emit(
                                "WARN",
                                "Poligon ROI tidak valid (titik segaris/cekung/terlalu kecil). Silakan klik ulang 4 sudut.",
                            )
                        else:
                            target_idx = self._redo_roi_idx
                            if target_idx < len(self.rois):
                                self.rois[target_idx] = poly
                            else:
                                self.rois.append(poly)
                            self._rebuild_masks()
                            self._redo_roi_idx = None
                            self.config.saved_rois = [r.tolist() for r in self.rois]
                            self.config.save(CONFIG_FILE)
                            self._invalidate_reference()
                            r_name = self.config.get_roi_name(target_idx)
                            self.log_event.emit("SUCCESS", f"{r_name} berhasil digambar ulang dan disimpan.")
                elif len(self.rois) < self.config.jumlah_roi:
                    # Mode pembuatan ROI baru
                    self.titik_klik.append((px, py))
                    if len(self.titik_klik) == 4:
                        poly = self._normalize_and_validate_roi(self.titik_klik)
                        self.titik_klik.clear()
                        if poly is None:
                            self.log_event.emit(
                                "WARN",
                                "Poligon ROI tidak valid (titik segaris/cekung/terlalu kecil). Silakan klik ulang 4 sudut.",
                            )
                        else:
                            m = np.zeros((self.frame_h, self.frame_w), np.uint8)
                            cv2.fillPoly(m, [poly], 255)
                            self.rois.append(poly)
                            self.masks.append(m)
                            self.config.saved_rois = [r.tolist() for r in self.rois]
                            self.config.save(CONFIG_FILE)
                            self._invalidate_reference()

            elif cmd == "recapture_ref":
                if len(self.rois) == self.config.jumlah_roi:
                    self._invalidate_reference()
                    self._init_roi_arrays()
                    self._cycle_cooldown_until = None
                    self.log_event.emit("INFO", "Menunggu area kerja bersih tanpa tangan untuk mengambil referensi baru...")

            elif cmd == "clear_status":
                self._init_roi_arrays()
                self._cycle_cooldown_until = None
                self.log_event.emit("INFO", "Status urutan langkah direset manual ke Langkah 1.")

            elif cmd == "reset_parts":
                self.parts_completed = 0
                self.log_event.emit("INFO", "Counter Part telah direset ke 0.")

        if cam_props_to_apply and self.cap is not None and self.cap.isOpened():
            for name, val in cam_props_to_apply.items():
                if name == "auto_focus":
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if val else 0)
                    if not val:
                        self.cap.set(cv2.CAP_PROP_FOCUS, self.config.focus)
                elif name == "auto_exposure":
                    if val:
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    else:
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
                elif name == "auto_wb":
                    self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if val else 0)
                    if not val:
                        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.config.wb_temp)
                elif name == "focus" and not self.config.auto_focus:
                    self.cap.set(cv2.CAP_PROP_FOCUS, int(val))
                elif name == "exposure" and not self.config.auto_exposure:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, int(val))
                elif name == "wb_temp" and not self.config.auto_wb:
                    self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, int(val))
                elif name == "gain":
                    self.cap.set(cv2.CAP_PROP_GAIN, int(val))

    def run(self) -> None:
        self.running = True
        try:
            self._run_worker_loop()
        except Exception as e:
            print(f"[Error Fatal CameraWorker] {e}")
        finally:
            self._cleanup_resources()

    def _run_worker_loop(self) -> None:
        self._open_camera()

        # Pre-load model di awal agar tidak terjadi frame stutter saat mulai
        self._preload_selected_backend()

        # Muat ROI tersimpan jika ada dan jumlahnya cocok
        if self.config.saved_rois and len(self.config.saved_rois) == self.config.jumlah_roi:
            self.load_saved_rois(self.config.saved_rois)

        prev_time = time.perf_counter()
        fps = 0.0

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                self._process_pending_actions(None)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            if not self.running:
                break

            # Pastikan ukuran array sinkron dengan jumlah ROI
            self._ensure_arrays_size()

            if (frame.shape[1], frame.shape[0]) != (self.frame_w, self.frame_h):
                self.frame_h, self.frame_w = frame.shape[:2]
                if self.rois:
                    self._rebuild_masks()

            gray_now = self.siapkan_gray(frame)
            self._process_pending_actions(gray_now)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            jumlah_roi = self.config.jumlah_roi
            banner_text = ""
            banner_state = "setup"  # setup | active | error | done
            roi_telemetry = []
            steps_completed = 0

            # ==========================================================
            # MODE 1: SETUP POLIGON ROI (KLIK 4 SUDUT PER ROI)
            # ==========================================================
            if len(self.rois) < jumlah_roi or self._redo_roi_idx is not None:
                if self._redo_roi_idx is not None:
                    r_name = self.config.get_roi_name(self._redo_roi_idx)
                    titik_ke = len(self.titik_klik) + 1
                    banner_text = (
                        f"MODE GAMBAR ULANG [{r_name}]: Klik sudut {titik_ke}/4 pada video "
                        f"(Klik kanan untuk batal)"
                    )
                else:
                    n = len(self.rois) + 1
                    titik_ke = len(self.titik_klik) + 1
                    banner_text = (
                        f"MODE SETUP ROI {n}/{jumlah_roi}: Klik sudut {titik_ke}/4 pada video "
                        f"(Klik kanan untuk batal titik)"
                    )
                banner_state = "setup"

                for i, poly in enumerate(self.rois):
                    warna = WARNA_ROI_BGR[i % len(WARNA_ROI_BGR)]
                    self.isi_transparan(frame, poly, warna, alpha=0.18)
                    cv2.polylines(frame, [poly], True, warna, 2)
                    for pt in poly:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, warna, 1, cv2.LINE_AA)
                    r_name = self.config.get_roi_name(i)
                    lbl_text = f"{r_name} (Diedit...)" if self._redo_roi_idx == i else f"{r_name} (Siap)"
                    cv2.putText(
                        frame,
                        lbl_text,
                        (int(poly[:, 0].min()), max(int(poly[:, 1].min()) - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        warna,
                        2,
                    )

                if self.titik_klik:
                    for a, b in zip(self.titik_klik, self.titik_klik[1:] + [self.kursor]):
                        cv2.line(frame, a, b, (0, 255, 255), 2)
                    for p in self.titik_klik:
                        cv2.circle(frame, p, 6, (0, 255, 255), -1)

                for i in range(jumlah_roi):
                    if i < len(self.rois):
                        roi_telemetry.append({
                            "index": i,
                            "state": "READY",
                            "label": "Poligon Siap",
                            "ratio": 0.0,
                            "debounce": "",
                        })
                    elif i == len(self.rois):
                        roi_telemetry.append({
                            "index": i,
                            "state": "SETUP",
                            "label": f"Klik sudut {len(self.titik_klik)}/4",
                            "ratio": 0.0,
                            "debounce": "",
                        })
                    else:
                        roi_telemetry.append({
                            "index": i,
                            "state": "LOCKED",
                            "label": "Menunggu dibuat",
                            "ratio": 0.0,
                            "debounce": "",
                        })

            # ==========================================================
            # MODE 2: MONITORING DETEKSI TANGAN & PERUBAHAN PIKSEL ROI
            # ==========================================================
            else:
                # Deteksi tangan dengan Dual-Backend (RTMlib / MediaPipe + ROI Cropping) pada frame bersih
                tangan, crop_boxes = self.detect_hands_with_crop(frame)

                # Cek apakah ada objek bergerak di dalam salah satu ROI (antar-frame differencing)
                motion_ratios = [self.rasio_gerak_roi(gray_now, i) for i in range(len(self.rois))]
                any_roi_motion = any(mr >= self.config.motion_fallback_thr for mr in motion_ratios)

                # Akumulasi frame bersih (tanpa tangan & tanpa gerakan) untuk penguncian referensi stabil
                needed_clean = max(1, int(self.config.ref_clean_frames))
                if len(tangan) == 0 and not any_roi_motion:
                    self._no_hand_frames += 1
                    self._stable_ref_buffer.append(gray_now.copy())
                    if len(self._stable_ref_buffer) > needed_clean:
                        self._stable_ref_buffer.pop(0)
                else:
                    self._no_hand_frames = 0
                    self._stable_ref_buffer.clear()

                # Kunci referensi hanya jika area benar-benar bersih tanpa tangan selama N frame
                if self.referensi is None or self.referensi.shape != gray_now.shape:
                    if self._no_hand_frames >= needed_clean and len(self._stable_ref_buffer) >= needed_clean:
                        self.referensi = np.mean(self._stable_ref_buffer, axis=0).astype(np.uint8)
                        self.log_event.emit(
                            "INFO",
                            f"Referensi background bersih berhasil dikunci (rata-rata {needed_clean} frame stabil).",
                        )

                # Opsional: tampilkan highlight piksel yang berubah jika Debug Diff aktif (visualisasi overlay)
                if self.config.debug_diff and self.referensi is not None:
                    combined_mask = np.zeros_like(gray_now)
                    for m in self.masks:
                        combined_mask = cv2.bitwise_or(combined_mask, m)
                    cleaned_all = self._clean_diff_mask(gray_now, self.referensi, combined_mask, self.config.ambang_piksel)
                    frame[cleaned_all > 0] = [0, 0, 255]

                # Visualisasi batas kotak crop jika diaktifkan
                if self.config.use_roi_crop and self.config.show_crop_box:
                    for (cx1, cy1, cx2, cy2) in crop_boxes:
                        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 128), 1, cv2.LINE_AA)

                aktif = self.roi_aktif()
                salah_urut = None
                overlap_thr = float(self.config.hand_overlap_thr)
                debounce_err = max(2, int(self.config.debounce_masuk))

                for idx, poly in enumerate(self.rois):
                    ada_objek_bergerak = (
                        idx < len(motion_ratios) and motion_ratios[idx] >= self.config.motion_fallback_thr
                    )

                    if self.config.hand_tracking_mode == "full_hand":
                        # Mode Seluruh Tangan: Menggunakan rasio luas irisan convex hull tangan dan ROI konveks
                        tangan_masuk_roi = False
                        ada_tangan_model = False
                        for h in tangan:
                            # Hitung rasio luas irisan poligon convex hull tangan dengan poligon ROI konveks
                            overlap_ratio = self.hull_roi_overlap_ratio(poly, h["hull"])
                            pc_in = (
                                cv2.pointPolygonTest(
                                    poly,
                                    (float(h["palm_center"][0]), float(h["palm_center"][1])),
                                    False,
                                )
                                >= 0
                            )
                            titik_di_roi = sum(
                                1
                                for p in h["all_points"]
                                if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
                            )

                            if overlap_ratio >= overlap_thr or (pc_in and titik_di_roi >= 5):
                                tangan_masuk_roi = True
                            if overlap_ratio >= overlap_thr * 0.5 or (pc_in and titik_di_roi >= 3):
                                ada_tangan_model = True

                        semua_jari = tangan_masuk_roi

                    elif self.config.hand_tracking_mode == "palm_center":
                        # Mode Pusat Punggung Tangan: Cukup pusat punggung tangan masuk ke dalam ROI (Bebas Menggenggam)
                        tangan_masuk_roi = any(
                            cv2.pointPolygonTest(poly, h["palm_center"], False) >= 0
                            for h in tangan
                        )
                        ada_tangan_model = any(
                            cv2.pointPolygonTest(poly, h["palm_center"], False) >= 0
                            or sum(cv2.pointPolygonTest(poly, p, False) >= 0 for p in h["dorsum_points"]) >= 2
                            for h in tangan
                        )
                        semua_jari = tangan_masuk_roi
                    else:
                        # Mode 5 Ujung Jari: Wajib kelima ujung jari berada di dalam ROI
                        jumlah = [
                            sum(cv2.pointPolygonTest(poly, p, False) >= 0 for p in h["fingertips"])
                            for h in tangan
                        ]
                        semua_jari = any(j == 5 for j in jumlah)
                        ada_tangan_model = any(j >= 2 for j in jumlah)
                        tangan_masuk_roi = semua_jari

                    # Gabungkan deteksi model dengan fallback frame differencing (objek bergerak di ROI)
                    # agar saat sarung tangan gagal terdeteksi sesaat oleh model, hitungan keluar tidak jalan terlalu cepat
                    ada_tangan = ada_tangan_model or ada_objek_bergerak

                    debounce_info = ""

                    # Jika referensi belum terkunci (menunggu tangan keluar dari frame), tahan evaluasi langkah
                    if self.referensi is None:
                        warna = WARNA_ROI_BGR[idx % len(WARNA_ROI_BGR)]
                        teks = f"Menunggu Ref Bersih ({self._no_hand_frames}/{needed_clean})"
                        state_code = "READY"
                    # 1. ROI sudah DONE
                    elif self.status[idx] == 2:
                        self.error_count[idx] = 0
                        warna, teks = (0, 200, 0), "DONE"
                        self.isi_transparan(frame, poly, warna, alpha=0.28)
                        state_code = "DONE"

                    # 2. ROI aktif (giliran sekarang)
                    elif idx == aktif:
                        self.error_count[idx] = 0
                        if self.status[idx] == 0:
                            self.masuk[idx] = self.masuk[idx] + 1 if semua_jari else 0
                            if self.masuk[idx] > 0:
                                debounce_info = f"Masuk: {self.masuk[idx]}/{self.config.debounce_masuk}"
                            if self.masuk[idx] >= self.config.debounce_masuk:
                                self.status[idx] = 1
                                self.keluar[idx] = 0
                                self._post_hand_frames[idx].clear()

                        elif self.status[idx] == 1:
                            if not ada_tangan:
                                self.keluar[idx] += 1
                                self._post_hand_frames[idx].append(gray_now.copy())
                                max_post = max(2, min(5, int(self.config.debounce_keluar)))
                                if len(self._post_hand_frames[idx]) > max_post:
                                    self._post_hand_frames[idx].pop(0)
                            else:
                                self.keluar[idx] = 0
                                self._post_hand_frames[idx].clear()

                            debounce_info = f"Keluar: {self.keluar[idx]}/{self.config.debounce_keluar}"
                            if self.keluar[idx] >= self.config.debounce_keluar:
                                # Bandingkan rata-rata beberapa frame stabil sesudah tangan keluar terhadap referensi
                                if self._post_hand_frames[idx]:
                                    gray_after = np.mean(self._post_hand_frames[idx], axis=0).astype(np.uint8)
                                else:
                                    gray_after = gray_now
                                self._post_hand_frames[idx].clear()

                                rasio = self.rasio_perubahan(gray_after, idx)
                                self.rasio_terakhir[idx] = rasio
                                if rasio >= self.config.ambang_rasio:
                                    self.status[idx] = 2
                                    self.log_event.emit(
                                        "SUCCESS",
                                        f"Langkah {idx + 1} (ROI {idx + 1}) SELESAI — Perubahan: {rasio * 100:.1f}%",
                                    )
                                else:
                                    self.status[idx] = 0
                                    self.log_event.emit(
                                        "WARN",
                                        f"Langkah {idx + 1} tidak terdeteksi: Diff {rasio * 100:.1f}% < target {self.config.ambang_rasio * 100:.1f}%",
                                    )
                                self.masuk[idx] = 0

                        if self.status[idx] == 2:
                            warna, teks = (0, 200, 0), "DONE"
                            self.isi_transparan(frame, poly, warna, alpha=0.28)
                            state_code = "DONE"
                        elif self.status[idx] == 1:
                            warna, teks = (0, 165, 255), "Tangan di ROI..."
                            self.isi_transparan(frame, poly, warna, alpha=0.20)
                            state_code = "HAND_IN"
                        else:
                            warna = WARNA_ROI_BGR[idx % len(WARNA_ROI_BGR)]
                            teks = "LANGKAH AKTIF"
                            state_code = "ACTIVE"

                    # 3. ROI terkunci (belum gilirannya) — dengan debounce untuk status ERROR
                    else:
                        state_code = "LOCKED"
                        target_aktif = (aktif + 1) if aktif is not None else 1
                        warna, teks = WARNA_KUNCI_BGR, f"Terkunci (Langkah {target_aktif} dulu)"
                        if tangan_masuk_roi:
                            self.error_count[idx] += 1
                        else:
                            self.error_count[idx] = 0

                        if self.error_count[idx] >= debounce_err:
                            salah_urut = idx
                            warna = (0, 0, 255)
                            teks = "URUTAN SALAH!"
                            state_code = "ERROR"
                            self.isi_transparan(frame, poly, warna, alpha=0.30)

                    ketebalan = 3 if self.status[idx] == 2 else 2
                    cv2.polylines(frame, [poly], True, warna, ketebalan)
                    # Gambar titik sudut interaktif (draggable handles)
                    for pt in poly:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, warna, 1, cv2.LINE_AA)

                    r_name = self.config.get_roi_name(idx)
                    cv2.putText(
                        frame,
                        f"{r_name}: {teks}",
                        (int(poly[:, 0].min()), max(int(poly[:, 1].min()) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        warna,
                        2,
                        cv2.LINE_AA,
                    )

                    roi_telemetry.append({
                        "index": idx,
                        "state": state_code,
                        "label": teks,
                        "ratio": self.rasio_terakhir[idx],
                        "debounce": debounce_info,
                    })

                # Visualisasi tangan sesuai mode pelacakan aktif
                for h in tangan:
                    if self.config.hand_tracking_mode == "full_hand":
                        # 1. Gambar selubung transparan convex hull seluruh tangan
                        if "hull" in h and len(h["hull"]) >= 3:
                            cv2.polylines(frame, [h["hull"]], True, (0, 255, 170), 1, cv2.LINE_AA)
                            self.isi_transparan(frame, h["hull"], (0, 255, 170), alpha=0.12)

                        # 2. Gambar garis kerangka sambungan tangan lengkap (21 sambungan tulang)
                        kpts_21 = h.get("keypoints_21", h["all_points"])
                        for i_a, i_b in SAMBUNGAN_TANGAN:
                            if (
                                i_a < len(kpts_21)
                                and i_b < len(kpts_21)
                                and kpts_21[i_a] is not None
                                and kpts_21[i_b] is not None
                            ):
                                cv2.line(frame, kpts_21[i_a], kpts_21[i_b], (0, 230, 255), 2, cv2.LINE_AA)

                        # 3. Gambar sendi-sendi tangan lengkap
                        for idx_pt, p in enumerate(kpts_21):
                            if p is None:
                                continue
                            if idx_pt in UJUNG_JARI:
                                cv2.circle(frame, p, 5, (0, 255, 255), -1)   # Ujung jari kuning cerah
                                cv2.circle(frame, p, 6, (0, 180, 220), 1)
                            elif idx_pt == 0:
                                cv2.circle(frame, p, 6, (255, 120, 0), -1)   # Pergelangan oranye-biru
                            else:
                                cv2.circle(frame, p, 3, (0, 255, 128), -1)   # Buku ruas jari hijau

                        # 4. Label dan titik pusat tangan
                        pc = h["palm_center"]
                        cv2.circle(frame, pc, 7, (0, 255, 255), -1)
                        cv2.circle(frame, pc, 9, (0, 0, 0), 2)
                        cv2.putText(
                            frame,
                            "Seluruh Tangan",
                            (pc[0] + 12, pc[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    elif self.config.hand_tracking_mode == "palm_center":
                        # Gambar sendi-sendi punggung tangan (wrist + buku pangkal jari)
                        if h["dorsum_points"]:
                            for p in h["dorsum_points"]:
                                cv2.circle(frame, p, 4, (0, 200, 255), -1)
                            # Garis kerangka penghubung punggung tangan
                            wrist = h["dorsum_points"][0]
                            for kn in h["dorsum_points"][1:]:
                                cv2.line(frame, wrist, kn, (0, 200, 255), 1, cv2.LINE_AA)
                        # Gambar titik pusat punggung tangan (Pusat Tracking)
                        pc = h["palm_center"]
                        cv2.circle(frame, pc, 9, (0, 255, 255), -1)
                        cv2.circle(frame, pc, 11, (0, 0, 0), 2)
                        cv2.putText(
                            frame,
                            "Pusat Punggung",
                            (pc[0] + 12, pc[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        # Mode 5 Ujung Jari
                        for p in h["fingertips"]:
                            cv2.circle(frame, p, 6, (255, 255, 0), -1)
                            cv2.circle(frame, p, 6, (0, 0, 0), 1)

                # Tentukan pesan banner utama & logika Auto-Increment Part
                aktif_now = self.roi_aktif()
                steps_completed = sum(
                    1 for i in range(min(len(self.rois), jumlah_roi))
                    if i < len(self.status) and self.status[i] == 2
                )

                if self.referensi is None:
                    banner_text = (
                        f"MENUNGGU AREA BERSIH: Jauhkan tangan dari ROI untuk mengunci referensi "
                        f"({self._no_hand_frames}/{needed_clean} frame stabil)..."
                    )
                    banner_state = "setup"
                elif aktif_now is None:
                    # Seluruh langkah (misal 3/3) telah selesai!
                    if self._cycle_cooldown_until is None:
                        self._cycle_cooldown_until = now + 1.2
                        self.parts_completed += 1
                        self.log_event.emit(
                            "PART",
                            f"★ PART #{self.parts_completed} SELESAI! ({jumlah_roi}/{jumlah_roi} Langkah Lengkap)"
                        )

                    banner_state = "done"

                    if now >= self._cycle_cooldown_until:
                        # Ambil referensi baru hanya saat tidak ada tangan & tidak ada gerakan selama N frame
                        if self._no_hand_frames >= needed_clean and len(self._stable_ref_buffer) >= needed_clean:
                            self._cycle_cooldown_until = None
                            self._init_roi_arrays()
                            self.referensi = np.mean(self._stable_ref_buffer, axis=0).astype(np.uint8)
                            self.log_event.emit("INFO", f"Mulai siklus Part #{self.parts_completed + 1}...")
                            banner_text = f"★ PART #{self.parts_completed} SELESAI (+1)! Memulai siklus berikutnya..."
                        else:
                            banner_text = (
                                f"★ PART #{self.parts_completed} SELESAI! Jauhkan tangan untuk kunci referensi siklus baru "
                                f"({self._no_hand_frames}/{needed_clean})..."
                            )
                    else:
                        banner_text = f"★ PART #{self.parts_completed} SELESAI (+1)! Menyiapkan siklus berikutnya..."
                else:
                    self._cycle_cooldown_until = None
                    if salah_urut is not None:
                        target_name = self.config.get_roi_name(aktif_now)
                        wrong_name = self.config.get_roi_name(salah_urut)
                        banner_text = f"URUTAN SALAH! Tangan masuk ke [{wrong_name}]. Kerjakan [{target_name}] terlebih dahulu!"
                        banner_state = "error"
                        if now - self._last_error_log_time > 2.0:
                            self._last_error_log_time = now
                            self.log_event.emit(
                                "WARN",
                                f"Urutan salah! Tangan di [{wrong_name}] (Seharusnya [{target_name}])"
                            )
                    else:
                        step_name = self.config.get_roi_name(aktif_now)
                        if self.config.hand_tracking_mode == "full_hand":
                            banner_text = f"GILIRAN KERJA: Ambil [{step_name}] pada ROI {aktif_now + 1} (Bebas / Seluruh Tangan)"
                        elif self.config.hand_tracking_mode == "palm_center":
                            banner_text = f"GILIRAN KERJA: Ambil [{step_name}] pada ROI {aktif_now + 1} (Genggam Bebas)"
                        else:
                            banner_text = f"GILIRAN KERJA: Masukkan 5 jari dan ambil [{step_name}] pada ROI {aktif_now + 1}"
                        banner_state = "active"

            self._prev_gray = gray_now.copy()

            # Konversi BGR ke QImage untuk dikirim ke GUI
            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_out.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_out.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

            self.frame_ready.emit(qimg, w, h)
            self.telemetry_updated.emit({
                "banner_text": banner_text,
                "banner_state": banner_state,
                "rois": roi_telemetry,
                "rois_coords": [r.tolist() for r in self.rois],
                "fps": fps,
                "has_reference": self.referensi is not None,
                "roi_defined": len(self.rois),
                "roi_total": jumlah_roi,
                "steps_completed": steps_completed,
                "parts_completed": self.parts_completed,
            })

    def stop(self) -> None:
        """Menghentikan thread dan merilis kamera secara aman tanpa memutus paksa native driver."""
        self.running = False
        # 1. Tunggu thread keluar dari loop run() dan mengeksekusi blok finally
        if not self.wait(2500):
            print("[Info] Menunggu pelepasan kamera dari thread...")
            # Jika thread terhambat di operasi blocking cap.read(), rilis secara eksplisit
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    print(f"[Peringatan] Gagal rilis cap di stop: {e}")

            # Beri kesempatan thread untuk menyelesaikan pembersihan
            if not self.wait(1500):
                print("[Peringatan] Thread tidak merespons setelah rilis, memaksa terminasi.")
                self.terminate()
                self.wait(500)


class VideoCanvas(QLabel):
    """Widget penampil video interaktif yang memetakan klik/gerakan mouse dan mendukung drag & drop sudut ROI."""

    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal()
    mouse_moved = pyqtSignal(int, int)
    vertex_dragged = pyqtSignal(int, int, int, int)       # (roi_idx, vertex_idx, new_x, new_y)
    vertex_drag_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "background-color: #0d1117; border: 2px solid #30363d; border-radius: 8px;"
        )
        self._frame_w = 1280
        self._frame_h = 720
        self._pixmap_rect = None
        self._rois_coords: List[List[List[int]]] = []
        self._drag_target: Optional[Tuple[int, int]] = None

    def set_rois_coords(self, coords: List[List[List[int]]]) -> None:
        self._rois_coords = coords

    def update_frame(self, qimg: QImage, frame_w: int, frame_h: int) -> None:
        self._frame_w = frame_w
        self._frame_h = frame_h
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

        # Hitung posisi offset gambar di dalam QLabel untuk konversi koordinat mouse
        pw, ph = scaled.width(), scaled.height()
        ox = (self.width() - pw) // 2
        oy = (self.height() - ph) // 2
        self._pixmap_rect = (ox, oy, pw, ph)

    def _map_to_frame(self, pos_x: int, pos_y: int) -> Optional[Tuple[int, int]]:
        if not self._pixmap_rect:
            return None
        ox, oy, pw, ph = self._pixmap_rect
        if pw <= 0 or ph <= 0:
            return None
        rel_x = pos_x - ox
        rel_y = pos_y - oy
        if 0 <= rel_x <= pw and 0 <= rel_y <= ph:
            fx = int(rel_x * self._frame_w / pw)
            fy = int(rel_y * self._frame_h / ph)
            return fx, fy
        return None

    def _find_vertex_near(self, fx: int, fy: int, radius: int = 24) -> Optional[Tuple[int, int]]:
        for roi_idx, poly in enumerate(self._rois_coords):
            for v_idx, pt in enumerate(poly):
                dist = ((fx - pt[0]) ** 2 + (fy - pt[1]) ** 2) ** 0.5
                if dist <= radius:
                    return roi_idx, v_idx
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            mapped = self._map_to_frame(int(event.position().x()), int(event.position().y()))
            if mapped:
                fx, fy = mapped
                target = self._find_vertex_near(fx, fy)
                if target is not None:
                    self._drag_target = target
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return
                self.left_clicked.emit(fx, fy)
        elif event.button() == Qt.MouseButton.RightButton:
            if self._drag_target is not None:
                self._drag_target = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.right_clicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        mapped = self._map_to_frame(int(event.position().x()), int(event.position().y()))
        if mapped:
            fx, fy = mapped
            if self._drag_target is not None:
                roi_idx, v_idx = self._drag_target
                if roi_idx < len(self._rois_coords) and v_idx < len(self._rois_coords[roi_idx]):
                    self._rois_coords[roi_idx][v_idx] = [fx, fy]
                self.vertex_dragged.emit(roi_idx, v_idx, fx, fy)
            else:
                near = self._find_vertex_near(fx, fy)
                if near is not None:
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            self.mouse_moved.emit(fx, fy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_target is not None:
            self._drag_target = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.vertex_drag_finished.emit()
        super().mouseReleaseEvent(event)


class ROICardWidget(QFrame):
    """Kartu indikator status untuk satu ROI dengan penamaan kustom dan tombol gambar ulang spesifik."""

    redo_requested = pyqtSignal(int)
    name_changed = pyqtSignal(int, str)

    def __init__(self, idx: int, config: AppConfig, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.config = config
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        # Input Penamaan Kustom Komponen per ROI
        self.name_edit = QLineEdit(self.config.get_roi_name(idx))
        self.name_edit.setPlaceholderText(f"Nama ROI #{idx + 1}")
        self.name_edit.setToolTip("Klik untuk mengubah nama komponen/langkah ini (misal: Baut, Ring, Balm)")
        self.name_edit.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px; "
            "color: #58a6ff; font-weight: bold; font-size: 11px; padding: 2px 6px;"
        )
        self.name_edit.editingFinished.connect(self._on_name_edited)

        # Tombol Gambar Ulang Khusus ROI ini
        self.btn_redo = QPushButton("↺")
        self.btn_redo.setToolTip(f"Gambar ulang sudut kotak {self.config.get_roi_name(idx)}")
        self.btn_redo.setFixedSize(24, 22)
        self.btn_redo.setStyleSheet(
            "background-color: #21262d; border: 1px solid #30363d; border-radius: 4px; "
            "color: #e6edf3; font-weight: bold; font-size: 12px; padding: 0px;"
        )
        self.btn_redo.clicked.connect(lambda: self.redo_requested.emit(self.idx))

        self.badge_lbl = QLabel("SETUP")
        self.badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_lbl.setFixedHeight(22)
        self.badge_lbl.setMinimumWidth(105)

        top_row.addWidget(self.name_edit, stretch=1)
        top_row.addWidget(self.btn_redo)
        top_row.addWidget(self.badge_lbl)

        bottom_row = QHBoxLayout()
        self.ratio_lbl = QLabel("Perubahan: 0.0%")
        self.ratio_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.debounce_lbl = QLabel("")
        self.debounce_lbl.setStyleSheet("color: #58a6ff; font-size: 11px; font-weight: bold;")

        bottom_row.addWidget(self.ratio_lbl)
        bottom_row.addStretch()
        bottom_row.addWidget(self.debounce_lbl)

        layout.addLayout(top_row)
        layout.addLayout(bottom_row)

        self.update_card("SETUP", "Menunggu dibuat", 0.0, "", 0.08)

    def _on_name_edited(self) -> None:
        new_name = self.name_edit.text().strip()
        if not new_name:
            new_name = f"ROI #{self.idx + 1}"
            self.name_edit.setText(new_name)
        self.config.set_roi_name(self.idx, new_name)
        self.config.save(CONFIG_FILE)
        self.name_changed.emit(self.idx, new_name)

    def update_card(self, state: str, label: str, ratio: float, debounce: str, ambang_rasio: float) -> None:
        styles = {
            "DONE": ("#122d1f", "#2ea043", "#3fb950"),
            "ACTIVE": ("#2b230f", "#d29922", "#e3b341"),
            "HAND_IN": ("#311b10", "#db6d28", "#f0883e"),
            "ERROR": ("#361417", "#f85149", "#ff7b72"),
            "READY": ("#13233a", "#388bfd", "#58a6ff"),
            "SETUP": ("#161b22", "#484f58", "#8b949e"),
            "LOCKED": ("#161b22", "#30363d", "#6e7681"),
        }
        bg, border, badge_color = styles.get(state, styles["LOCKED"])
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: 1.5px solid {border}; border-radius: 6px; }}"
            f"QLabel {{ border: none; background: transparent; }}"
        )
        self.badge_lbl.setText(label)
        self.badge_lbl.setStyleSheet(
            f"background-color: {border}; color: #ffffff; font-weight: bold; "
            f"font-size: 11px; border-radius: 4px; padding: 2px 6px;"
        )
        self.ratio_lbl.setText(f"Rasio Diff: {ratio * 100:.1f}% (Target ≥ {ambang_rasio * 100:.1f}%)")
        self.debounce_lbl.setText(debounce)


class SettingsDialog(QDialog):
    """Modal Dialog Pengaturan Sistem, Engine AI, Deteksi ROI, CLAHE, Cropping, dan Kalibrasi Kamera."""

    def __init__(self, config: AppConfig, worker: CameraWorker, parent=None):
        super().__init__(parent)
        self.config = config
        self.worker = worker
        self.setWindowTitle("⚙ Pengaturan Sistem dan Kalibrasi Kamera")

        # Ukuran modal: 80% dari tinggi layar yang tersedia agar tidak memenuhi seluruh layar
        screen = self.screen() or (parent.screen() if parent else None) or QApplication.primaryScreen()
        if screen:
            avail_geo = screen.availableGeometry()
            target_h = int(avail_geo.height() * 0.80)
            target_w = min(600, int(avail_geo.width() * 0.90))
            self.resize(target_w, target_h)
            self.setMaximumHeight(target_h)
        else:
            self.resize(580, 560)

        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_engine_param_tab(), "Engine dan Deteksi ROI")
        self.tabs.addTab(self._create_glove_crop_tab(), "Sarung Tangan dan Crop")
        self.tabs.addTab(self._create_camera_tab(), "Kalibrasi Kamera dan Profil")
        layout.addWidget(self.tabs, stretch=1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        btn_save_cfg = QPushButton("💾 Simpan ke config.json")
        btn_save_cfg.setStyleSheet("background-color: #1f5f35; border: 1px solid #3fb950; font-weight: bold; padding: 7px 16px;")
        btn_save_cfg.clicked.connect(self.save_configuration)

        btn_load_cfg = QPushButton("📂 Muat dari config.json")
        btn_load_cfg.setStyleSheet("background-color: #21262d; border: 1px solid #30363d; padding: 7px 16px;")
        btn_load_cfg.clicked.connect(self.load_configuration)

        bottom_row.addWidget(btn_save_cfg)
        bottom_row.addWidget(btn_load_cfg)
        bottom_row.addStretch()

        btn_close = QPushButton("Tutup Pengaturan")
        btn_close.setStyleSheet("background-color: #21262d; border: 1px solid #30363d; padding: 7px 22px; font-weight: bold;")
        btn_close.clicked.connect(self.accept)
        bottom_row.addWidget(btn_close)
        layout.addLayout(bottom_row)

    def _create_engine_param_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # 1. Engine Deteksi Tangan (Dual-Backend)
        engine_group = QGroupBox("Engine Deteksi Tangan (Dual-Backend)")
        engine_layout = QGridLayout(engine_group)
        engine_layout.setVerticalSpacing(8)

        engine_layout.addWidget(QLabel("Backend Deteksi:"), 0, 0)
        self.combo_backend = QComboBox()
        self.combo_backend.addItem("RTMlib (RTMPose Hand - Rekomendasi)", "rtmlib")
        self.combo_backend.addItem("Google MediaPipe Hands", "mediapipe")
        idx_b = 0 if self.config.hand_backend == "rtmlib" else 1
        self.combo_backend.setCurrentIndex(idx_b)
        self.combo_backend.currentIndexChanged.connect(self.on_backend_changed)
        engine_layout.addWidget(self.combo_backend, 0, 1)

        engine_layout.addWidget(QLabel("Threshold Skor Sendi:"), 1, 0)
        self.spin_rtm_score = QDoubleSpinBox()
        self.spin_rtm_score.setRange(0.10, 0.95)
        self.spin_rtm_score.setSingleStep(0.05)
        self.spin_rtm_score.setValue(self.config.rtm_score_thr)
        self.spin_rtm_score.valueChanged.connect(
            lambda v: setattr(self.config, "rtm_score_thr", float(v))
        )
        engine_layout.addWidget(self.spin_rtm_score, 1, 1)

        engine_layout.addWidget(QLabel("Mode Area Tangan:"), 2, 0)
        self.combo_tracking_mode = QComboBox()
        self.combo_tracking_mode.addItem("Seluruh Tangan (21 Sendi dan Kontur - Rekomendasi)", "full_hand")
        self.combo_tracking_mode.addItem("Pusat Punggung Tangan (Genggam Bebas)", "palm_center")
        self.combo_tracking_mode.addItem("5 Ujung Jari (Kaku / Wajib Terbuka)", "fingertips")
        mode_idx_map = {"full_hand": 0, "palm_center": 1, "fingertips": 2}
        self.combo_tracking_mode.setCurrentIndex(mode_idx_map.get(self.config.hand_tracking_mode, 0))
        self.combo_tracking_mode.currentIndexChanged.connect(self.on_tracking_mode_changed)
        engine_layout.addWidget(self.combo_tracking_mode, 2, 1)

        layout.addWidget(engine_group)

        # 2. Parameter Deteksi ROI
        param_group = QGroupBox("Parameter Urutan dan Sensitivitas ROI")
        grid = QGridLayout(param_group)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("Jumlah Langkah (ROI):"), 0, 0)
        self.spin_jumlah_roi = QSpinBox()
        self.spin_jumlah_roi.setRange(1, 6)
        self.spin_jumlah_roi.setValue(self.config.jumlah_roi)
        self.spin_jumlah_roi.valueChanged.connect(self.on_jumlah_roi_changed)
        grid.addWidget(self.spin_jumlah_roi, 0, 1)

        grid.addWidget(QLabel("Ambang Intensitas Piksel:"), 1, 0)
        self.spin_ambang_piksel = QSpinBox()
        self.spin_ambang_piksel.setRange(5, 200)
        self.spin_ambang_piksel.setValue(self.config.ambang_piksel)
        self.spin_ambang_piksel.setSuffix(" / 255")
        self.spin_ambang_piksel.valueChanged.connect(
            lambda v: setattr(self.config, "ambang_piksel", v)
        )
        grid.addWidget(self.spin_ambang_piksel, 1, 1)

        grid.addWidget(QLabel("Ambang Rasio Area Berubah:"), 2, 0)
        self.spin_ambang_rasio = QDoubleSpinBox()
        self.spin_ambang_rasio.setRange(0.5, 95.0)
        self.spin_ambang_rasio.setSingleStep(0.5)
        self.spin_ambang_rasio.setDecimals(1)
        self.spin_ambang_rasio.setSuffix(" %")
        self.spin_ambang_rasio.setValue(self.config.ambang_rasio * 100.0)
        self.spin_ambang_rasio.valueChanged.connect(
            lambda v: setattr(self.config, "ambang_rasio", round(v / 100.0, 4))
        )
        grid.addWidget(self.spin_ambang_rasio, 2, 1)

        grid.addWidget(QLabel("Debounce Masuk (Tangan Masuk):"), 3, 0)
        self.spin_debounce_masuk = QSpinBox()
        self.spin_debounce_masuk.setRange(1, 60)
        self.spin_debounce_masuk.setSuffix(" frame")
        self.spin_debounce_masuk.setValue(self.config.debounce_masuk)
        self.spin_debounce_masuk.valueChanged.connect(
            lambda v: setattr(self.config, "debounce_masuk", v)
        )
        grid.addWidget(self.spin_debounce_masuk, 3, 1)

        grid.addWidget(QLabel("Debounce Keluar (Tangan Keluar):"), 4, 0)
        self.spin_debounce_keluar = QSpinBox()
        self.spin_debounce_keluar.setRange(1, 90)
        self.spin_debounce_keluar.setSuffix(" frame")
        self.spin_debounce_keluar.setValue(self.config.debounce_keluar)
        self.spin_debounce_keluar.valueChanged.connect(
            lambda v: setattr(self.config, "debounce_keluar", v)
        )
        grid.addWidget(self.spin_debounce_keluar, 4, 1)

        self.chk_debug_diff = QCheckBox("Tampilkan Highlight Piksel Berubah (Mode Debug Diff)")
        self.chk_debug_diff.setChecked(self.config.debug_diff)
        self.chk_debug_diff.toggled.connect(lambda c: setattr(self.config, "debug_diff", c))
        grid.addWidget(self.chk_debug_diff, 5, 0, 1, 2)

        layout.addWidget(param_group)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(tab)
        return scroll

    def _create_glove_crop_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # 1. Optimasi Sarung Tangan
        glove_group = QGroupBox("Optimasi Sarung Tangan (CLAHE dan Pseudo-Skin)")
        glove_layout = QGridLayout(glove_group)
        glove_layout.setVerticalSpacing(8)

        self.chk_use_clahe = QCheckBox("Aktifkan Filter Sarung Tangan (CLAHE + Pseudo-Skin)")
        self.chk_use_clahe.setChecked(self.config.use_clahe)
        self.chk_use_clahe.toggled.connect(self.on_use_clahe_toggled)
        glove_layout.addWidget(self.chk_use_clahe, 0, 0, 1, 2)

        self.chk_clahe_invert = QCheckBox("Invert Warna (Khusus Sarung Tangan Hitam/Gelap)")
        self.chk_clahe_invert.setChecked(self.config.clahe_invert_dark)
        self.chk_clahe_invert.toggled.connect(lambda c: setattr(self.config, "clahe_invert_dark", c))
        glove_layout.addWidget(self.chk_clahe_invert, 1, 0, 1, 2)

        glove_layout.addWidget(QLabel("CLAHE Clip Limit:"), 2, 0)
        self.spin_clahe_clip = QDoubleSpinBox()
        self.spin_clahe_clip.setRange(0.5, 15.0)
        self.spin_clahe_clip.setSingleStep(0.5)
        self.spin_clahe_clip.setValue(self.config.clahe_clip_limit)
        self.spin_clahe_clip.valueChanged.connect(
            lambda v: setattr(self.config, "clahe_clip_limit", float(v))
        )
        glove_layout.addWidget(self.spin_clahe_clip, 2, 1)

        self.chk_clahe_preview = QCheckBox("Tampilkan Citra Pseudo-Skin di Layar (Debug Preview)")
        self.chk_clahe_preview.setChecked(self.config.show_clahe_preview)
        self.chk_clahe_preview.toggled.connect(lambda c: setattr(self.config, "show_clahe_preview", c))
        glove_layout.addWidget(self.chk_clahe_preview, 3, 0, 1, 2)
        layout.addWidget(glove_group)

        # 2. Optimasi ROI Cropping
        crop_group = QGroupBox("Optimasi ROI Cropping (Fokus Area Kerja)")
        crop_layout = QGridLayout(crop_group)
        crop_layout.setVerticalSpacing(8)

        self.chk_use_roi_crop = QCheckBox("Aktifkan ROI Cropping (Pangkas Area Deteksi)")
        self.chk_use_roi_crop.setChecked(self.config.use_roi_crop)
        self.chk_use_roi_crop.toggled.connect(self.on_use_roi_crop_toggled)
        crop_layout.addWidget(self.chk_use_roi_crop, 0, 0, 1, 2)

        crop_layout.addWidget(QLabel("Mode Pemotongan:"), 1, 0)
        self.combo_crop_mode = QComboBox()
        self.combo_crop_mode.addItem("Gabungan Semua ROI (Cepat dan Sinkron)", "union")
        self.combo_crop_mode.addItem("Individual Tiap Kotak ROI", "individual")
        idx_crop = 0 if self.config.roi_crop_mode == "union" else 1
        self.combo_crop_mode.setCurrentIndex(idx_crop)
        self.combo_crop_mode.currentIndexChanged.connect(
            lambda _: setattr(self.config, "roi_crop_mode", self.combo_crop_mode.currentData())
        )
        crop_layout.addWidget(self.combo_crop_mode, 1, 1)

        crop_layout.addWidget(QLabel("Padding Margin:"), 2, 0)
        self.spin_crop_padding = QSpinBox()
        self.spin_crop_padding.setRange(0, 200)
        self.spin_crop_padding.setSingleStep(5)
        self.spin_crop_padding.setSuffix(" px")
        self.spin_crop_padding.setValue(self.config.roi_crop_padding)
        self.spin_crop_padding.valueChanged.connect(
            lambda v: setattr(self.config, "roi_crop_padding", int(v))
        )
        crop_layout.addWidget(self.spin_crop_padding, 2, 1)

        self.chk_show_crop_box = QCheckBox("Tampilkan Garis Pembatas Crop di Layar")
        self.chk_show_crop_box.setChecked(self.config.show_crop_box)
        self.chk_show_crop_box.toggled.connect(
            lambda c: setattr(self.config, "show_crop_box", c)
        )
        crop_layout.addWidget(self.chk_show_crop_box, 3, 0, 1, 2)
        layout.addWidget(crop_group)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(tab)
        return scroll

    def _create_camera_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 1. Pemilihan Kamera
        cam_group = QGroupBox("Perangkat Webcam")
        cam_layout = QHBoxLayout(cam_group)
        cam_layout.addWidget(QLabel("Index Kamera:"))
        self.combo_cam = QComboBox()
        for idx in range(5):
            self.combo_cam.addItem(f"Kamera {idx}", idx)
        self.combo_cam.setCurrentIndex(min(max(self.config.camera_index, 0), 4))
        btn_switch_cam = QPushButton("Hubungkan")
        btn_switch_cam.clicked.connect(
            lambda: self.worker.request_camera_switch(self.combo_cam.currentData())
        )
        cam_layout.addWidget(self.combo_cam, 1)
        cam_layout.addWidget(btn_switch_cam)
        layout.addWidget(cam_group)

        # 2. Mode Auto vs Manual
        auto_group = QGroupBox("Kunci Fitur Otomatis Kamera")
        auto_layout = QHBoxLayout(auto_group)
        self.chk_af = QCheckBox("Auto Focus")
        self.chk_af.setChecked(self.config.auto_focus)
        self.chk_af.toggled.connect(self.on_auto_focus_toggled)

        self.chk_ae = QCheckBox("Auto Exposure")
        self.chk_ae.setChecked(self.config.auto_exposure)
        self.chk_ae.toggled.connect(self.on_auto_exposure_toggled)

        self.chk_awb = QCheckBox("Auto WB")
        self.chk_awb.setChecked(self.config.auto_wb)
        self.chk_awb.toggled.connect(self.on_auto_wb_toggled)

        auto_layout.addWidget(self.chk_af)
        auto_layout.addWidget(self.chk_ae)
        auto_layout.addWidget(self.chk_awb)
        layout.addWidget(auto_group)

        # 3. Slider Kalibrasi Manual Real-Time
        slider_group = QGroupBox("Kalibrasi Manual Real-Time")
        s_grid = QGridLayout(slider_group)
        s_grid.setVerticalSpacing(8)

        s_grid.addWidget(QLabel("Focus (0 - 255):"), 0, 0)
        self.slider_focus, self.spin_focus = self._make_slider_spin(
            0, 255, self.config.focus, lambda v: self.worker.update_camera_hardware("focus", v)
        )
        self.slider_focus.setEnabled(not self.config.auto_focus)
        self.spin_focus.setEnabled(not self.config.auto_focus)
        s_grid.addWidget(self.slider_focus, 0, 1)
        s_grid.addWidget(self.spin_focus, 0, 2)

        s_grid.addWidget(QLabel("Exposure (-13 s/d 0):"), 1, 0)
        self.slider_exp, self.spin_exp = self._make_slider_spin(
            -13, 0, self.config.exposure, lambda v: self.worker.update_camera_hardware("exposure", v)
        )
        self.slider_exp.setEnabled(not self.config.auto_exposure)
        self.spin_exp.setEnabled(not self.config.auto_exposure)
        s_grid.addWidget(self.slider_exp, 1, 1)
        s_grid.addWidget(self.spin_exp, 1, 2)

        s_grid.addWidget(QLabel("WB Temp (2800-6500K):"), 2, 0)
        self.slider_wb, self.spin_wb = self._make_slider_spin(
            2800, 6500, self.config.wb_temp, lambda v: self.worker.update_camera_hardware("wb_temp", v), step=50
        )
        self.slider_wb.setEnabled(not self.config.auto_wb)
        self.spin_wb.setEnabled(not self.config.auto_wb)
        s_grid.addWidget(self.slider_wb, 2, 1)
        s_grid.addWidget(self.spin_wb, 2, 2)

        s_grid.addWidget(QLabel("Gain / ISO (0 - 255):"), 3, 0)
        self.slider_gain, self.spin_gain = self._make_slider_spin(
            0, 255, self.config.gain, lambda v: self.worker.update_camera_hardware("gain", v)
        )
        s_grid.addWidget(self.slider_gain, 3, 1)
        s_grid.addWidget(self.spin_gain, 3, 2)
        layout.addWidget(slider_group)

        # 4. Manajemen Profil & Ekspor Kode
        prof_group = QGroupBox("Ekspor Kode OpenCV dan Info Konfigurasi")
        prof_layout = QVBoxLayout(prof_group)

        lbl_footer_tip = QLabel("💡 Tombol 'Simpan ke config.json' dan 'Muat dari config.json' kini selalu dapat diakses langsung pada footer bawah jendela pengaturan.")
        lbl_footer_tip.setStyleSheet("color: #8b949e; font-size: 11px;")
        lbl_footer_tip.setWordWrap(True)

        btn_copy_code = QPushButton("Cetak dan Salin Kode cap.set(...) ke Clipboard")
        btn_copy_code.clicked.connect(self.export_cap_set_snippet)

        prof_layout.addWidget(lbl_footer_tip)
        prof_layout.addWidget(btn_copy_code)
        layout.addWidget(prof_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(tab)
        return scroll

    @staticmethod
    def _make_slider_spin(min_val: int, max_val: int, init_val: int, callback, step: int = 1):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setSingleStep(step)
        slider.setValue(init_val)

        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(init_val)
        spin.setFixedWidth(75)

        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        spin.valueChanged.connect(callback)
        return slider, spin

    def on_backend_changed(self) -> None:
        selected = self.combo_backend.currentData()
        self.config.hand_backend = selected

    def on_tracking_mode_changed(self) -> None:
        selected = self.combo_tracking_mode.currentData()
        self.config.hand_tracking_mode = selected

    def on_jumlah_roi_changed(self, val: int) -> None:
        self.worker.set_jumlah_roi(val)

    def on_use_clahe_toggled(self, checked: bool) -> None:
        self.config.use_clahe = checked
        self.chk_clahe_invert.setEnabled(checked)
        self.spin_clahe_clip.setEnabled(checked)
        self.chk_clahe_preview.setEnabled(checked)

    def on_use_roi_crop_toggled(self, checked: bool) -> None:
        self.config.use_roi_crop = checked
        self.combo_crop_mode.setEnabled(checked)
        self.spin_crop_padding.setEnabled(checked)
        self.chk_show_crop_box.setEnabled(checked)

    def on_auto_focus_toggled(self, checked: bool) -> None:
        self.slider_focus.setEnabled(not checked)
        self.spin_focus.setEnabled(not checked)
        self.worker.update_camera_hardware("auto_focus", checked)

    def on_auto_exposure_toggled(self, checked: bool) -> None:
        self.slider_exp.setEnabled(not checked)
        self.spin_exp.setEnabled(not checked)
        self.worker.update_camera_hardware("auto_exposure", checked)

    def on_auto_wb_toggled(self, checked: bool) -> None:
        self.slider_wb.setEnabled(not checked)
        self.spin_wb.setEnabled(not checked)
        self.worker.update_camera_hardware("auto_wb", checked)

    def save_configuration(self) -> None:
        self.config.saved_rois = [r.tolist() for r in self.worker.rois]
        self.config.save(CONFIG_FILE)
        QMessageBox.information(
            self,
            "Konfigurasi Tersimpan",
            f"Konfigurasi dan {len(self.config.saved_rois)} ROI berhasil disimpan ke {CONFIG_FILE}.",
        )

    def load_configuration(self) -> None:
        loaded = AppConfig.load(CONFIG_FILE)
        self.config.__dict__.update(loaded.__dict__)
        self.worker.config = self.config

        self.spin_jumlah_roi.blockSignals(True)
        self.spin_jumlah_roi.setValue(self.config.jumlah_roi)
        self.spin_jumlah_roi.blockSignals(False)

        self.spin_ambang_piksel.setValue(self.config.ambang_piksel)
        self.spin_ambang_rasio.setValue(self.config.ambang_rasio * 100.0)
        self.spin_debounce_masuk.setValue(self.config.debounce_masuk)
        self.spin_debounce_keluar.setValue(self.config.debounce_keluar)
        self.chk_debug_diff.setChecked(self.config.debug_diff)
        self.chk_use_clahe.setChecked(self.config.use_clahe)
        self.chk_clahe_invert.setChecked(self.config.clahe_invert_dark)
        self.spin_clahe_clip.setValue(self.config.clahe_clip_limit)
        self.chk_clahe_preview.setChecked(self.config.show_clahe_preview)
        self.on_use_clahe_toggled(self.config.use_clahe)

        self.chk_use_roi_crop.setChecked(self.config.use_roi_crop)
        idx_crop = 0 if self.config.roi_crop_mode == "union" else 1
        self.combo_crop_mode.setCurrentIndex(idx_crop)
        self.spin_crop_padding.setValue(self.config.roi_crop_padding)
        self.chk_show_crop_box.setChecked(self.config.show_crop_box)
        self.on_use_roi_crop_toggled(self.config.use_roi_crop)

        idx_b = 0 if self.config.hand_backend == "rtmlib" else 1
        self.combo_backend.setCurrentIndex(idx_b)
        self.spin_rtm_score.setValue(self.config.rtm_score_thr)

        mode_map = {"full_hand": 0, "palm_center": 1, "fingertips": 2}
        idx_t = mode_map.get(getattr(self.config, "hand_tracking_mode", "full_hand"), 0)
        self.combo_tracking_mode.setCurrentIndex(idx_t)

        self.combo_cam.setCurrentIndex(min(max(self.config.camera_index, 0), 4))
        self.chk_af.setChecked(self.config.auto_focus)
        self.chk_ae.setChecked(self.config.auto_exposure)
        self.chk_awb.setChecked(self.config.auto_wb)
        self.spin_focus.setValue(self.config.focus)
        self.spin_exp.setValue(self.config.exposure)
        self.spin_wb.setValue(self.config.wb_temp)
        self.spin_gain.setValue(self.config.gain)

        if self.config.saved_rois:
            self.worker.load_saved_rois(self.config.saved_rois)
        else:
            self.worker.set_jumlah_roi(self.config.jumlah_roi)

        if self.parent() and hasattr(self.parent(), "_rebuild_roi_cards"):
            self.parent()._rebuild_roi_cards(self.config.jumlah_roi)

        QMessageBox.information(
            self,
            "Konfigurasi Dimuat",
            f"Konfigurasi berhasil dimuat dari {CONFIG_FILE}.",
        )

    def export_cap_set_snippet(self) -> None:
        snippet = (
            "=== SIMPAN NILAI INI KE KODE PRODUCTION ===\n"
            f"cap.set(cv2.CAP_PROP_AUTOFOCUS, {1 if self.config.auto_focus else 0})\n"
            f"cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, {0.75 if self.config.auto_exposure else 0.25})\n"
            f"cap.set(cv2.CAP_PROP_AUTO_WB, {1 if self.config.auto_wb else 0})\n"
            f"cap.set(cv2.CAP_PROP_FOCUS, {self.config.focus})\n"
            f"cap.set(cv2.CAP_PROP_EXPOSURE, {self.config.exposure})\n"
            f"cap.set(cv2.CAP_PROP_WB_TEMPERATURE, {self.config.wb_temp})\n"
            f"cap.set(cv2.CAP_PROP_GAIN, {self.config.gain})"
        )
        print("\n" + snippet)
        QApplication.clipboard().setText(snippet)
        QMessageBox.information(
            self,
            "Kode Kalibrasi Disalin",
            f"Konfigurasi kamera telah dicetak ke terminal dan disalin ke Clipboard:\n\n{snippet}",
        )


class MainWindow(QMainWindow):
    """Jendela Utama Operasional Produksi Balm Detection & Tracking Urutan Tangan."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Palm Detection")
        self.resize(1380, 840)

        self.config = AppConfig.load()
        self.roi_cards: List[ROICardWidget] = []

        # Inisialisasi Worker Thread Kamera sebelum _build_ui()
        # agar signal di ROICardWidget (_rebuild_roi_cards) dapat terhubung ke worker
        self.worker = CameraWorker(self.config, self)

        self._apply_dark_theme()
        self._build_ui()
        self._setup_shortcuts()

        self.worker.frame_ready.connect(self.canvas.update_frame)
        self.worker.telemetry_updated.connect(self.on_telemetry_updated)
        self.worker.camera_info_ready.connect(self.on_camera_info)
        self.worker.log_event.connect(self.add_log_entry)

        self.canvas.left_clicked.connect(self.worker.add_roi_point)
        self.canvas.right_clicked.connect(self.worker.undo_roi_point)
        self.canvas.mouse_moved.connect(self.worker.update_cursor)
        self.canvas.vertex_dragged.connect(self.worker.update_roi_vertex)
        self.canvas.vertex_drag_finished.connect(self.worker.finish_roi_vertex_drag)

        self.worker.start()
        self.add_log_entry("INFO", "Sistem siap. Mulai pemantauan urutan pengambilan part.")

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #e6edf3;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 5px;
                padding: 7px 12px;
                color: #e6edf3;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #8b949e;
            }
            QPushButton:pressed {
                background-color: #161b22;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                border-radius: 6px;
                background-color: #161b22;
            }
            QTabBar::tab {
                background-color: #0d1117;
                border: 1px solid #30363d;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                color: #8b949e;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #161b22;
                color: #58a6ff;
                border-bottom-color: #161b22;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e6edf3;
            }
            QSlider::groove:horizontal {
                border: 1px solid #30363d;
                height: 6px;
                background: #0d1117;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #2f81f7;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                border: 1px solid #1f6feb;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QListWidget {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 4px;
                font-family: 'Consolas', 'Segoe UI', monospace;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 3px 4px;
                border-bottom: 1px solid #161b22;
            }
            QProgressBar {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2ea043;
                border-radius: 4px;
            }
            QStatusBar {
                background-color: #161b22;
                border-top: 1px solid #30363d;
                color: #8b949e;
            }
        """)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(8)

        # ==========================================================
        # HEADER BAR: JUDUL + TOMBOL PENGATURAN & AKSI CEPAT
        # ==========================================================
        header_bar = QHBoxLayout()
        header_bar.setSpacing(8)

        title_lbl = QLabel("Palm Detection")
        title_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #58a6ff; letter-spacing: 1px;")

        sub_lbl = QLabel("Sistem Urutan Pengambilan Objek dan Verifikasi Langkah")
        sub_lbl.setStyleSheet("color: #8b949e; font-size: 11px; margin-left: 6px;")

        header_bar.addWidget(title_lbl)
        header_bar.addWidget(sub_lbl)
        header_bar.addStretch()

        btn_settings = QPushButton("⚙ Pengaturan (Ctrl+,)")
        btn_settings.setStyleSheet("background-color: #1f4f82; border-color: #58a6ff; font-weight: bold; padding: 6px 14px;")
        btn_settings.clicked.connect(self.open_settings_dialog)

        btn_clear_cycle = QPushButton("🔄 Reset Siklus (Ctrl+Shift+C)")
        btn_clear_cycle.clicked.connect(self.action_clear_status)

        btn_recapture_ref = QPushButton("📷 Ambil Ulang Ref (Ctrl+B)")
        btn_recapture_ref.clicked.connect(self.action_recapture_ref)

        btn_reset_roi = QPushButton("📐 Buat Ulang ROI (Ctrl+R)")
        btn_reset_roi.setStyleSheet("background-color: #491217; border-color: #f85149;")
        btn_reset_roi.clicked.connect(self.action_reset_roi)

        header_bar.addWidget(btn_settings)
        header_bar.addWidget(btn_clear_cycle)
        header_bar.addWidget(btn_recapture_ref)
        header_bar.addWidget(btn_reset_roi)

        root_layout.addLayout(header_bar)

        # ==========================================================
        # KONTEN UTAMA: SPLIT LAYOUT (KIRI: VIDEO, KANAN: MONITORING)
        # ==========================================================
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # PANEL KIRI: BANNER INSTRUKSI + VIDEO CANVAS INTERAKTIF
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        self.banner_lbl = QLabel("Memuat kamera dan menyiapkan sistem...")
        self.banner_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner_lbl.setFixedHeight(44)
        self.banner_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self._set_banner_style("setup")

        self.canvas = VideoCanvas()
        left_layout.addWidget(self.banner_lbl)
        left_layout.addWidget(self.canvas, stretch=1)

        # PANEL KANAN: MONITORING URUTAN & PRODUKSI
        right_panel = QWidget()
        right_panel.setFixedWidth(380)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 1. Box Ringkasan Produksi (Part Counter & Langkah Selesai)
        prod_group = QGroupBox("Ringkasan Produksi")
        prod_layout = QVBoxLayout(prod_group)
        prod_layout.setContentsMargins(10, 8, 10, 8)
        prod_layout.setSpacing(8)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)

        # Card Part Selesai
        card_part = QFrame()
        card_part.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;")
        cp_layout = QVBoxLayout(card_part)
        cp_layout.setContentsMargins(8, 6, 8, 6)
        cp_layout.setSpacing(2)
        self.part_count_lbl = QLabel("0")
        self.part_count_lbl.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self.part_count_lbl.setStyleSheet("color: #3fb950;")
        self.part_count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_p_desc = QLabel("Part Selesai")
        lbl_p_desc.setStyleSheet("color: #8b949e; font-size: 11px;")
        lbl_p_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cp_layout.addWidget(self.part_count_lbl)
        cp_layout.addWidget(lbl_p_desc)

        # Card Langkah Selesai
        card_step = QFrame()
        card_step.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;")
        cs_layout = QVBoxLayout(card_step)
        cs_layout.setContentsMargins(8, 6, 8, 6)
        cs_layout.setSpacing(2)
        self.step_count_lbl = QLabel("0 / 3")
        self.step_count_lbl.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.step_count_lbl.setStyleSheet("color: #58a6ff;")
        self.step_count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_s_desc = QLabel("Progres Langkah")
        lbl_s_desc.setStyleSheet("color: #8b949e; font-size: 11px;")
        lbl_s_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cs_layout.addWidget(self.step_count_lbl)
        cs_layout.addWidget(lbl_s_desc)

        cards_row.addWidget(card_part, 1)
        cards_row.addWidget(card_step, 1)
        prod_layout.addLayout(cards_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        prod_layout.addWidget(self.progress_bar)

        reset_part_row = QHBoxLayout()
        reset_part_row.addStretch()
        btn_reset_part = QPushButton("Reset Counter Part")
        btn_reset_part.setStyleSheet("font-size: 11px; padding: 3px 8px; color: #8b949e;")
        btn_reset_part.clicked.connect(self.action_reset_parts)
        reset_part_row.addWidget(btn_reset_part)
        prod_layout.addLayout(reset_part_row)

        right_layout.addWidget(prod_group)

        # 2. Box Status Urutan Langkah
        roi_group = QGroupBox("Status Urutan Langkah")
        roi_group_layout = QVBoxLayout(roi_group)
        roi_group_layout.setContentsMargins(8, 8, 8, 8)
        roi_scroll = QScrollArea()
        roi_scroll.setWidgetResizable(True)
        roi_scroll.setFrameShape(QFrame.Shape.NoFrame)
        roi_scroll.setMaximumHeight(220)

        self.roi_cards_container = QWidget()
        self.roi_cards_layout = QVBoxLayout(self.roi_cards_container)
        self.roi_cards_layout.setContentsMargins(2, 2, 2, 2)
        self.roi_cards_layout.setSpacing(6)
        self._rebuild_roi_cards(self.config.jumlah_roi)

        roi_scroll.setWidget(self.roi_cards_container)
        roi_group_layout.addWidget(roi_scroll)
        right_layout.addWidget(roi_group)

        # 3. Box Log Aktivitas Urutan
        log_group = QGroupBox("Log Aktivitas Urutan")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)

        self.log_list = QListWidget()
        log_layout.addWidget(self.log_list, stretch=1)

        log_btn_row = QHBoxLayout()
        lbl_auto = QLabel("Auto scroll aktif")
        lbl_auto.setStyleSheet("color: #484f58; font-size: 10px;")
        btn_clear_log = QPushButton("Bersihkan Log")
        btn_clear_log.setStyleSheet("font-size: 11px; padding: 3px 8px;")
        btn_clear_log.clicked.connect(self.log_list.clear)

        log_btn_row.addWidget(lbl_auto)
        log_btn_row.addStretch()
        log_btn_row.addWidget(btn_clear_log)
        log_layout.addLayout(log_btn_row)

        right_layout.addWidget(log_group, stretch=1)

        content_layout.addLayout(left_layout, stretch=1)
        content_layout.addWidget(right_panel)
        root_layout.addLayout(content_layout, stretch=1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.cam_status_lbl = QLabel("Kamera: Menunggu...")
        self.ref_status_lbl = QLabel("Referensi: Belum diambil")
        self.status_bar.addWidget(self.cam_status_lbl, 1)
        self.status_bar.addPermanentWidget(self.ref_status_lbl)

    def _rebuild_roi_cards(self, count: int) -> None:
        for card in self.roi_cards:
            self.roi_cards_layout.removeWidget(card)
            card.deleteLater()
        self.roi_cards.clear()

        for i in range(count):
            card = ROICardWidget(i, self.config)
            card.redo_requested.connect(self.worker.trigger_redo_roi)
            card.name_changed.connect(self.on_roi_name_changed)
            self.roi_cards.append(card)
            self.roi_cards_layout.addWidget(card)
        self.roi_cards_layout.addStretch()

    def on_roi_name_changed(self, idx: int, name: str) -> None:
        self.add_log_entry("INFO", f"Nama ROI #{idx + 1} diubah menjadi: [{name}].")

    def _setup_shortcuts(self) -> None:
        act_r = QAction(self)
        act_r.setShortcut(QKeySequence("Ctrl+R"))
        act_r.triggered.connect(self.action_reset_roi)
        self.addAction(act_r)

        act_b = QAction(self)
        act_b.setShortcut(QKeySequence("Ctrl+B"))
        act_b.triggered.connect(self.action_recapture_ref)
        self.addAction(act_b)

        act_c = QAction(self)
        act_c.setShortcut(QKeySequence("Ctrl+Shift+C"))
        act_c.triggered.connect(self.action_clear_status)
        self.addAction(act_c)

        act_set = QAction(self)
        act_set.setShortcut(QKeySequence("Ctrl+,"))
        act_set.triggered.connect(self.open_settings_dialog)
        self.addAction(act_set)

        act_quit = QAction(self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        self.addAction(act_quit)

    def _set_banner_style(self, state: str) -> None:
        styles = {
            "setup": ("#13233a", "#388bfd", "#79c0ff"),
            "active": ("#2b230f", "#d29922", "#ffd33d"),
            "error": ("#491217", "#f85149", "#ffa198"),
            "done": ("#122d1f", "#2ea043", "#56d364"),
        }
        bg, border, text_color = styles.get(state, styles["setup"])
        self.banner_lbl.setStyleSheet(
            f"background-color: {bg}; border: 2px solid {border}; "
            f"color: {text_color}; border-radius: 6px; padding: 4px 12px;"
        )

    # ---- Slots & Event Handlers ----
    def open_settings_dialog(self) -> None:
        dlg = SettingsDialog(self.config, self.worker, self)
        dlg.exec()

    @pyqtSlot(str, str)
    def add_log_entry(self, level: str, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}] {message}")

        colors = {
            "SUCCESS": "#3fb950",   # Hijau
            "PART": "#58a6ff",      # Biru terang
            "WARN": "#f85149",      # Merah
            "INFO": "#8b949e",      # Abu-abu
        }
        item.setForeground(QColor(colors.get(level, "#c9d1d9")))
        if level in ("PART", "SUCCESS"):
            item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    @pyqtSlot(dict)
    def on_telemetry_updated(self, data: dict) -> None:
        self.banner_lbl.setText(data["banner_text"])
        self._set_banner_style(data["banner_state"])

        # Update koordinat poligon ke canvas untuk interaksi drag & drop sudut
        if "rois_coords" in data:
            self.canvas.set_rois_coords(data["rois_coords"])

        # Update counter produksi & progres langkah
        parts = data.get("parts_completed", 0)
        steps = data.get("steps_completed", 0)
        total_steps = data.get("roi_total", self.config.jumlah_roi)

        self.part_count_lbl.setText(str(parts))
        self.step_count_lbl.setText(f"{steps} / {total_steps}")
        if total_steps > 0:
            pct = int((steps / total_steps) * 100)
            self.progress_bar.setValue(pct)

        rois_data = data["rois"]
        if len(rois_data) != len(self.roi_cards):
            self._rebuild_roi_cards(len(rois_data))

        for item, card in zip(rois_data, self.roi_cards):
            card.update_card(
                item["state"],
                item["label"],
                item["ratio"],
                item["debounce"],
                self.config.ambang_rasio,
            )

        engine_str = "RTMlib (RTMPose)" if self.config.hand_backend == "rtmlib" else "Google MediaPipe"
        if data["has_reference"]:
            self.ref_status_lbl.setText(f"Engine: {engine_str} | Referensi: Tersimpan | FPS: {data['fps']:.1f}")
        else:
            self.ref_status_lbl.setText(
                f"Engine: {engine_str} | Referensi: Menunggu ({data['roi_defined']}/{data['roi_total']} ROI) | FPS: {data['fps']:.1f}"
            )

    @pyqtSlot(str)
    def on_camera_info(self, info: str) -> None:
        self.cam_status_lbl.setText(info)

    def action_reset_roi(self) -> None:
        reply = QMessageBox.question(
            self,
            "Konfirmasi Buat Ulang ROI",
            "Apakah Anda yakin ingin menghapus semua ROI hasil kalibrasi dan membuat ulang dari awal?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.worker.trigger_reset_roi()

    def action_recapture_ref(self) -> None:
        self.worker.trigger_recapture_ref()

    def action_clear_status(self) -> None:
        self.worker.trigger_clear_status()

    def action_reset_parts(self) -> None:
        self.worker.trigger_reset_parts()

    def closeEvent(self, event) -> None:
        reply = QMessageBox.question(
            self,
            "Konfirmasi Keluar",
            "Apakah Anda yakin ingin menutup aplikasi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self.status_bar.showMessage("Menutup aplikasi dan mematikan kamera...", 2000)
        self.worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.worker.stop)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()