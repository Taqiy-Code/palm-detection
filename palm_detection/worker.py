"""CameraWorker: QThread orkestrator akuisisi kamera, deteksi AI, state machine, dan rendering."""

from __future__ import annotations

import atexit
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Pre-load MediaPipe sebelum PyQt5 untuk mencegah konflik DLL _framework_bindings di Windows
try:
    import mediapipe as _mp  # noqa: F401
except Exception:
    pass

try:
    from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QImage
except ImportError:
    class QThread(threading.Thread):
        """Fallback QThread menggunakan threading.Thread standar saat PyQt5 tidak terpasang."""

        def __init__(self, parent=None):
            super().__init__(daemon=True)
            self._is_running = False

        def isRunning(self) -> bool:
            return self.is_alive()

        def wait(self, timeout: Optional[float] = None) -> None:
            if timeout is not None:
                self.join(timeout / 1000.0)
            else:
                self.join()

        def start(self) -> None:
            self._is_running = True
            super().start()

    class _Signal:
        def __init__(self, *types):
            self._callbacks = []

        def connect(self, slot) -> None:
            if slot not in self._callbacks:
                self._callbacks.append(slot)

        def disconnect(self, slot) -> None:
            if slot in self._callbacks:
                self._callbacks.remove(slot)

        def emit(self, *args):
            for cb in list(self._callbacks):
                try:
                    cb(*args)
                except Exception:
                    pass

    def pyqtSignal(*types):
        return _Signal(*types)

    def pyqtSlot(*types):
        def decorator(fn):
            return fn
        return decorator

    class QImage:
        Format_RGB888 = 0

        def __init__(self, *args, **kwargs):
            pass

        def copy(self):
            return self

from palm_detection.config import AppConfig, CONFIG_FILE
from palm_detection.detector import HandDetectionPipeline
from palm_detection.renderer import OverlayRenderer, WARNA_ROI_BGR
from palm_detection.sequencer import StepSequencer
from palm_detection.states import BannerState, LogLevel, RoiState, StepStatus, TrackingMode
from palm_detection.vision import (
    clean_diff_mask,
    hull_roi_overlap_ratio,
    normalize_and_validate_roi,
    rasio_gerak_roi,
    rasio_perubahan,
    siapkan_gray,
)

# Registri global seluruh instance cv2.VideoCapture aktif agar dijamin mati saat aplikasi keluar
_ACTIVE_CAPTURES: List[cv2.VideoCapture] = []
_ACTIVE_CAPTURES_LOCK = threading.RLock()


def _register_capture(cap: Optional[cv2.VideoCapture]) -> None:
    if cap is None:
        return
    with _ACTIVE_CAPTURES_LOCK:
        if cap not in _ACTIVE_CAPTURES:
            _ACTIVE_CAPTURES.append(cap)


def _release_capture_safe(cap: Optional[cv2.VideoCapture]) -> None:
    if cap is None:
        return
    with _ACTIVE_CAPTURES_LOCK:
        if cap in _ACTIVE_CAPTURES:
            _ACTIVE_CAPTURES.remove(cap)
    try:
        cap.release()
    except Exception:
        pass


def release_all_cameras() -> None:
    """Mematikan dan melepas seluruh kamera/VideoCapture yang masih tercatat aktif."""
    with _ACTIVE_CAPTURES_LOCK:
        captures = list(_ACTIVE_CAPTURES)
        _ACTIVE_CAPTURES.clear()
    for cap in captures:
        try:
            cap.release()
        except Exception:
            pass


atexit.register(release_all_cameras)


class CameraWorker(QThread):
    """Thread pekerja yang mengorkestrasi akuisisi video, deteksi tangan, dan evaluasi langkah."""

    frame_ready = pyqtSignal(QImage, int, int)
    telemetry_updated = pyqtSignal(dict)
    camera_info_ready = pyqtSignal(str)
    log_event = pyqtSignal(str, str)  # (level, message)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.running = False
        self._is_stopped = False
        self._stop_lock = threading.Lock()
        self._cap_lock = threading.RLock()
        self._opened_captures: List[cv2.VideoCapture] = []

        # State kamera & video (HANYA dibuat, diakses, dan dirilis secara thread-safe)
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_w = config.frame_width
        self.frame_h = config.frame_height

        # State Input Video & Kontrol Frame oleh Pengguna
        self.is_video_mode: bool = False
        self.video_path: Optional[str] = None
        self.video_paused: bool = True
        self.video_fps: float = 30.0
        self.video_total_frames: int = 0
        self.video_current_frame: int = 0
        self.video_start_frame: int = 0
        self.video_end_frame: int = 0
        self._raw_frame: Optional[np.ndarray] = None
        self._needs_process_current: bool = False
        self._step_once: bool = False
        self._manual_ref_locked: bool = False
        self._cached_detect_key: Optional[tuple] = None
        self._cached_detect_res: Optional[Tuple[List[dict], List[Tuple[int, int, int, int]], np.ndarray]] = None

        # State Perekaman / Output Video Teranotasi
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_recording: bool = False
        self.recording_path: Optional[str] = None
        self.recorded_frames_count: int = 0

        # State ROI
        self.rois: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.titik_klik: List[Tuple[int, int]] = []
        self.kursor: Tuple[int, int] = (0, 0)
        self._redo_roi_idx: Optional[int] = None

        # State Background & Differencing
        self.referensi: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._no_hand_frames: int = 0
        self._roi_clear_frames: int = 0
        self._stable_ref_buffer: List[np.ndarray] = []
        self._post_hand_frames: List[List[np.ndarray]] = []
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Modul Terpisah
        self.sequencer = StepSequencer(
            num_steps=config.jumlah_roi,
            debounce_masuk=config.debounce_masuk,
            debounce_keluar=config.debounce_keluar,
            ambang_rasio=config.ambang_rasio,
            use_pixel_diff=config.use_pixel_diff,
        )
        self.pipeline = HandDetectionPipeline()
        self.renderer = OverlayRenderer()

        # Antrian perintah thread-safe dari GUI
        self._cmd_queue: queue.Queue = queue.Queue()

        # State Web Stream & Headless Telemetry
        self._web_lock = threading.Lock()
        self._latest_bgr_frame: Optional[np.ndarray] = None
        self._latest_telemetry: dict = {}
        self._logs_lock = threading.Lock()
        self._recent_logs: List[dict] = []
        self.log_event.connect(self._record_internal_log)

        self._init_post_hand_frames()

    def _record_internal_log(self, level: str, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self._logs_lock:
            self._recent_logs.append({"time": ts, "level": level, "message": message})
            if len(self._recent_logs) > 200:
                self._recent_logs.pop(0)

    def get_latest_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """Mengambil frame teranotasi terbaru dalam format bytes JPEG untuk streaming web."""
        with self._web_lock:
            if self._latest_bgr_frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self._latest_bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return buf.tobytes() if ok else None

    def get_latest_telemetry(self) -> dict:
        """Mengambil data telemetri real-time terbaru secara thread-safe."""
        with self._web_lock:
            return dict(self._latest_telemetry)

    def get_recent_logs(self, limit: int = 50) -> List[dict]:
        """Mengambil riwayat log event terbaru."""
        with self._logs_lock:
            return list(self._recent_logs[-limit:])

    def _init_post_hand_frames(self) -> None:
        n = max(self.config.jumlah_roi, len(self.rois), 1)
        self._post_hand_frames = [[] for _ in range(n)]

    def _ensure_post_hand_frames(self) -> None:
        needed = max(self.config.jumlah_roi, len(self.rois), 1)
        curr = len(self._post_hand_frames)
        if curr < needed:
            self._post_hand_frames.extend([[] for _ in range(needed - curr)])

    # =========================================================================
    # Antarmuka Komunikasi GUI -> Worker (Seluruhnya lewat Command Queue)
    # =========================================================================
    def post_command(self, cmd: str, arg=None) -> None:
        """Mengirimkan perintah ke antrian worker thread."""
        self._cmd_queue.put((cmd, arg))

    def request_camera_switch(self, new_index: int) -> None:
        self.post_command("switch_camera", int(new_index))

    def open_video_file(self, filepath: str) -> None:
        self.post_command("open_video", str(filepath))

    def set_video_paused(self, paused: bool) -> None:
        self.post_command("set_video_paused", bool(paused))

    def toggle_video_play(self) -> None:
        self.post_command("toggle_video_play", None)

    def seek_video_frame(self, frame_idx: int) -> None:
        self.post_command("seek_video", int(frame_idx))

    def step_video_frame(self, delta: int) -> None:
        self.post_command("step_video", int(delta))

    def set_video_frame_range(self, start_frame: int, end_frame: int) -> None:
        self.post_command("set_video_range", (int(start_frame), int(end_frame)))

    def lock_current_frame_as_ref(self, roi_mode: str = "keep") -> None:
        """Mengunci frame saat ini sebagai referensi bersih dengan opsi penanganan ROI.

        Args:
            roi_mode:
                - "keep"  : Gunakan ROI saat ini yang sudah ada.
                - "add"   : Tambahkan ROI baru di atas frame referensi ini.
                - "reset" : Hapus ROI lama dan buat ulang ROI dari awal pada frame ini.
        """
        self.post_command("lock_current_ref", str(roi_mode))

    def start_video_recording(
        self,
        output_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> None:
        self.post_command("start_recording", (str(output_path), start_frame, end_frame))

    def stop_video_recording(self) -> None:
        self.post_command("stop_recording", None)

    def update_camera_hardware(self, prop_name: str, value) -> None:
        setattr(self.config, prop_name, value)
        self.post_command("update_cam_prop", (prop_name, value))

    def add_roi_point(self, x: int, y: int) -> None:
        self.post_command("add_point", (int(x), int(y)))

    def undo_roi_point(self) -> None:
        self.post_command("undo_point", None)

    def update_cursor(self, x: int, y: int) -> None:
        self.post_command("update_cursor", (int(x), int(y)))

    def trigger_reset_roi(self) -> None:
        self.post_command("reset_roi", None)

    def trigger_recapture_ref(self) -> None:
        self.post_command("recapture_ref", None)

    def trigger_clear_status(self) -> None:
        self.sequencer.clear_statuses()
        self.post_command("clear_status", None)

    def trigger_reset_parts(self) -> None:
        self.sequencer.reset_parts()
        self.post_command("reset_parts", None)

    def update_roi_vertex(self, roi_idx: int, v_idx: int, nx: int, ny: int) -> None:
        self.post_command("update_vertex", (int(roi_idx), int(v_idx), int(nx), int(ny)))

    def finish_roi_vertex_drag(self) -> None:
        self.post_command("rebuild_masks", None)

    def trigger_redo_roi(self, idx: int) -> None:
        self.post_command("redo_roi", int(idx))

    def load_saved_rois(self, saved_rois: List[List[List[int]]]) -> None:
        self.post_command("load_rois", saved_rois)

    def set_jumlah_roi(self, count: int) -> None:
        self.config.jumlah_roi = int(count)
        self.post_command("set_jumlah_roi", int(count))

    # =========================================================================
    # Pengelolaan Kamera & Video (Hanya berjalan di worker thread)
    # =========================================================================
    def _stop_recording_internal(self, completed: bool = False) -> None:
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        if self.is_recording:
            saved_path = self.recording_path or ""
            frames_written = self.recorded_frames_count
            self.is_recording = False
            self.recording_path = None
            status_msg = "selesai diproses dan disimpan" if completed else "dihentikan dan disimpan"
            self.log_event.emit(
                LogLevel.SUCCESS.value,
                f"Video output ({frames_written} frame) {status_msg}: {saved_path}",
            )

    def _start_recording_internal(
        self,
        output_path: str,
        start_frame: Optional[int],
        end_frame: Optional[int],
    ) -> bool:
        self._stop_recording_internal(completed=False)
        if self.cap is None or not self.cap.isOpened():
            self.log_event.emit(LogLevel.WARN.value, "Gagal merekam: Sumber video/kamera belum aktif.")
            return False

        if self.is_video_mode:
            max_idx = max(0, self.video_total_frames - 1)
            sf = max(0, min(int(start_frame) if start_frame is not None else self.video_start_frame, max_idx))
            ef = max(sf, min(int(end_frame) if end_frame is not None else self.video_end_frame, max_idx))
            self.video_start_frame = sf
            self.video_end_frame = ef
            self._seek_to_frame(sf)
            self._needs_process_current = True
            self.sequencer.clear_statuses()
            self._init_post_hand_frames()
            self._roi_clear_frames = 0
            if not self._manual_ref_locked and self.referensi is None:
                self._no_hand_frames = 0
                self._stable_ref_buffer.clear()
            self.video_paused = False
            out_fps = self.video_fps if self.video_fps > 0 else 30.0
        else:
            out_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if out_fps <= 1.0 or out_fps > 120.0 or np.isnan(out_fps):
                out_fps = 30.0

        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (self.frame_w, self.frame_h))
        if not writer.isOpened():
            self.log_event.emit(LogLevel.WARN.value, f"Gagal membuat file video output: {output_path}")
            return False

        self.video_writer = writer
        self.is_recording = True
        self.recording_path = output_path
        self.recorded_frames_count = 0
        if self.is_video_mode:
            self.log_event.emit(
                LogLevel.INFO.value,
                f"Memulai perekaman output video (Frame #{self.video_start_frame} s/d #{self.video_end_frame}) -> {os.path.basename(output_path)}",
            )
        else:
            self.log_event.emit(
                LogLevel.INFO.value,
                f"Memulai perekaman kamera live -> {os.path.basename(output_path)}",
            )
        return True

    def _seek_to_frame(self, target_idx: int) -> bool:
        if not self.is_video_mode or self.cap is None or not self.cap.isOpened():
            return False
        max_idx = max(0, self.video_total_frames - 1)
        clamped = max(0, min(int(target_idx), max_idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
        ok, f = self.cap.read()
        if ok and f is not None:
            self._raw_frame = f.copy()
            self.video_current_frame = clamped
            self._prev_gray = siapkan_gray(f)
            self._cached_detect_key = None
            return True
        return False

    def _release_all_worker_captures(self) -> None:
        """Melepas seluruh instance VideoCapture yang pernah dibuka oleh worker ini."""
        with self._cap_lock:
            if self.cap is not None:
                _release_capture_safe(self.cap)
                self.cap = None
            while self._opened_captures:
                c = self._opened_captures.pop()
                _release_capture_safe(c)

    def _open_video(self, filepath: str) -> bool:
        if self._is_stopped:
            return False
        self._stop_recording_internal(completed=False)
        if not os.path.exists(filepath):
            self.log_event.emit(LogLevel.WARN.value, f"File video tidak ditemukan: {filepath}")
            return False

        c = cv2.VideoCapture(filepath)
        _register_capture(c)
        if not c.isOpened():
            _release_capture_safe(c)
            self.log_event.emit(LogLevel.WARN.value, f"Gagal membuka file video: {filepath}")
            return False

        ok, first_frame = c.read()
        if not ok or first_frame is None:
            _release_capture_safe(c)
            self.log_event.emit(LogLevel.WARN.value, f"Gagal membaca frame dari video: {filepath}")
            return False

        with self._cap_lock:
            self._release_all_worker_captures()
            self.cap = c
            self._opened_captures.append(c)

        self.is_video_mode = True
        self.video_path = filepath
        self.video_paused = True
        self._step_once = False
        self._needs_process_current = False

        total = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = float(c.get(cv2.CAP_PROP_FPS) or 30.0)
        if v_fps <= 1.0 or v_fps > 240.0 or np.isnan(v_fps):
            v_fps = 30.0

        self.video_fps = v_fps
        self.video_total_frames = max(1, total)
        self.video_current_frame = 0
        self.video_start_frame = 0
        self.video_end_frame = max(0, self.video_total_frames - 1)

        self._raw_frame = first_frame.copy()
        self._cached_detect_key = None
        self.frame_h, self.frame_w = first_frame.shape[:2]
        if self.rois:
            self._rebuild_masks()

        self._invalidate_reference()
        self.sequencer.clear_statuses()
        self._init_post_hand_frames()
        self._prev_gray = siapkan_gray(first_frame)

        fname = os.path.basename(filepath)
        self.camera_info_ready.emit(
            f"Video: {fname} ({self.frame_w}x{self.frame_h} @ {self.video_fps:.1f} FPS) | Total: {self.video_total_frames} frame"
        )
        self.log_event.emit(
            LogLevel.INFO.value,
            f"Video '{fname}' dimuat ({self.video_total_frames} frame). Silakan pilih frame untuk atur ROI / Referensi.",
        )
        return True

    def _open_camera(self) -> bool:
        if self._is_stopped:
            return False
        self._stop_recording_internal(completed=False)
        self.is_video_mode = False
        self.video_path = None
        self.video_paused = False
        self._raw_frame = None
        self._cached_detect_key = None

        self._release_all_worker_captures()

        def _try_open(idx: int):
            if self._is_stopped:
                return None, None
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            _register_capture(c)
            if not c.isOpened():
                _release_capture_safe(c)
                if self._is_stopped:
                    return None, None
                c = cv2.VideoCapture(idx)
                _register_capture(c)
            if c.isOpened():
                if self._is_stopped:
                    _release_capture_safe(c)
                    return None, None
                c.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    try:
                        c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 150)
                    except Exception:
                        pass
                success, f = c.read()
                if success and f is not None and not self._is_stopped:
                    return c, f
                _release_capture_safe(c)
            else:
                _release_capture_safe(c)
            return None, None

        cap_obj, frame = _try_open(self.config.camera_index)
        if cap_obj is None and not self._is_stopped:
            for alt_idx in [1, 0, 2]:
                if self._is_stopped:
                    break
                if alt_idx == self.config.camera_index:
                    continue
                cap_obj, frame = _try_open(alt_idx)
                if cap_obj is not None:
                    self.config.camera_index = alt_idx
                    break

        if self._is_stopped:
            if cap_obj is not None:
                _release_capture_safe(cap_obj)
            return False

        if cap_obj is None or not cap_obj.isOpened():
            if cap_obj is not None:
                _release_capture_safe(cap_obj)
            self.cap = None
            self.camera_info_ready.emit(f"Gagal membuka Kamera Index {self.config.camera_index}")
            return False

        with self._cap_lock:
            self.cap = cap_obj
            self._opened_captures.append(cap_obj)

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
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.config.auto_focus else 0)
        if self.config.auto_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if self.config.auto_wb else 0)

        if not self.config.auto_focus:
            self.cap.set(cv2.CAP_PROP_FOCUS, self.config.focus)
        if not self.config.auto_exposure:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
        if not self.config.auto_wb:
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.config.wb_temp)
        self.cap.set(cv2.CAP_PROP_GAIN, self.config.gain)

    def _rebuild_masks(self) -> None:
        self.masks.clear()
        for poly in self.rois:
            m = np.zeros((self.frame_h, self.frame_w), np.uint8)
            cv2.fillPoly(m, [poly], 255)
            self.masks.append(m)

    def _invalidate_reference(self, force: bool = True) -> None:
        if not force and self._manual_ref_locked and self.referensi is not None:
            return
        self.referensi = None
        self._manual_ref_locked = False
        self._no_hand_frames = 0
        self._roi_clear_frames = 0
        self._stable_ref_buffer.clear()

    def _do_reset_roi(self) -> None:
        self.rois.clear()
        self.masks.clear()
        self.titik_klik.clear()
        self._redo_roi_idx = None
        self._invalidate_reference(force=True)
        self.sequencer.reset(self.config.jumlah_roi)
        self._init_post_hand_frames()

    def _cleanup_resources(self) -> None:
        """Melepas seluruh kamera, perekam video, dan model AI secara idempoten."""
        with self._cap_lock:
            self._stop_recording_internal(completed=False)
            self._release_all_worker_captures()
            release_all_cameras()
            self.pipeline.close()

    # =========================================================================
    # Pemrosesan Perintah Masuk dari GUI Thread
    # =========================================================================
    def _process_pending_commands(self, gray_now: Optional[np.ndarray]) -> None:
        cam_props_to_apply: dict = {}
        pending_seek_target: Optional[int] = None

        while True:
            try:
                cmd, arg = self._cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "update_cursor":
                self.kursor = arg

            elif cmd == "switch_camera":
                self.config.camera_index = int(arg)
                self._open_camera()

            elif cmd == "open_video":
                self._open_video(str(arg))

            elif cmd == "set_video_paused":
                if self.is_video_mode:
                    self.video_paused = bool(arg)
                    if not self.video_paused and self.video_current_frame >= self.video_end_frame:
                        self._seek_to_frame(self.video_start_frame)
                        self._needs_process_current = True

            elif cmd == "toggle_video_play":
                if self.is_video_mode:
                    self.video_paused = not self.video_paused
                    if not self.video_paused and self.video_current_frame >= self.video_end_frame:
                        self._seek_to_frame(self.video_start_frame)
                        self._needs_process_current = True

            elif cmd == "seek_video":
                if self.is_video_mode:
                    pending_seek_target = int(arg)

            elif cmd == "step_video":
                if self.is_video_mode:
                    delta = int(arg)
                    self.video_paused = True
                    if delta == 1 and pending_seek_target is None and self.video_current_frame < self.video_total_frames - 1:
                        self._step_once = True
                    else:
                        base_frame = pending_seek_target if pending_seek_target is not None else self.video_current_frame
                        pending_seek_target = max(0, min(base_frame + delta, max(0, self.video_total_frames - 1)))

            elif cmd == "set_video_range":
                if self.is_video_mode:
                    sf, ef = arg
                    max_idx = max(0, self.video_total_frames - 1)
                    self.video_start_frame = max(0, min(int(sf), max_idx))
                    self.video_end_frame = max(self.video_start_frame, min(int(ef), max_idx))

            elif cmd == "lock_current_ref":
                roi_mode = str(arg or "keep").lower()
                src_gray = None
                if self._raw_frame is not None:
                    src_gray = siapkan_gray(self._raw_frame)
                elif gray_now is not None:
                    src_gray = gray_now.copy()

                if src_gray is not None:
                    if self.is_video_mode:
                        self.video_paused = True

                    if roi_mode == "reset":
                        self.rois.clear()
                        self.masks.clear()
                        self.titik_klik.clear()
                        self._redo_roi_idx = None
                    elif roi_mode == "add":
                        self.titik_klik.clear()
                        self._redo_roi_idx = None
                        if len(self.rois) >= self.config.jumlah_roi:
                            self.config.jumlah_roi = len(self.rois) + 1

                    self.referensi = src_gray
                    self._manual_ref_locked = True
                    needed = max(1, int(self.config.ref_clean_frames))
                    self._no_hand_frames = needed
                    self._stable_ref_buffer = [src_gray.copy() for _ in range(needed)]
                    self.sequencer.reset(self.config.jumlah_roi)
                    self._init_post_hand_frames()

                    frame_label = (
                        f"Frame #{self.video_current_frame}" if self.is_video_mode else "Frame saat ini"
                    )
                    if roi_mode == "add":
                        self.log_event.emit(
                            LogLevel.INFO.value,
                            f"{frame_label} dikunci sebagai referensi. Silakan klik 4 sudut pada video untuk menambahkan ROI #{len(self.rois) + 1}.",
                        )
                    elif roi_mode == "reset":
                        self.log_event.emit(
                            LogLevel.INFO.value,
                            f"{frame_label} dikunci sebagai referensi. Silakan klik 4 sudut pada video untuk membuat ROI baru.",
                        )
                    else:
                        self.log_event.emit(
                            LogLevel.SUCCESS.value,
                            f"{frame_label} dikunci sebagai referensi background bersih (menggunakan {len(self.rois)} ROI saat ini).",
                        )

            elif cmd == "start_recording":
                out_path, sf, ef = arg
                self._start_recording_internal(out_path, sf, ef)

            elif cmd == "stop_recording":
                self._stop_recording_internal(completed=False)
                if self.is_video_mode:
                    self.video_paused = True

            elif cmd == "update_cam_prop":
                prop_name, val = arg
                cam_props_to_apply[prop_name] = val

            elif cmd == "reset_roi":
                self._do_reset_roi()

            elif cmd == "set_jumlah_roi":
                self.config.jumlah_roi = int(arg)
                self.sequencer.reset(self.config.jumlah_roi)
                self._do_reset_roi()

            elif cmd == "load_rois":
                loaded = arg
                self._do_reset_roi()
                for pts in loaded[: self.config.jumlah_roi]:
                    poly = normalize_and_validate_roi(pts)
                    if poly is not None:
                        self.rois.append(poly)
                    else:
                        self.log_event.emit("WARN", "Satu ROI tersimpan diabaikan karena poligon tidak valid.")
                self._rebuild_masks()
                self._invalidate_reference(force=False)

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
                    norm_poly = normalize_and_validate_roi(self.rois[i])
                    if norm_poly is not None:
                        self.rois[i] = norm_poly
                self._rebuild_masks()
                self.config.saved_rois = [r.tolist() for r in self.rois]
                self.config.save(CONFIG_FILE)
                self._invalidate_reference(force=False)
                self.log_event.emit("INFO", "Bentuk sudut ROI diperbarui.")

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
                    self.titik_klik.append((px, py))
                    if len(self.titik_klik) == 4:
                        poly = normalize_and_validate_roi(self.titik_klik)
                        self.titik_klik.clear()
                        if poly is None:
                            self.log_event.emit("WARN", "Bentuk poligon ROI tidak valid. Klik ulang 4 sudut.")
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
                            self._invalidate_reference(force=False)
                            r_name = self.config.get_roi_name(target_idx)
                            self.log_event.emit("SUCCESS", f"{r_name} berhasil digambar ulang dan disimpan.")
                elif len(self.rois) < self.config.jumlah_roi:
                    self.titik_klik.append((px, py))
                    if len(self.titik_klik) == 4:
                        poly = normalize_and_validate_roi(self.titik_klik)
                        self.titik_klik.clear()
                        if poly is None:
                            self.log_event.emit("WARN", "Bentuk poligon ROI tidak valid. Klik ulang 4 sudut.")
                        else:
                            m = np.zeros((self.frame_h, self.frame_w), np.uint8)
                            cv2.fillPoly(m, [poly], 255)
                            self.rois.append(poly)
                            self.masks.append(m)
                            self.config.saved_rois = [r.tolist() for r in self.rois]
                            self.config.save(CONFIG_FILE)
                            self._invalidate_reference(force=False)

            elif cmd == "recapture_ref":
                if len(self.rois) == self.config.jumlah_roi:
                    self._invalidate_reference(force=True)
                    self.sequencer.clear_statuses()
                    self.log_event.emit("INFO", "Menunggu area kerja bersih untuk mengambil referensi baru...")

            elif cmd == "clear_status":
                self.sequencer.clear_statuses()
                self._init_post_hand_frames()
                self._roi_clear_frames = 0
                self.log_event.emit("INFO", "Status urutan langkah direset manual ke Langkah 1.")

            elif cmd == "reset_parts":
                self.sequencer.reset_parts()
                self.log_event.emit("INFO", "Counter Part telah direset ke 0.")

        if pending_seek_target is not None and self.is_video_mode:
            self._seek_to_frame(pending_seek_target)

        # Terapkan perubahan setting hardware kamera (hanya pada mode kamera live)
        if cam_props_to_apply and not self.is_video_mode and self.cap is not None and self.cap.isOpened():
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

    # =========================================================================
    # Loop Utama Thread Pekerja
    # =========================================================================
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

        # Muat ROI tersimpan
        if self.config.saved_rois and len(self.config.saved_rois) == self.config.jumlah_roi:
            self.load_saved_rois(self.config.saved_rois)

        prev_time = time.perf_counter()
        fps = 0.0

        while self.running:
            loop_start = time.perf_counter()
            self._process_pending_commands(self._prev_gray)

            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            is_new_frame = False
            frame: Optional[np.ndarray] = None

            if not self.is_video_mode:
                ok, raw = self.cap.read()
                if not self.running:
                    break
                if not ok or raw is None:
                    time.sleep(0.01)
                    continue
                self._raw_frame = raw
                frame = raw.copy()
                is_new_frame = True
            else:
                should_advance = (not self.video_paused) or self._step_once
                self._step_once = False

                if should_advance:
                    if self._needs_process_current and self._raw_frame is not None:
                        self._needs_process_current = False
                        frame = self._raw_frame.copy()
                        is_new_frame = True
                    elif self.video_current_frame >= self.video_end_frame:
                        if self.is_recording:
                            self._stop_recording_internal(completed=True)
                        self.video_paused = True
                        if self._raw_frame is not None:
                            frame = self._raw_frame.copy()
                    else:
                        ok, raw = self.cap.read()
                        if ok and raw is not None:
                            self.video_current_frame += 1
                            self._raw_frame = raw.copy()
                            frame = raw.copy()
                            is_new_frame = True
                        else:
                            if self.is_recording:
                                self._stop_recording_internal(completed=True)
                            self.video_paused = True
                            if self._raw_frame is not None:
                                frame = self._raw_frame.copy()
                else:
                    if self._raw_frame is not None:
                        frame = self._raw_frame.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            # Sinkronisasi ukuran
            self.sequencer.ensure_size(len(self.rois))
            self._ensure_post_hand_frames()

            if (frame.shape[1], frame.shape[0]) != (self.frame_w, self.frame_h):
                self.frame_h, self.frame_w = frame.shape[:2]
                if self.rois:
                    self._rebuild_masks()

            gray_now = siapkan_gray(frame)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            jumlah_roi = self.config.jumlah_roi
            banner_text = ""
            banner_state = BannerState.SETUP
            roi_telemetry = []

            # -----------------------------------------------------------------
            # MODE 1: SETUP POLIGON ROI
            # -----------------------------------------------------------------
            if len(self.rois) < jumlah_roi or self._redo_roi_idx is not None:
                if self._redo_roi_idx is not None:
                    r_name = self.config.get_roi_name(self._redo_roi_idx)
                    titik_ke = len(self.titik_klik) + 1
                    banner_text = (
                        f"MODE GAMBAR ULANG [{r_name}]: Klik sudut {titik_ke}/4 pada video (Klik kanan untuk batal)"
                    )
                else:
                    n = len(self.rois) + 1
                    titik_ke = len(self.titik_klik) + 1
                    banner_text = (
                        f"MODE SETUP ROI {n}/{jumlah_roi}: Klik sudut {titik_ke}/4 pada video (Klik kanan untuk batal)"
                    )
                banner_state = BannerState.SETUP

                self.renderer.draw_setup_mode(
                    frame=frame,
                    rois=self.rois,
                    titik_klik=self.titik_klik,
                    kursor=self.kursor,
                    redo_roi_idx=self._redo_roi_idx,
                    get_roi_name=self.config.get_roi_name,
                )

                for i in range(jumlah_roi):
                    r_name_i = self.config.get_roi_name(i)
                    if i < len(self.rois):
                        roi_telemetry.append({
                            "index": i,
                            "name": r_name_i,
                            "state": RoiState.READY.value,
                            "label": "Poligon Siap",
                            "ratio": 0.0,
                            "diff_ratio": 0.0,
                            "ambang_rasio": self.config.ambang_rasio,
                            "debounce": "",
                        })
                    elif i == len(self.rois):
                        roi_telemetry.append({
                            "index": i,
                            "name": r_name_i,
                            "state": RoiState.SETUP.value,
                            "label": f"Klik sudut {len(self.titik_klik)}/4",
                            "ratio": 0.0,
                            "diff_ratio": 0.0,
                            "ambang_rasio": self.config.ambang_rasio,
                            "debounce": "",
                        })
                    else:
                        roi_telemetry.append({
                            "index": i,
                            "name": r_name_i,
                            "state": RoiState.LOCKED.value,
                            "label": "Menunggu dibuat",
                            "ratio": 0.0,
                            "diff_ratio": 0.0,
                            "ambang_rasio": self.config.ambang_rasio,
                            "debounce": "",
                        })

            # -----------------------------------------------------------------
            # MODE 2: MONITORING DETEKSI & VERIFIKASI LANGKAH
            # -----------------------------------------------------------------
            else:
                detect_key = (
                    self.video_current_frame if self.is_video_mode else -1,
                    tuple(r.tobytes() for r in self.rois),
                    self.config.use_roi_crop,
                    self.config.roi_crop_mode,
                    self.config.roi_crop_padding,
                    self.config.use_clahe,
                    self.config.clahe_clip_limit,
                    self.config.clahe_invert_dark,
                    self.config.show_clahe_preview,
                    self.config.hand_backend,
                    self.config.rtm_score_thr,
                    self.config.yolo_model_path,
                    getattr(self.config, "yolo_target_class", "gloves"),
                )
                if (
                    not is_new_frame
                    and self.is_video_mode
                    and self._cached_detect_key == detect_key
                    and self._cached_detect_res is not None
                ):
                    tangan, crop_boxes, cached_bgr = self._cached_detect_res
                    if self.config.show_clahe_preview:
                        np.copyto(frame, cached_bgr)
                else:
                    try:
                        tangan, crop_boxes = self.pipeline.detect_hands(
                            frame=frame,
                            rois=self.rois,
                            config=self.config,
                        )
                    except Exception as exc:
                        self.log_event.emit(
                            LogLevel.WARN.value,
                            f"Gagal menjalankan backend '{self.config.hand_backend}': {exc}",
                        )
                        cur_h, cur_w = frame.shape[:2]
                        tangan, crop_boxes = [], [(0, 0, cur_w, cur_h)]
                    if self.is_video_mode:
                        self._cached_detect_key = detect_key
                        self._cached_detect_res = (tangan, crop_boxes, frame.copy())

                # Sinkronisasi parameter konfigurasi terbaru ke sequencer
                use_pixel_diff = bool(getattr(self.config, "use_pixel_diff", False))
                self.sequencer.debounce_masuk = max(1, int(self.config.debounce_masuk))
                self.sequencer.debounce_keluar = max(1, int(self.config.debounce_keluar))
                self.sequencer.ambang_rasio = float(self.config.ambang_rasio)
                self.sequencer.use_pixel_diff = use_pixel_diff

                # Deteksi gerak antar frame (fallback)
                motion_ratios = [
                    rasio_gerak_roi(
                        gray_now,
                        self._prev_gray,
                        self.masks[i],
                        max(10, self.config.ambang_piksel // 2),
                        self._morph_kernel,
                    )
                    for i in range(len(self.rois))
                ]
                any_roi_motion = any(mr >= self.config.motion_fallback_thr for mr in motion_ratios)

                # Stabilisasi referensi bersih (hanya diperlukan jika perbandingan piksel aktif)
                needed_clean = max(1, int(self.config.ref_clean_frames))
                if use_pixel_diff and is_new_frame:
                    if len(tangan) == 0 and not any_roi_motion:
                        self._no_hand_frames += 1
                        self._stable_ref_buffer.append(gray_now.copy())
                        if len(self._stable_ref_buffer) > needed_clean:
                            self._stable_ref_buffer.pop(0)
                    else:
                        self._no_hand_frames = 0
                        self._stable_ref_buffer.clear()

                    if self.referensi is None or self.referensi.shape != gray_now.shape:
                        if self._no_hand_frames >= needed_clean and len(self._stable_ref_buffer) >= needed_clean:
                            self.referensi = np.mean(self._stable_ref_buffer, axis=0).astype(np.uint8)
                            self.log_event.emit(
                                LogLevel.INFO.value,
                                f"Referensi background bersih berhasil dikunci (rata-rata {needed_clean} frame stabil).",
                            )

                # Visualisasi batas crop
                if self.config.use_roi_crop and self.config.show_crop_box:
                    self.renderer.draw_crop_boxes(frame, crop_boxes)

                aktif = self.sequencer.get_active_step()
                salah_urut: Optional[int] = None
                double_pick: Optional[int] = None
                overlap_thr = float(self.config.hand_overlap_thr)
                debounce_err = 2  # Minimal 2 frame untuk mencegah false alarm 1-frame flicker
                debounce_reentry = max(2, int(self.config.debounce_masuk))
                waiting_ref = use_pixel_diff and self.referensi is None

                # Pass 1: Hitung kehadiran tangan & gerak untuk setiap ROI
                roi_hand_flags: List[Tuple[bool, bool, bool]] = []
                for idx, poly in enumerate(self.rois):
                    ada_objek_bergerak = (
                        idx < len(motion_ratios) and motion_ratios[idx] >= self.config.motion_fallback_thr
                    )

                    # Uji apakah tangan masuk ke ROI
                    tangan_masuk_roi = False
                    ada_tangan_model = False

                    if self.config.hand_tracking_mode == TrackingMode.FULL_HAND:
                        for h in tangan:
                            ov_ratio = hull_roi_overlap_ratio(poly, h["hull"])
                            pc_in = cv2.pointPolygonTest(
                                poly, (float(h["palm_center"][0]), float(h["palm_center"][1])), False
                            ) >= 0
                            titik_di_roi = sum(
                                1 for p in h["all_points"]
                                if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
                            )
                            if ov_ratio >= overlap_thr or (pc_in and titik_di_roi >= 5):
                                tangan_masuk_roi = True
                            if ov_ratio >= overlap_thr * 0.5 or (pc_in and titik_di_roi >= 3):
                                ada_tangan_model = True

                    elif self.config.hand_tracking_mode == TrackingMode.PALM_CENTER:
                        tangan_masuk_roi = any(
                            cv2.pointPolygonTest(poly, h["palm_center"], False) >= 0 for h in tangan
                        )
                        ada_tangan_model = any(
                            cv2.pointPolygonTest(poly, h["palm_center"], False) >= 0
                            or sum(cv2.pointPolygonTest(poly, p, False) >= 0 for p in h["dorsum_points"]) >= 2
                            for h in tangan
                        )
                    else:  # FINGERTIPS
                        jumlah = [
                            sum(cv2.pointPolygonTest(poly, p, False) >= 0 for p in h["fingertips"])
                            for h in tangan
                        ]
                        tangan_masuk_roi = any(j == 5 for j in jumlah)
                        ada_tangan_model = any(j >= 2 for j in jumlah)

                    ada_tangan_atau_gerak = ada_tangan_model or ada_objek_bergerak
                    roi_hand_flags.append((tangan_masuk_roi, ada_tangan_model, ada_tangan_atau_gerak))

                # Hitung frame berturut-turut saat seluruh area ROI bebas tangan (untuk auto-cycle aman)
                any_hand_in_any_roi = any(
                    in_roi or has_hand for in_roi, has_hand, _ in roi_hand_flags
                )
                if is_new_frame:
                    if not any_hand_in_any_roi:
                        self._roi_clear_frames += 1
                    else:
                        self._roi_clear_frames = 0

                # Evaluasi seluruh langkah terkunci (jumping) & langkah DONE (double-pick) terlebih dahulu
                locked_errors: Dict[int, bool] = {}
                done_reentry_errors: Dict[int, bool] = {}
                if not waiting_ref and aktif is not None:
                    for idx in range(len(self.rois)):
                        if idx == aktif:
                            continue
                        tangan_masuk_roi = roi_hand_flags[idx][0]
                        if self.sequencer.statuses[idx] == StepStatus.DONE:
                            if is_new_frame:
                                is_reentry = self.sequencer.process_done_step_reentry(
                                    idx, tangan_masuk_roi, debounce_reentry
                                )
                            else:
                                is_reentry = tangan_masuk_roi or (
                                    self.sequencer.reentry_counts[idx] >= debounce_reentry
                                )
                            done_reentry_errors[idx] = is_reentry
                            if is_reentry and double_pick is None:
                                double_pick = idx
                        else:
                            if is_new_frame:
                                is_err = self.sequencer.process_locked_step(idx, tangan_masuk_roi, debounce_err)
                            else:
                                is_err = tangan_masuk_roi or (self.sequencer.error_counts[idx] >= debounce_err)
                            locked_errors[idx] = is_err
                            if is_err and salah_urut is None:
                                salah_urut = idx

                # Jika terjadi jumping proses atau double-pick, hentikan dan reset debounce langkah aktif ke 0
                step_violation = (salah_urut is not None) or (double_pick is not None)
                if step_violation and aktif is not None and is_new_frame:
                    self.sequencer.reset_active_step_debounce(aktif)
                    if aktif < len(self._post_hand_frames):
                        self._post_hand_frames[aktif].clear()

                # Pass 2: Evaluasi langkah aktif (hanya jika tidak ada pelanggaran step) & render overlay
                for idx, poly in enumerate(self.rois):
                    tangan_masuk_roi, _, ada_tangan_atau_gerak = roi_hand_flags[idx]
                    debounce_info = ""

                    # Evaluasi state langkah
                    if waiting_ref:
                        roi_state = RoiState.WAIT_REF
                        teks = f"Menunggu Ref Bersih ({self._no_hand_frames}/{needed_clean})"
                    elif self.sequencer.statuses[idx] == StepStatus.DONE:
                        if done_reentry_errors.get(idx, False):
                            roi_state = RoiState.ERROR
                            teks = "DOUBLE PICK"
                        else:
                            roi_state = RoiState.DONE
                            teks = "DONE"
                    elif idx == aktif:
                        if is_new_frame and not step_violation:
                            # Langkah aktif: proses masuk
                            if self.sequencer.statuses[idx] == StepStatus.WAITING:
                                _, debounce_info = self.sequencer.process_step_entry(idx, tangan_masuk_roi)

                            # Langkah aktif: proses keluar
                            if self.sequencer.statuses[idx] == StepStatus.HAND_IN:
                                if use_pixel_diff:
                                    if not ada_tangan_atau_gerak:
                                        self._post_hand_frames[idx].append(gray_now.copy())
                                        if len(self._post_hand_frames[idx]) > 5:
                                            self._post_hand_frames[idx].pop(0)
                                    else:
                                        self._post_hand_frames[idx].clear()

                                should_eval, debounce_info = self.sequencer.process_step_exit(
                                    idx, ada_tangan_atau_gerak
                                )
                                if should_eval:
                                    r_name = self.config.get_roi_name(idx)
                                    if use_pixel_diff:
                                        if self._post_hand_frames[idx]:
                                            gray_after = np.mean(self._post_hand_frames[idx], axis=0).astype(np.uint8)
                                        else:
                                            gray_after = gray_now
                                        self._post_hand_frames[idx].clear()

                                        ratio: Optional[float] = rasio_perubahan(
                                            gray_after,
                                            self.referensi,
                                            self.masks[idx],
                                            self.config.ambang_piksel,
                                            self._morph_kernel,
                                        )
                                    else:
                                        self._post_hand_frames[idx].clear()
                                        ratio = None

                                    completed, evt = self.sequencer.finalize_step_exit(idx, ratio, r_name)
                                    self.log_event.emit(evt.level.value, evt.message)
                                    if completed:
                                        self.sequencer.clear_error_counts()
                                        aktif = self.sequencer.get_active_step()

                        if self.sequencer.statuses[idx] == StepStatus.DONE:
                            roi_state = RoiState.DONE
                            teks = "DONE"
                        elif self.sequencer.statuses[idx] == StepStatus.HAND_IN or (
                            not is_new_frame and tangan_masuk_roi and not step_violation
                        ):
                            roi_state = RoiState.HAND_IN
                            teks = "Tangan di ROI"
                        else:
                            roi_state = RoiState.ACTIVE
                            teks = "Aktif"
                    else:
                        # Langkah terkunci (hasil evaluasi Pass 1)
                        is_err = locked_errors.get(idx, False)
                        if is_err:
                            roi_state = RoiState.ERROR
                            teks = "URUTAN SALAH"
                        else:
                            roi_state = RoiState.LOCKED
                            target_aktif = (aktif + 1) if aktif is not None else 1
                            teks = f"Terkunci (Langkah {target_aktif})"

                    self.renderer.draw_roi_card_overlay(
                        frame=frame,
                        poly=poly,
                        idx=idx,
                        state=roi_state,
                        label_text=teks,
                        roi_name=self.config.get_roi_name(idx),
                    )

                    roi_telemetry.append({
                        "index": idx,
                        "name": self.config.get_roi_name(idx),
                        "state": roi_state.value,
                        "label": teks,
                        "ratio": self.sequencer.last_ratios[idx],
                        "diff_ratio": self.sequencer.last_ratios[idx],
                        "ambang_rasio": self.config.ambang_rasio,
                        "use_pixel_diff": use_pixel_diff,
                        "debounce": debounce_info,
                    })

                # Visualisasi tangan
                self.renderer.draw_hands(frame, tangan, self.config.hand_tracking_mode)

                # Banner Text & Auto-Cycle Completion
                aktif_now = self.sequencer.get_active_step()
                if use_pixel_diff and self.referensi is None:
                    if self.is_video_mode and self.video_paused:
                        banner_text = (
                            f"PILIH FRAME REFERENSI: Geser ke frame bersih lalu klik 'Jadikan Frame Ini Referensi' "
                            f"atau Putar video ({self._no_hand_frames}/{needed_clean})"
                        )
                    else:
                        banner_text = (
                            f"MENUNGGU AREA BERSIH: Jauhkan tangan dari ROI ({self._no_hand_frames}/{needed_clean})..."
                        )
                    banner_state = BannerState.SETUP
                elif aktif_now is None:
                    if is_new_frame:
                        is_done, cooldown_expired, part_evt = self.sequencer.handle_cycle_completion(now)
                        if part_evt:
                            self.log_event.emit(part_evt.level.value, part_evt.message)
                    else:
                        cooldown_expired = False

                    banner_state = BannerState.DONE
                    if cooldown_expired:
                        if not use_pixel_diff:
                            if self._roi_clear_frames >= needed_clean:
                                self.sequencer.clear_statuses()
                                self._roi_clear_frames = 0
                                self.log_event.emit(
                                    LogLevel.INFO.value,
                                    f"Mulai siklus Part #{self.sequencer.parts_completed + 1}...",
                                )
                                banner_text = f"PART #{self.sequencer.parts_completed} SELESAI (+1). Memulai siklus baru..."
                            else:
                                banner_text = (
                                    f"PART #{self.sequencer.parts_completed} SELESAI. Jauhkan tangan dari ROI untuk siklus baru "
                                    f"({self._roi_clear_frames}/{needed_clean})..."
                                )
                        elif self._no_hand_frames >= needed_clean and len(self._stable_ref_buffer) >= needed_clean:
                            self.sequencer.clear_statuses()
                            self._roi_clear_frames = 0
                            self.referensi = np.mean(self._stable_ref_buffer, axis=0).astype(np.uint8)
                            self.log_event.emit(
                                LogLevel.INFO.value,
                                f"Mulai siklus Part #{self.sequencer.parts_completed + 1}...",
                            )
                            banner_text = f"PART #{self.sequencer.parts_completed} SELESAI (+1). Memulai siklus baru..."
                        else:
                            banner_text = (
                                f"PART #{self.sequencer.parts_completed} SELESAI. Jauhkan tangan untuk kalibrasi ref "
                                f"({self._no_hand_frames}/{needed_clean})..."
                            )
                    else:
                        banner_text = f"PART #{self.sequencer.parts_completed} SELESAI (+1). Menyiapkan siklus berikutnya..."
                else:
                    if salah_urut is not None and salah_urut != aktif_now:
                        target_name = self.config.get_roi_name(aktif_now)
                        wrong_name = self.config.get_roi_name(salah_urut)
                        banner_text = f"URUTAN SALAH! Kerjakan [{target_name}], bukan [{wrong_name}]"
                        banner_state = BannerState.ERROR
                        if is_new_frame and now - self.sequencer.last_error_log_time > 2.0:
                            self.sequencer.last_error_log_time = now
                            self.log_event.emit(
                                LogLevel.WARN.value,
                                f"Urutan salah! Tangan di [{wrong_name}] (Seharusnya [{target_name}])",
                            )
                    elif double_pick is not None:
                        target_name = self.config.get_roi_name(aktif_now)
                        done_name = self.config.get_roi_name(double_pick)
                        banner_text = f"DOUBLE PICK! [{done_name}] sudah selesai, lanjut ke [{target_name}]"
                        banner_state = BannerState.ERROR
                        if is_new_frame and now - self.sequencer.last_error_log_time > 2.0:
                            self.sequencer.last_error_log_time = now
                            self.log_event.emit(
                                LogLevel.WARN.value,
                                f"Double pick terdeteksi! Tangan kembali ke [{done_name}] yang sudah selesai (Lanjut ke [{target_name}])",
                            )
                    else:
                        step_name = self.config.get_roi_name(aktif_now)
                        banner_text = f"Ambil [{step_name}] pada ROI {aktif_now + 1}"
                        banner_state = BannerState.ACTIVE

            if is_new_frame:
                self._prev_gray = gray_now.copy()

            # Tulis frame teranotasi penuh ke file video output jika sedang merekam
            if self.is_recording and self.video_writer is not None and is_new_frame:
                try:
                    self.video_writer.write(frame)
                    self.recorded_frames_count += 1
                except Exception as e:
                    self.log_event.emit(LogLevel.WARN.value, f"Gagal menulis frame ke video output: {e}")
                    self._stop_recording_internal(completed=False)

            # Jika telah mencapai frame akhir saat perekaman video, selesaikan otomatis
            if (
                self.is_video_mode
                and self.is_recording
                and is_new_frame
                and self.video_current_frame >= self.video_end_frame
            ):
                self._stop_recording_internal(completed=True)
                self.video_paused = True

            # Konversi BGR -> QImage dan pancarkan sinyal
            try:
                rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_out.shape
                qimg = QImage(rgb_out.data, w, h, ch * w, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg, w, h)
            except Exception:
                pass

            telemetry_data = {
                "banner_text": banner_text,
                "banner_state": banner_state.value if isinstance(banner_state, BannerState) else banner_state,
                "rois": roi_telemetry,
                "rois_coords": [r.tolist() for r in self.rois],
                "fps": fps,
                "has_reference": self.referensi is not None,
                "use_pixel_diff": bool(getattr(self.config, "use_pixel_diff", False)),
                "roi_defined": len(self.rois),
                "roi_total": jumlah_roi,
                "steps_completed": self.sequencer.get_steps_completed(),
                "parts_completed": self.sequencer.parts_completed,
                "is_video_mode": self.is_video_mode,
                "video_path": self.video_path or "",
                "video_paused": self.video_paused,
                "video_fps": self.video_fps,
                "video_total_frames": self.video_total_frames,
                "video_current_frame": self.video_current_frame,
                "video_start_frame": self.video_start_frame,
                "video_end_frame": self.video_end_frame,
                "is_recording": self.is_recording,
                "recording_path": self.recording_path or "",
                "recorded_frames_count": self.recorded_frames_count,
            }

            with self._web_lock:
                self._latest_bgr_frame = frame
                self._latest_telemetry = telemetry_data

            self.telemetry_updated.emit(telemetry_data)

            # Pacing loop pada mode video
            if self.is_video_mode:
                if self.video_paused:
                    time.sleep(0.03)
                elif not self.is_recording and self.video_fps > 0:
                    elapsed = time.perf_counter() - loop_start
                    target_dt = 1.0 / self.video_fps
                    if elapsed < target_dt:
                        time.sleep(max(0.001, target_dt - elapsed))

    def stop(self) -> None:
        """Menghentikan thread secara aman, mengosongkan antrian, dan mematikan seluruh kamera."""
        with self._stop_lock:
            self._is_stopped = True
            self.running = False

        # Kosongkan antrian perintah agar tidak ada pembukaan kamera susulan saat shutdown
        while True:
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                break

        # Tunggu thread keluar dari loop run() sebelum melepas paksa seluruh resource
        if self.isRunning() and QThread.currentThread() != self:
            self.wait(3000)

        # Pastikan seluruh kamera dan perekam video benar-benar dilepas
        self._cleanup_resources()
