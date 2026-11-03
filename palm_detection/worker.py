"""CameraWorker: QThread orkestrator akuisisi kamera, deteksi AI, state machine, dan rendering."""

from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from palm_detection.config import AppConfig, CONFIG_FILE
from palm_detection.detector import HandDetectionPipeline
from palm_detection.renderer import OverlayRenderer, WARNA_ROI_BGR
from palm_detection.sequencer import StepSequencer
from palm_detection.states import BannerState, LogLevel, RoiState, TrackingMode
from palm_detection.vision import (
    clean_diff_mask,
    hull_roi_overlap_ratio,
    normalize_and_validate_roi,
    rasio_gerak_roi,
    rasio_perubahan,
    siapkan_gray,
)


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

        # State kamera (HANYA dibuat, diakses, dan dirilis oleh thread worker)
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_w = config.frame_width
        self.frame_h = config.frame_height

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
        self._stable_ref_buffer: List[np.ndarray] = []
        self._post_hand_frames: List[List[np.ndarray]] = []
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Modul Terpisah
        self.sequencer = StepSequencer(
            num_steps=config.jumlah_roi,
            debounce_masuk=config.debounce_masuk,
            debounce_keluar=config.debounce_keluar,
            ambang_rasio=config.ambang_rasio,
        )
        self.pipeline = HandDetectionPipeline()
        self.renderer = OverlayRenderer()

        # Antrian perintah thread-safe dari GUI
        self._cmd_queue: queue.Queue = queue.Queue()

        self._init_post_hand_frames()

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
        self.post_command("clear_status", None)

    def trigger_reset_parts(self) -> None:
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
    # Pengelolaan Kamera (Hanya berjalan di worker thread)
    # =========================================================================
    def _open_camera(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        def _try_open(idx: int):
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not c.isOpened():
                c.release()
                c = cv2.VideoCapture(idx)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    try:
                        c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 150)
                    except Exception:
                        pass
                success, f = c.read()
                if success and f is not None:
                    return c, f
                c.release()
            return None, None

        cap_obj, frame = _try_open(self.config.camera_index)
        if cap_obj is None:
            for alt_idx in [1, 0, 2]:
                if alt_idx == self.config.camera_index:
                    continue
                cap_obj, frame = _try_open(alt_idx)
                if cap_obj is not None:
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
        self.sequencer.reset(self.config.jumlah_roi)
        self._init_post_hand_frames()

    def _cleanup_resources(self) -> None:
        """Melepas kamera dan model AI di akhir eksekusi thread worker."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[Peringatan] Gagal merilis VideoCapture: {e}")
            finally:
                self.cap = None

        self.pipeline.close()

    # =========================================================================
    # Pemrosesan Perintah Masuk dari GUI Thread
    # =========================================================================
    def _process_pending_commands(self, gray_now: Optional[np.ndarray]) -> None:
        cam_props_to_apply: dict = {}

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
                    norm_poly = normalize_and_validate_roi(self.rois[i])
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
                            self._invalidate_reference()
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
                            self._invalidate_reference()

            elif cmd == "recapture_ref":
                if len(self.rois) == self.config.jumlah_roi:
                    self._invalidate_reference()
                    self.sequencer.clear_statuses()
                    self.log_event.emit("INFO", "Menunggu area kerja bersih untuk mengambil referensi baru...")

            elif cmd == "clear_status":
                self.sequencer.clear_statuses()
                self._init_post_hand_frames()
                self.log_event.emit("INFO", "Status urutan langkah direset manual ke Langkah 1.")

            elif cmd == "reset_parts":
                self.sequencer.reset_parts()
                self.log_event.emit("INFO", "Counter Part telah direset ke 0.")

        # Terapkan perubahan setting hardware kamera
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
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                self._process_pending_commands(None)
                continue

            ok, frame = self.cap.read()
            if not self.running:
                break
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # Sinkronisasi ukuran
            self.sequencer.ensure_size(len(self.rois))
            self._ensure_post_hand_frames()

            if (frame.shape[1], frame.shape[0]) != (self.frame_w, self.frame_h):
                self.frame_h, self.frame_w = frame.shape[:2]
                if self.rois:
                    self._rebuild_masks()

            gray_now = siapkan_gray(frame)
            self._process_pending_commands(gray_now)

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
                    if i < len(self.rois):
                        roi_telemetry.append({
                            "index": i,
                            "state": RoiState.READY.value,
                            "label": "Poligon Siap",
                            "ratio": 0.0,
                            "debounce": "",
                        })
                    elif i == len(self.rois):
                        roi_telemetry.append({
                            "index": i,
                            "state": RoiState.SETUP.value,
                            "label": f"Klik sudut {len(self.titik_klik)}/4",
                            "ratio": 0.0,
                            "debounce": "",
                        })
                    else:
                        roi_telemetry.append({
                            "index": i,
                            "state": RoiState.LOCKED.value,
                            "label": "Menunggu dibuat",
                            "ratio": 0.0,
                            "debounce": "",
                        })

            # -----------------------------------------------------------------
            # MODE 2: MONITORING DETEKSI & VERIFIKASI LANGKAH
            # -----------------------------------------------------------------
            else:
                tangan, crop_boxes = self.pipeline.detect_hands(
                    frame=frame,
                    rois=self.rois,
                    config=self.config,
                )

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

                # Stabilisasi referensi bersih
                needed_clean = max(1, int(self.config.ref_clean_frames))
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
                overlap_thr = float(self.config.hand_overlap_thr)
                debounce_err = max(2, int(self.config.debounce_masuk))

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
                    debounce_info = ""

                    # Evaluasi state langkah
                    if self.referensi is None:
                        roi_state = RoiState.WAIT_REF
                        teks = f"Menunggu Ref Bersih ({self._no_hand_frames}/{needed_clean})"
                    elif self.sequencer.statuses[idx] == self.sequencer.statuses[idx].DONE:
                        roi_state = RoiState.DONE
                        teks = "DONE"
                    elif idx == aktif:
                        # Langkah aktif: proses masuk
                        if self.sequencer.statuses[idx] == self.sequencer.statuses[idx].WAITING:
                            _, debounce_info = self.sequencer.process_step_entry(idx, tangan_masuk_roi)

                        # Langkah aktif: proses keluar
                        if self.sequencer.statuses[idx] == self.sequencer.statuses[idx].HAND_IN:
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
                                if self._post_hand_frames[idx]:
                                    gray_after = np.mean(self._post_hand_frames[idx], axis=0).astype(np.uint8)
                                else:
                                    gray_after = gray_now
                                self._post_hand_frames[idx].clear()

                                ratio = rasio_perubahan(
                                    gray_after,
                                    self.referensi,
                                    self.masks[idx],
                                    self.config.ambang_piksel,
                                    self._morph_kernel,
                                )
                                r_name = self.config.get_roi_name(idx)
                                _, evt = self.sequencer.finalize_step_exit(idx, ratio, r_name)
                                self.log_event.emit(evt.level.value, evt.message)

                        if self.sequencer.statuses[idx] == self.sequencer.statuses[idx].DONE:
                            roi_state = RoiState.DONE
                            teks = "DONE"
                        elif self.sequencer.statuses[idx] == self.sequencer.statuses[idx].HAND_IN:
                            roi_state = RoiState.HAND_IN
                            teks = "Tangan di ROI"
                        else:
                            roi_state = RoiState.ACTIVE
                            teks = "Aktif"
                    else:
                        # Langkah terkunci
                        is_err = self.sequencer.process_locked_step(idx, tangan_masuk_roi, debounce_err)
                        if is_err:
                            salah_urut = idx
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
                        "state": roi_state.value,
                        "label": teks,
                        "ratio": self.sequencer.last_ratios[idx],
                        "debounce": debounce_info,
                    })

                # Visualisasi tangan
                self.renderer.draw_hands(frame, tangan, self.config.hand_tracking_mode)

                # Banner Text & Auto-Cycle Completion
                aktif_now = self.sequencer.get_active_step()
                if self.referensi is None:
                    banner_text = (
                        f"MENUNGGU AREA BERSIH: Jauhkan tangan dari ROI ({self._no_hand_frames}/{needed_clean})..."
                    )
                    banner_state = BannerState.SETUP
                elif aktif_now is None:
                    is_done, cooldown_expired, part_evt = self.sequencer.handle_cycle_completion(now)
                    if part_evt:
                        self.log_event.emit(part_evt.level.value, part_evt.message)

                    banner_state = BannerState.DONE
                    if cooldown_expired:
                        if self._no_hand_frames >= needed_clean and len(self._stable_ref_buffer) >= needed_clean:
                            self.sequencer.clear_statuses()
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
                    if salah_urut is not None:
                        target_name = self.config.get_roi_name(aktif_now)
                        wrong_name = self.config.get_roi_name(salah_urut)
                        banner_text = f"URUTAN SALAH! Kerjakan [{target_name}], bukan [{wrong_name}]"
                        banner_state = BannerState.ERROR
                        if now - self.sequencer.last_error_log_time > 2.0:
                            self.sequencer.last_error_log_time = now
                            self.log_event.emit(
                                LogLevel.WARN.value,
                                f"Urutan salah! Tangan di [{wrong_name}] (Seharusnya [{target_name}])",
                            )
                    else:
                        step_name = self.config.get_roi_name(aktif_now)
                        banner_text = f"Ambil [{step_name}] pada ROI {aktif_now + 1}"
                        banner_state = BannerState.ACTIVE

            self._prev_gray = gray_now.copy()

            # Konversi BGR -> QImage dan pancarkan sinyal
            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_out.shape
            qimg = QImage(rgb_out.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

            self.frame_ready.emit(qimg, w, h)
            self.telemetry_updated.emit({
                "banner_text": banner_text,
                "banner_state": banner_state.value if isinstance(banner_state, BannerState) else banner_state,
                "rois": roi_telemetry,
                "rois_coords": [r.tolist() for r in self.rois],
                "fps": fps,
                "has_reference": self.referensi is not None,
                "roi_defined": len(self.rois),
                "roi_total": jumlah_roi,
                "steps_completed": self.sequencer.get_steps_completed(),
                "parts_completed": self.sequencer.parts_completed,
            })

    def stop(self) -> None:
        """Menghentikan thread secara aman, non-blocking, dan idempoten tanpa terminate()."""
        with self._stop_lock:
            if self._is_stopped:
                return
            self._is_stopped = True
            self.running = False

        # Tunggu thread keluar dari loop run() dengan batas waktu maksimal 600ms
        self.wait(600)
