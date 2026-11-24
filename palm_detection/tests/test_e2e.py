"""End-to-End (E2E) Test Suite untuk seluruh fitur aplikasi Palm Detection.

Mencakup pengujian menyeluruh untuk:
    1. Konfigurasi & persistensi JSON (AppConfig)
    2. Utilitas visi komputer & geometri poligon (vision)
    3. Preprocessing sarung tangan CLAHE & pipeline deteksi tangan (detector)
    4. State machine urutan langkah Poka-Yoke (StepSequencer)
    5. Penggambaran overlay grafis ROI & tangan (OverlayRenderer)
    6. Orkestrasi CameraWorker: input video, penentuan frame oleh pengguna (seek/step/range),
       kunci frame referensi manual, deteksi urutan langkah, dan ekspor output video teranotasi
    7. Integrasi penuh antarmuka PyQt5 (MainWindow, VideoCanvas, ROICardWidget, SettingsDialog)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
from typing import List, Tuple
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication

from palm_detection.config import AppConfig
from palm_detection.detector import (
    MODEL_DIR,
    HandDetectionPipeline,
    HandDetector,
    SENDI_PUNGGUNG,
    UJUNG_JARI,
    YOLOPoseHandDetector,
    ensure_model_dir,
    list_available_yolo_models,
    preprocess_glove,
    resolve_yolo_model_path,
)
from palm_detection.gui import MainWindow, ROICardWidget, SettingsDialog, VideoCanvas
from palm_detection.renderer import OverlayRenderer
from palm_detection.sequencer import StepSequencer
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
from palm_detection.worker import CameraWorker, _ACTIVE_CAPTURES, release_all_cameras

# Pastikan satu instance QApplication tersedia untuk seluruh test GUI/QThread
_QT_APP = QApplication.instance() or QApplication(sys.argv)


class DummyHandDetector(HandDetector):
    """Detektor tangan deterministik untuk pengujian E2E tanpa bergantung model eksternal."""

    def __init__(self, target_center: Tuple[int, int] | None = None) -> None:
        self.target_center = target_center

    @staticmethod
    def make_hand_at(cx: int, cy: int, spread: int = 25) -> dict:
        """Membuat struktur data tangan 21 sendi lengkap di sekitar titik (cx, cy)."""
        kpts: List[Tuple[int, int]] = []
        for i in range(21):
            angle = (i / 21.0) * 2.0 * np.pi
            r = spread if i in UJUNG_JARI else spread // 2
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            kpts.append((px, py))

        fingertips = [kpts[i] for i in UJUNG_JARI]
        dorsum = [kpts[i] for i in SENDI_PUNGGUNG]
        hull = cv2.convexHull(np.array(kpts, dtype=np.int32))
        return {
            "fingertips": fingertips,
            "palm_center": (cx, cy),
            "dorsum_points": dorsum,
            "all_points": kpts,
            "keypoints_21": list(kpts),
            "hull": hull,
        }

    def detect(self, crop_bgr: np.ndarray) -> List[dict]:
        if self.target_center is None:
            return []
        return [self.make_hand_at(self.target_center[0], self.target_center[1])]

    def close(self) -> None:
        self.target_center = None


def create_synthetic_assembly_video(
    filepath: str,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> Tuple[List[List[List[int]]], List[Tuple[int, int] | None]]:
    """Membuat file video sintetis (.mp4) untuk simulasi siklus pengambilan 2 komponen.

    Skenario frame (total 20 frame):
        - Frame 0..3   : Meja bersih (kedua komponen masih di dalam ROI 1 dan ROI 2)
        - Frame 4..6   : Tangan masuk ke ROI 1 (komponen 1 diambil)
        - Frame 7..10  : Tangan keluar dari ROI 1 (area ROI 1 berubah warna karena part terambil)
        - Frame 11..13 : Tangan masuk ke ROI 2 (komponen 2 diambil)
        - Frame 14..19 : Tangan keluar dari ROI 2 (area ROI 2 berubah warna -> Siklus Part #1 selesai)
    """
    roi_1 = [[80, 120], [240, 120], [240, 300], [80, 300]]
    roi_2 = [[360, 120], [520, 120], [520, 300], [360, 300]]
    rois = [roi_1, roi_2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    assert writer.isOpened(), f"Gagal membuat video sintetis di {filepath}"

    hand_positions: List[Tuple[int, int] | None] = []

    for f_idx in range(20):
        frame = np.full((height, width, 3), 45, dtype=np.uint8)

        # Gambar tekstur part di ROI 1 jika belum diambil (f_idx < 7)
        if f_idx < 7:
            cv2.rectangle(frame, (100, 140), (220, 280), (200, 200, 200), -1)
        else:
            # Setelah part 1 diambil, piksel di ROI 1 berubah signifikan dibanding referensi
            cv2.rectangle(frame, (100, 140), (220, 280), (20, 20, 20), -1)

        # Gambar tekstur part di ROI 2 jika belum diambil (f_idx < 14)
        if f_idx < 14:
            cv2.rectangle(frame, (380, 140), (500, 280), (200, 200, 200), -1)
        else:
            # Setelah part 2 diambil, piksel di ROI 2 berubah signifikan dibanding referensi
            cv2.rectangle(frame, (380, 140), (500, 280), (20, 20, 20), -1)

        # Tentukan posisi tangan per frame
        if 4 <= f_idx <= 6:
            hand_pos: Tuple[int, int] | None = (160, 210)  # Di tengah ROI 1
        elif 11 <= f_idx <= 13:
            hand_pos = (440, 210)  # Di tengah ROI 2
        else:
            hand_pos = None

        hand_positions.append(hand_pos)
        writer.write(frame)

    writer.release()
    return rois, hand_positions


class TestAppConfigAndVision(unittest.TestCase):
    """Pengujian unit & integrasi untuk AppConfig dan fungsi pengolahan citra vision.py."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_config_save_load_and_roi_names(self) -> None:
        cfg_path = os.path.join(self.tmpdir, "test_config.json")
        cfg = AppConfig(jumlah_roi=3, ambang_piksel=42, ambang_rasio=0.12)
        cfg.set_roi_name(0, "Baut M8")
        cfg.set_roi_name(2, "Soket Lampu")
        cfg.saved_rois = [[[10, 10], [100, 10], [100, 100], [10, 100]]]
        cfg.save(cfg_path)

        loaded = AppConfig.load(cfg_path)
        self.assertEqual(loaded.jumlah_roi, 3)
        self.assertEqual(loaded.ambang_piksel, 42)
        self.assertAlmostEqual(loaded.ambang_rasio, 0.12)
        self.assertEqual(loaded.get_roi_name(0), "Baut M8")
        self.assertEqual(loaded.get_roi_name(1), "ROI #2")
        self.assertEqual(loaded.get_roi_name(2), "Soket Lampu")
        self.assertEqual(len(loaded.saved_rois), 1)
        self.assertTrue(AppConfig.has_field("hand_backend"))
        self.assertFalse(AppConfig.has_field("non_existent_field"))

        other = AppConfig(jumlah_roi=5)
        other.update_from(loaded)
        self.assertEqual(other.jumlah_roi, 3)

    def test_vision_geometry_and_pixel_diff(self) -> None:
        # 1. Validasi poligon konveks 4 titik
        pts = [(200, 100), (100, 100), (100, 200), (200, 200)]
        poly = normalize_and_validate_roi(pts)
        self.assertIsNotNone(poly)
        self.assertEqual(poly.shape, (4, 2))

        # Poligon degeneratif (luas terlalu kecil / kolinear) harus ditolak
        bad_pts = [(10, 10), (11, 10), (12, 10), (13, 10)]
        self.assertIsNone(normalize_and_validate_roi(bad_pts))

        # 2. Overlap rasio antara hull tangan dan ROI
        hand = DummyHandDetector.make_hand_at(150, 150, spread=30)
        ov = hull_roi_overlap_ratio(poly, hand["hull"])
        self.assertGreater(ov, 0.15)

        hand_outside = DummyHandDetector.make_hand_at(400, 400, spread=20)
        self.assertAlmostEqual(hull_roi_overlap_ratio(poly, hand_outside["hull"]), 0.0)

        # 3. Rasio perubahan piksel & gerak di dalam ROI
        img_ref = np.full((300, 300, 3), 50, dtype=np.uint8)
        img_after = img_ref.copy()
        cv2.rectangle(img_after, (110, 110), (190, 190), (220, 220, 220), -1)

        g_ref = siapkan_gray(img_ref)
        g_after = siapkan_gray(img_after)
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        ratio_same = rasio_perubahan(g_ref, g_ref, mask, 25, kernel)
        self.assertEqual(ratio_same, 0.0)

        ratio_diff = rasio_perubahan(g_after, g_ref, mask, 25, kernel)
        self.assertGreater(ratio_diff, 0.40)

        motion = rasio_gerak_roi(g_after, g_ref, mask, 15, kernel)
        self.assertGreater(motion, 0.40)


class TestDetectorAndRenderer(unittest.TestCase):
    """Pengujian pipeline deteksi tangan, filter CLAHE sarung tangan, dan OverlayRenderer."""

    def test_preprocess_glove_and_pipeline_crop(self) -> None:
        frame = np.random.randint(40, 180, (480, 640, 3), dtype=np.uint8)
        pseudo_rgb, clahe_obj = preprocess_glove(frame, clip_limit=3.5, invert_dark=True)
        self.assertEqual(pseudo_rgb.shape, (480, 640, 3))
        self.assertIsNotNone(clahe_obj)

        pipeline = HandDetectionPipeline()
        roi_1 = np.array([[50, 50], [200, 50], [200, 200], [50, 200]], dtype=np.int32)
        roi_2 = np.array([[300, 50], [450, 50], [450, 200], [300, 200]], dtype=np.int32)

        # Union crop menghasilkan 1 kotak gabungan
        union_boxes = pipeline.compute_crop_regions(
            [roi_1, roi_2], 640, 480, use_crop=True, crop_mode=CropMode.UNION, padding=20
        )
        self.assertEqual(len(union_boxes), 1)
        self.assertEqual(union_boxes[0], (30, 30, 470, 220))

        # Individual crop menghasilkan 2 kotak terpisah
        indiv_boxes = pipeline.compute_crop_regions(
            [roi_1, roi_2], 640, 480, use_crop=True, crop_mode=CropMode.INDIVIDUAL, padding=10
        )
        self.assertEqual(len(indiv_boxes), 2)

        # Uji detect_hands dengan DummyHandDetector dan show_clahe_preview=True
        cfg = AppConfig(
            use_roi_crop=True,
            roi_crop_mode=CropMode.UNION,
            roi_crop_padding=20,
            use_clahe=True,
            show_clahe_preview=True,
            hand_backend=HandBackend.RTMLIB,
        )
        dummy_det = DummyHandDetector(target_center=(100, 100))
        with patch.object(pipeline, "get_rtm", return_value=dummy_det):
            hands, boxes = pipeline.detect_hands(frame, [roi_1, roi_2], cfg)

        self.assertEqual(len(hands), 1)
        # Koordinat lokal (100, 100) di dalam crop (30, 30) harus ditransformasi ke global (130, 130)
        self.assertEqual(hands[0]["palm_center"], (130, 130))
        pipeline.close()

    def test_yolo26n_pose_hand_detector_21kpt_and_17kpt_coco(self) -> None:
        """Menguji YOLOPoseHandDetector untuk model 21-keypoint Hand Pose maupun 17-keypoint COCO (yolo26n-pose.pt)."""
        # 1. Uji model YOLO Hand Pose 21-keypoint
        kpts_21 = np.zeros((1, 21, 2), dtype=np.float32)
        confs_21 = np.full((1, 21), 0.9, dtype=np.float32)
        for i in range(21):
            angle = (i / 21.0) * 2.0 * np.pi
            kpts_21[0, i, 0] = 200.0 + 30.0 * np.cos(angle)
            kpts_21[0, i, 1] = 180.0 + 30.0 * np.sin(angle)

        hands_21 = YOLOPoseHandDetector.parse_keypoints_array(kpts_21, confs_21, score_thr=0.3)
        self.assertEqual(len(hands_21), 1)
        self.assertEqual(len(hands_21[0]["keypoints_21"]), 21)
        self.assertEqual(len(hands_21[0]["fingertips"]), 5)
        self.assertIsNotNone(hands_21[0]["hull"])

        # 2. Uji model standar Ultralytics yolo26n-pose.pt (17-keypoint COCO Body Pose)
        #    Indeks 7=left_elbow, 9=left_wrist, 8=right_elbow, 10=right_wrist
        kpts_17 = np.zeros((1, 17, 2), dtype=np.float32)
        confs_17 = np.zeros((1, 17), dtype=np.float32)
        # Set siku kiri di (100, 300) dan pergelangan kiri di (100, 220) -> arah telapak ke atas (y mengecil)
        kpts_17[0, 7] = [100.0, 300.0]
        confs_17[0, 7] = 0.85
        kpts_17[0, 9] = [100.0, 220.0]
        confs_17[0, 9] = 0.92

        hands_17 = YOLOPoseHandDetector.parse_keypoints_array(kpts_17, confs_17, score_thr=0.3)
        self.assertEqual(len(hands_17), 1)
        self.assertEqual(len(hands_17[0]["keypoints_21"]), 21)
        # Pusat telapak harus terproyeksi di depan pergelangan tangan (y < 220)
        self.assertLess(hands_17[0]["palm_center"][1], 220)

        # 3. Uji integrasi HandDetectionPipeline dengan hand_backend = HandBackend.YOLO26N
        pipeline = HandDetectionPipeline()
        cfg = AppConfig(
            use_roi_crop=False,
            use_clahe=False,
            hand_backend=HandBackend.YOLO26N,
            yolo_model_path="yolo26n-pose.pt",
        )
        dummy_yolo = DummyHandDetector(target_center=(250, 200))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = np.array([[50, 50], [400, 50], [400, 400], [50, 400]], dtype=np.int32)
        with patch.object(pipeline, "get_yolo", return_value=dummy_yolo):
            detected, _ = pipeline.detect_hands(frame, [roi], cfg)
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0]["palm_center"], (250, 200))
        pipeline.close()

    def test_yolo_object_detection_glove_boxes_and_pipeline(self) -> None:
        """Menguji pemrosesan bounding box deteksi sarung tangan dari model YOLO Object Detection."""
        # 1. Mock kelas Boxes dari Ultralytics YOLO
        class MockBoxes:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = np.array(xyxy, dtype=np.float32)
                self.conf = np.array(conf, dtype=np.float32)
                self.cls = np.array(cls, dtype=np.int32)

        # 4 kotak: 1 valid 'gloves', 1 confidence rendah, 1 koordinat tidak valid, 1 class lain ('bulb' -> harus diabaikan)
        mock_boxes = MockBoxes(
            xyxy=[
                [50.0, 60.0, 150.0, 160.0],
                [200.0, 200.0, 250.0, 250.0],
                [100.0, 100.0, 50.0, 50.0],
                [300.0, 300.0, 400.0, 400.0],
            ],
            conf=[0.88, 0.20, 0.95, 0.92],
            cls=[0, 0, 0, 1],
        )

        names_dict = {0: "gloves", 1: "bulb"}
        hands = YOLOPoseHandDetector.parse_boxes(
            mock_boxes, names_dict=names_dict, score_thr=0.35, target_classes="gloves"
        )
        self.assertEqual(len(hands), 1)
        glove = hands[0]
        self.assertEqual(glove["palm_center"], (100, 110))
        self.assertEqual(glove["bbox"], (50, 60, 150, 160))
        self.assertAlmostEqual(glove["confidence"], 0.88, places=2)
        self.assertEqual(glove["label"], "gloves")
        self.assertEqual(len(glove["fingertips"]), 5)
        self.assertEqual(len(glove["keypoints_21"]), 21)
        self.assertIsNotNone(glove["hull"])
        self.assertEqual(len(glove["hull"]), 4)

        # 2. Uji pipeline integrasi dengan HandBackend.YOLO_DETECT
        pipeline = HandDetectionPipeline()
        cfg = AppConfig(
            use_roi_crop=False,
            use_clahe=False,
            hand_backend=HandBackend.YOLO_DETECT,
            yolo_model_path="custom_glove_yolo.pt",
        )
        dummy_yolo = DummyHandDetector(target_center=(100, 110))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = np.array([[30, 30], [200, 30], [200, 200], [30, 200]], dtype=np.int32)
        with patch.object(pipeline, "get_yolo", return_value=dummy_yolo):
            detected, _ = pipeline.detect_hands(frame, [roi], cfg)
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0]["palm_center"], (100, 110))
        pipeline.close()

        # 3. Uji render overlay untuk bounding box sarung tangan
        renderer = OverlayRenderer()
        annotated = frame.copy()
        renderer.draw_hands(annotated, hands, TrackingMode.FULL_HAND)
        self.assertTrue(np.any(annotated[60:160, 50:150] > 0))

        # 4. Uji resolusi path model dari folder palm_detection/model
        mdir = ensure_model_dir()
        self.assertTrue(os.path.isdir(mdir))
        dummy_model_file = os.path.join(mdir, "test_dummy_glove.pt")
        try:
            with open(dummy_model_file, "wb") as f:
                f.write(b"dummy")
            avail = list_available_yolo_models()
            self.assertIn("test_dummy_glove.pt", avail)
            resolved = resolve_yolo_model_path("test_dummy_glove.pt")
            self.assertEqual(os.path.normcase(resolved), os.path.normcase(dummy_model_file))
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)

    def test_overlay_renderer_all_modes(self) -> None:
        renderer = OverlayRenderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poly = np.array([[100, 100], [250, 100], [250, 250], [100, 250]], dtype=np.int32)

        # Setup mode rendering
        renderer.draw_setup_mode(
            frame=frame,
            rois=[poly],
            titik_klik=[(300, 300), (350, 300)],
            kursor=(350, 350),
            redo_roi_idx=None,
            get_roi_name=lambda i: f"Part {i + 1}",
        )
        self.assertGreater(int(frame.sum()), 0)

        # ROI card overlay untuk setiap state
        for state in (RoiState.ACTIVE, RoiState.HAND_IN, RoiState.DONE, RoiState.ERROR, RoiState.LOCKED):
            renderer.draw_roi_card_overlay(frame, poly, 0, state, state.value, "ROI #1")

        # Hand rendering untuk ketiga mode pelacakan
        hand = DummyHandDetector.make_hand_at(200, 200)
        for mode in (TrackingMode.FULL_HAND, TrackingMode.PALM_CENTER, TrackingMode.FINGERTIPS):
            renderer.draw_hands(frame, [hand], mode)

        renderer.draw_crop_boxes(frame, [(80, 80, 270, 270)])


class TestStepSequencer(unittest.TestCase):
    """Pengujian state machine urutan perakitan Poka-Yoke (StepSequencer)."""

    def test_full_poka_yoke_sequence_and_wrong_order(self) -> None:
        seq = StepSequencer(num_steps=2, debounce_masuk=2, debounce_keluar=2, ambang_rasio=0.08)
        self.assertEqual(seq.get_active_step(), 0)

        # Urutan salah: tangan masuk ke langkah 2 (index 1) saat langkah 1 (index 0) aktif
        self.assertFalse(seq.process_locked_step(1, True, debounce_err=2))
        self.assertTrue(seq.process_locked_step(1, True, debounce_err=2))
        seq.clear_error_counts()

        # Langkah 1: Masuk (debounce 2 frame)
        entered, _ = seq.process_step_entry(0, True)
        self.assertFalse(entered)
        entered, _ = seq.process_step_entry(0, True)
        self.assertTrue(entered)
        self.assertEqual(seq.statuses[0], StepStatus.HAND_IN)

        # Langkah 1: Keluar (debounce 2 frame) + diff gagal (< 0.08)
        should_eval, _ = seq.process_step_exit(0, False)
        self.assertFalse(should_eval)
        should_eval, _ = seq.process_step_exit(0, False)
        self.assertTrue(should_eval)

        ok, evt = seq.finalize_step_exit(0, 0.02, "Komponen A")
        self.assertFalse(ok)
        self.assertEqual(evt.level, LogLevel.WARN)
        self.assertEqual(seq.statuses[0], StepStatus.WAITING)

        # Langkah 1: Ulangi dan berhasil (diff >= 0.08)
        seq.process_step_entry(0, True)
        seq.process_step_entry(0, True)
        seq.process_step_exit(0, False)
        seq.process_step_exit(0, False)
        ok, evt = seq.finalize_step_exit(0, 0.25, "Komponen A")
        self.assertTrue(ok)
        self.assertEqual(evt.level, LogLevel.SUCCESS)
        self.assertEqual(seq.statuses[0], StepStatus.DONE)
        self.assertEqual(seq.get_active_step(), 1)

        # Langkah 2: Masuk & Keluar berhasil
        seq.process_step_entry(1, True)
        seq.process_step_entry(1, True)
        seq.process_step_exit(1, False)
        seq.process_step_exit(1, False)
        ok, _ = seq.finalize_step_exit(1, 0.30, "Komponen B")
        self.assertTrue(ok)
        self.assertTrue(seq.is_cycle_completed())

        # Penyelesaian siklus & auto-increment part
        t0 = 100.0
        is_done, expired, part_evt = seq.handle_cycle_completion(t0, cooldown_seconds=1.0)
        self.assertTrue(is_done)
        self.assertFalse(expired)
        self.assertIsNotNone(part_evt)
        self.assertEqual(seq.parts_completed, 1)

        _, expired_after, _ = seq.handle_cycle_completion(t0 + 1.5, cooldown_seconds=1.0)
        self.assertTrue(expired_after)

    def test_step_sequencer_without_pixel_diff_uses_debounce_only(self) -> None:
        """Memastikan mode tanpa perbandingan piksel menyelesaikan langkah murni berdasarkan debounce."""
        seq = StepSequencer(
            num_steps=2,
            debounce_masuk=2,
            debounce_keluar=3,
            ambang_rasio=0.50,
            use_pixel_diff=False,
        )
        # Debounce masuk (butuh 2 frame)
        self.assertFalse(seq.process_step_entry(0, True)[0])
        self.assertTrue(seq.process_step_entry(0, True)[0])
        self.assertEqual(seq.statuses[0], StepStatus.HAND_IN)

        # Debounce keluar (butuh 3 frame)
        self.assertFalse(seq.process_step_exit(0, False)[0])
        self.assertFalse(seq.process_step_exit(0, False)[0])
        self.assertTrue(seq.process_step_exit(0, False)[0])

        # Meskipun diff_ratio = None / 0.0 (< ambang_rasio 0.50), tetap selesai karena pixel diff nonaktif
        ok, evt = seq.finalize_step_exit(0, None, "Komponen A")
        self.assertTrue(ok)
        self.assertEqual(evt.level, LogLevel.SUCCESS)
        self.assertEqual(seq.statuses[0], StepStatus.DONE)
        self.assertEqual(seq.get_active_step(), 1)



class TestEndToEndVideoWorkerAndGUI(unittest.TestCase):
    """Pengujian End-to-End penuh untuk CameraWorker (input/output video & kontrol frame) dan GUI."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.in_video_path = os.path.join(self.tmpdir, "input_assembly.mp4")
        self.out_video_path = os.path.join(self.tmpdir, "output_annotated.mp4")
        self.rois, self.hand_positions = create_synthetic_assembly_video(self.in_video_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_worker_video_frame_selection_roi_editing_and_annotated_export(self) -> None:
        """Menguji alur lengkap: buka video -> pilih frame -> kunci ref -> edit ROI -> rekam video output."""
        cfg = AppConfig(
            jumlah_roi=2,
            debounce_masuk=2,
            debounce_keluar=2,
            ambang_piksel=25,
            ambang_rasio=0.05,
            ref_clean_frames=2,
            use_roi_crop=False,
            use_clahe=False,
            hand_backend=HandBackend.RTMLIB,
            hand_tracking_mode=TrackingMode.FULL_HAND,
        )
        worker = CameraWorker(cfg)

        # Mock pipeline.detect_hands agar menggunakan jadwal posisi tangan dari video sintetis
        def fake_detect_hands(frame, rois, config):
            f_idx = worker.video_current_frame
            pos = self.hand_positions[f_idx] if 0 <= f_idx < len(self.hand_positions) else None
            if pos is None:
                return [], [(0, 0, worker.frame_w, worker.frame_h)]
            return [DummyHandDetector.make_hand_at(pos[0], pos[1], spread=30)], [
                (0, 0, worker.frame_w, worker.frame_h)
            ]

        worker.pipeline.detect_hands = fake_detect_hands  # type: ignore[method-assign]

        # 1. Buka file video
        worker.open_video_file(self.in_video_path)
        worker._process_pending_commands(None)
        self.assertTrue(worker.is_video_mode)
        self.assertTrue(worker.video_paused)
        self.assertEqual(worker.video_total_frames, 20)
        self.assertEqual(worker.video_current_frame, 0)

        # 2. Uji menggambar ROI pertama secara interaktif via klik 4 titik sudut
        for px, py in self.rois[0]:
            worker.add_roi_point(px, py)
        with patch.object(cfg, "save"):
            worker._process_pending_commands(None)
        self.assertEqual(len(worker.rois), 1)

        # 3. Uji penentuan frame oleh pengguna: Seek ke Frame #2 (frame bersih)
        #    dan kunci sebagai referensi dengan opsi "add" (menambahkan ROI ke-2)
        worker.seek_video_frame(2)
        worker._process_pending_commands(None)
        self.assertEqual(worker.video_current_frame, 2)

        worker.lock_current_frame_as_ref(roi_mode="add")
        worker._process_pending_commands(None)
        self.assertIsNotNone(worker.referensi)
        self.assertTrue(worker._manual_ref_locked)
        self.assertEqual(len(worker.rois), 1)

        # Gambar ROI ke-2 di atas frame referensi tersebut
        for px, py in self.rois[1]:
            worker.add_roi_point(px, py)
        with patch.object(cfg, "save"):
            worker._process_pending_commands(None)
        self.assertEqual(len(worker.rois), 2)
        self.assertIsNotNone(worker.referensi)

        # Uji juga opsi "keep" (menggunakan ROI saat ini)
        worker.lock_current_frame_as_ref(roi_mode="keep")
        worker._process_pending_commands(None)
        self.assertEqual(len(worker.rois), 2)
        self.assertIsNotNone(worker.referensi)

        # 4. Uji navigasi maju/mundur 1 frame (step_video_frame)
        worker.step_video_frame(-1)
        worker._process_pending_commands(None)
        self.assertEqual(worker.video_current_frame, 1)

        # 5. Jalankan loop worker untuk memproses & merekam rentang Frame #0 s/d Frame #19 ke video output
        emitted_frames: List[Tuple[int, int]] = []
        emitted_telemetry: List[dict] = []
        emitted_logs: List[Tuple[str, str]] = []

        worker.frame_ready.connect(lambda qimg, w, h: emitted_frames.append((w, h)))
        worker.telemetry_updated.connect(lambda data: emitted_telemetry.append(data))
        worker.log_event.connect(lambda lvl, msg: emitted_logs.append((lvl, msg)))

        # Patch _open_camera di awal _run_worker_loop agar tidak menimpa video yang sudah dibuka
        with patch.object(worker, "_open_camera", return_value=True):
            worker.start()
            worker.start_video_recording(self.out_video_path, start_frame=0, end_frame=19)

            # Tunggu hingga perekaman selesai otomatis saat mencapai frame 19
            deadline = time.time() + 8.0
            while time.time() < deadline:
                _QT_APP.processEvents()
                if not worker.is_recording and worker.recorded_frames_count >= 20:
                    break
                time.sleep(0.02)

            worker.stop()
            _QT_APP.processEvents()

        # 6. Verifikasi hasil pemrosesan urutan langkah dan output video
        self.assertEqual(worker.recorded_frames_count, 20)
        self.assertFalse(worker.is_recording)
        self.assertEqual(worker.sequencer.get_steps_completed(), 2)
        self.assertEqual(worker.sequencer.parts_completed, 1)
        self.assertGreater(len(emitted_frames), 0)
        self.assertGreater(len(emitted_telemetry), 0)

        # Pastikan file video output benar-benar terbuat, valid, dan berdimensi 640x480 dengan 20 frame
        self.assertTrue(os.path.exists(self.out_video_path))
        cap_out = cv2.VideoCapture(self.out_video_path)
        self.assertTrue(cap_out.isOpened())
        out_frames_read = 0
        first_out_frame = None
        while True:
            ok, f_out = cap_out.read()
            if not ok or f_out is None:
                break
            if first_out_frame is None:
                first_out_frame = f_out
            out_frames_read += 1
        cap_out.release()

        self.assertEqual(out_frames_read, 20)
        self.assertIsNotNone(first_out_frame)
        self.assertEqual(first_out_frame.shape, (480, 640, 3))

    def test_worker_completes_without_pixel_diff_and_without_reference(self) -> None:
        """Memastikan CameraWorker menyelesaikan langkah tanpa referensi & tanpa perubahan piksel saat use_pixel_diff=False."""
        static_video_path = os.path.join(self.tmpdir, "static_no_diff.mp4")
        writer = cv2.VideoWriter(static_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
        static_frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        for _ in range(18):
            writer.write(static_frame)
        writer.release()

        cfg = AppConfig(
            jumlah_roi=2,
            debounce_masuk=2,
            debounce_keluar=2,
            use_pixel_diff=False,
            use_roi_crop=False,
            use_clahe=False,
            hand_backend=HandBackend.RTMLIB,
            hand_tracking_mode=TrackingMode.FULL_HAND,
        )
        worker = CameraWorker(cfg)

        # Jadwal tangan masuk ROI 1 (frame 1-3), keluar (frame 4-7), masuk ROI 2 (frame 8-10), keluar (frame 11-17)
        def fake_detect_hands(frame, rois, config):
            f_idx = worker.video_current_frame
            if 1 <= f_idx <= 3:
                pos = (160, 210)
            elif 8 <= f_idx <= 10:
                pos = (440, 210)
            else:
                pos = None
            if pos is None:
                return [], [(0, 0, worker.frame_w, worker.frame_h)]
            return [DummyHandDetector.make_hand_at(pos[0], pos[1], spread=30)], [
                (0, 0, worker.frame_w, worker.frame_h)
            ]

        worker.pipeline.detect_hands = fake_detect_hands  # type: ignore[method-assign]
        worker.open_video_file(static_video_path)
        worker.load_saved_rois(self.rois)
        worker._process_pending_commands(None)
        self.assertIsNone(worker.referensi)

        with patch.object(worker, "_open_camera", return_value=True):
            worker.start()
            worker.start_video_recording(self.out_video_path, start_frame=0, end_frame=17)
            deadline = time.time() + 8.0
            while time.time() < deadline:
                _QT_APP.processEvents()
                if not worker.is_recording and worker.recorded_frames_count >= 18:
                    break
                time.sleep(0.02)
            worker.stop()
            _QT_APP.processEvents()

        # Tetap tanpa referensi dan tanpa perubahan piksel, namun kedua langkah & part selesai via debounce
        self.assertIsNone(worker.referensi)
        self.assertEqual(worker.sequencer.get_steps_completed(), 2)
        self.assertEqual(worker.sequencer.parts_completed, 1)


    def test_gui_mainwindow_and_widgets_end_to_end(self) -> None:
        """Menguji integrasi komponen GUI: MainWindow, kontrol frame video, VideoCanvas, dan SettingsDialog."""
        isolated_cfg = AppConfig(jumlah_roi=2, saved_rois=self.rois)
        with (
            patch.object(AppConfig, "load", return_value=isolated_cfg),
            patch.object(CameraWorker, "_open_camera", return_value=False),
            patch.object(
                HandDetectionPipeline,
                "detect_hands",
                return_value=([], [(0, 0, 640, 480)]),
            ),
            patch.object(AppConfig, "save"),
        ):
            win = MainWindow()
            try:
                # Muat video sintetis ke worker melalui MainWindow
                win.worker.open_video_file(self.in_video_path)
                win.worker.load_saved_rois(self.rois)

                deadline = time.time() + 3.0
                while time.time() < deadline:
                    _QT_APP.processEvents()
                    if win._is_video_mode_ui and win._last_video_total == 20:
                        break
                    time.sleep(0.02)

                self.assertTrue(win._is_video_mode_ui)
                self.assertEqual(win.slider_frame.maximum(), 19)
                self.assertEqual(win.spin_end_frame.value(), 19)

                # Uji perubahan posisi frame oleh pengguna melalui spinbox & tombol rentang
                win.spin_current_frame.setValue(6)
                _QT_APP.processEvents()
                win._set_start_from_current()
                self.assertEqual(win.spin_start_frame.value(), 6)

                win.spin_current_frame.setValue(15)
                _QT_APP.processEvents()
                win._set_end_from_current()
                self.assertEqual(win.spin_end_frame.value(), 15)

                # Uji tombol kunci frame saat ini sebagai referensi dengan opsi gunakan ROI saat ini ("keep")
                win.action_lock_current_frame_ref(roi_mode="keep")
                t_wait = time.time() + 2.0
                while time.time() < t_wait and (
                    not win.worker._cmd_queue.empty() or win.worker.referensi is None
                ):
                    _QT_APP.processEvents()
                    time.sleep(0.02)
                self.assertIsNotNone(win.worker.referensi)
                self.assertEqual(len(win.worker.rois), 2)

                # Uji tombol kunci frame sebagai referensi dengan opsi tambah ROI ("add")
                win.action_lock_current_frame_ref(roi_mode="add")
                t_wait = time.time() + 2.0
                while time.time() < t_wait and (
                    not win.worker._cmd_queue.empty() or win.worker.config.jumlah_roi < 3
                ):
                    _QT_APP.processEvents()
                    time.sleep(0.02)
                self.assertEqual(win.worker.config.jumlah_roi, 3)

                # Uji interaksi mouse pada VideoCanvas
                win.canvas.resize(640, 480)
                dummy_qimg = win.canvas.pixmap().toImage() if win.canvas.pixmap() else None
                if dummy_qimg is not None:
                    win.canvas.update_frame(dummy_qimg, 640, 480)
                click_evt = QMouseEvent(
                    QEvent.MouseButtonPress,
                    QPointF(100.0, 100.0),
                    Qt.LeftButton,
                    Qt.LeftButton,
                    Qt.NoModifier,
                )
                win.canvas.mousePressEvent(click_evt)

                # Uji ROICardWidget perubahan nama komponen
                if win.roi_cards:
                    win.roi_cards[0].name_edit.setText("Komponen Uji 1")
                    win.roi_cards[0]._on_name_edited()
                    self.assertEqual(win.config.get_roi_name(0), "Komponen Uji 1")

                # Uji SettingsDialog inisialisasi & perubahan parameter (termasuk opsi YOLO26n-Pose)
                dlg = SettingsDialog(win.config, win.worker, win)
                dlg.spin_ambang_piksel.setValue(35)
                self.assertEqual(win.config.ambang_piksel, 35)
                dlg.combo_backend.setCurrentIndex(2)  # YOLO26n-Pose (Ultralytics)
                self.assertEqual(win.config.hand_backend, "yolo26n")
                self.assertTrue(dlg.edit_yolo_model.isEnabled())
                dlg.close()

            finally:
                win.shutdown_all()
                self.assertIsNone(win.worker.cap)
                self.assertEqual(len(_ACTIVE_CAPTURES), 0)
                win.deleteLater()
                _QT_APP.processEvents()

    def test_all_cameras_released_on_app_shutdown(self) -> None:
        """Memastikan seluruh instance kamera/VideoCapture dimatikan saat aplikasi/worker berhenti."""
        cfg = AppConfig(jumlah_roi=2)
        worker = CameraWorker(cfg)
        worker.open_video_file(self.in_video_path)
        worker._process_pending_commands(None)
        self.assertIsNotNone(worker.cap)
        self.assertTrue(worker.cap.isOpened())
        self.assertGreater(len(_ACTIVE_CAPTURES), 0)

        cap_ref = worker.cap
        worker.stop()

        self.assertIsNone(worker.cap)
        self.assertFalse(cap_ref.isOpened())
        self.assertEqual(len(worker._opened_captures), 0)
        self.assertEqual(len(_ACTIVE_CAPTURES), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
