"""Deteksi tangan dan optimasi sarung tangan industri (CLAHE & Pseudo-Skin).

Menyediakan:
    - preprocess_glove: penajaman siluet dan lipatan sendi sarung tangan
    - HandDetector: interface abstrak model deteksi tangan
    - RTMHandDetector: implementasi RTMPose via rtmlib & onnxruntime
    - MediaPipeHandDetector: implementasi Google MediaPipe Hands
    - HandDetectionPipeline: orkestrator cropping, pemrosesan per-region, dan deduplikasi
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Indeks keypoint standar 21 sendi tangan
UJUNG_JARI = [4, 8, 12, 16, 20]      # Thumb, Index, Middle, Ring, Pinky
SENDI_PUNGGUNG = [0, 5, 9, 13, 17]   # Pergelangan + 4 buku pangkal jari (MCP)


def preprocess_glove(
    frame_bgr: np.ndarray,
    clip_limit: float = 3.0,
    invert_dark: bool = False,
    lut_r: Optional[np.ndarray] = None,
    lut_g: Optional[np.ndarray] = None,
    lut_b: Optional[np.ndarray] = None,
    cached_clahe: Optional[cv2.CLAHE] = None,
) -> Tuple[np.ndarray, cv2.CLAHE]:
    """Mempertajam siluet dan lipatan jari sarung tangan dengan CLAHE dan Pseudo-Skin Tone.

    Menggantikan nama lama `preprocess_for_mediapipe` karena filter ini digunakan
    oleh kedua backend (RTMlib RTMPose maupun Google MediaPipe Hands).
    """
    if lut_r is None:
        lut_r = np.clip(np.arange(256) * 1.15, 0, 255).astype(np.uint8)
    if lut_g is None:
        lut_g = np.clip(np.arange(256) * 0.95, 0, 255).astype(np.uint8)
    if lut_b is None:
        lut_b = np.clip(np.arange(256) * 0.80, 0, 255).astype(np.uint8)

    # 1. Grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Invert untuk sarung tangan gelap / hitam
    if invert_dark:
        gray = cv2.bitwise_not(gray)

    # 3. CLAHE
    limit = max(0.1, float(clip_limit))
    clahe_obj = cached_clahe
    if clahe_obj is None:
        clahe_obj = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))

    enhanced = clahe_obj.apply(gray)

    # 4. Sintesis warna kulit semu (Pseudo-Skin) via Look-Up Table
    r = cv2.LUT(enhanced, lut_r)
    g = cv2.LUT(enhanced, lut_g)
    b = cv2.LUT(enhanced, lut_b)
    rgb_pseudo = cv2.merge([r, g, b])
    return rgb_pseudo, clahe_obj


class HandDetector(ABC):
    """Interface abstrak detektor tangan."""

    @abstractmethod
    def detect(self, crop_bgr: np.ndarray) -> List[dict]:
        """Mendeteksi tangan dalam satu crop citra BGR.

        Mengembalikan list dictionary dengan format:
            {
                "fingertips": List[(x, y)],
                "palm_center": (x, y),
                "dorsum_points": List[(x, y)],
                "all_points": List[(x, y)],
                "keypoints_21": List[Optional[(x, y)]],
                "hull": np.ndarray,
            }
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Melepas sumber daya AI dari memori."""
        pass


class RTMHandDetector(HandDetector):
    """Implementasi deteksi tangan menggunakan RTMlib (RTMPose Hand)."""

    def __init__(self, score_thr: float = 0.35, device: str = "cpu") -> None:
        from rtmlib import Hand as RTMHand

        self.score_thr = float(score_thr)
        self.tracker = RTMHand(device=device, backend="onnxruntime")

    def detect(self, crop_bgr: np.ndarray) -> List[dict]:
        results: List[dict] = []
        keypoints, scores = self.tracker(crop_bgr)
        thr = self.score_thr

        if len(keypoints) == 0:
            return results

        for kpts, scs in zip(keypoints, scores):
            valid_mask = [float(s) >= thr for s in scs]
            valid_count = sum(valid_mask)
            mean_score = float(np.mean(scs)) if len(scs) > 0 else 0.0

            # Validasi confidence score agregat
            if valid_count < 5 or mean_score < thr:
                continue

            raw_points = [(int(pt[0]), int(pt[1])) for pt in kpts]
            keypoints_21 = [
                raw_points[i] if valid_mask[i] else None for i in range(len(raw_points))
            ]

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
                raw_points[i] for i in range(len(raw_points)) if valid_mask[i]
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

            results.append({
                "fingertips": fingertips,
                "palm_center": (pc_x, pc_y),
                "dorsum_points": dorsum_points,
                "all_points": all_points,
                "keypoints_21": keypoints_21,
                "hull": hull,
            })

        return results

    def close(self) -> None:
        self.tracker = None


class MediaPipeHandDetector(HandDetector):
    """Implementasi deteksi tangan menggunakan Google MediaPipe Hands."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise RuntimeError("MediaPipe solutions.hands tidak tersedia.")

        self._mp_hands_cls = mp.solutions.hands.Hands
        self.tracker = self._mp_hands_cls(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, crop_bgr: np.ndarray) -> List[dict]:
        results: List[dict] = []
        if self.tracker is None:
            return results

        ch, cw = crop_bgr.shape[:2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        res = self.tracker.process(crop_rgb)

        if not res.multi_hand_landmarks:
            return results

        for hand in res.multi_hand_landmarks:
            all_points = [
                (int(lm.x * cw), int(lm.y * ch)) for lm in hand.landmark
            ]
            fingertips = [all_points[i] for i in UJUNG_JARI]
            dorsum_points = [all_points[i] for i in SENDI_PUNGGUNG]

            pc_x = int(sum(p[0] for p in dorsum_points) / len(dorsum_points))
            pc_y = int(sum(p[1] for p in dorsum_points) / len(dorsum_points))
            hull = cv2.convexHull(np.array(all_points, dtype=np.int32))

            results.append({
                "fingertips": fingertips,
                "palm_center": (pc_x, pc_y),
                "dorsum_points": dorsum_points,
                "all_points": all_points,
                "keypoints_21": all_points,
                "hull": hull,
            })

        return results

    def close(self) -> None:
        if self.tracker is not None:
            try:
                self.tracker.close()
            except Exception:
                pass
            self.tracker = None


class HandDetectionPipeline:
    """Mengelola cropping region ROI, filter sarung tangan, deteksi AI, dan pencegahan deteksi ganda."""

    def __init__(self) -> None:
        self.rtm_detector: Optional[RTMHandDetector] = None
        self.mp_detector: Optional[MediaPipeHandDetector] = None
        self.mp_region_detectors: Dict[int, MediaPipeHandDetector] = {}

        self.lut_r = np.clip(np.arange(256) * 1.15, 0, 255).astype(np.uint8)
        self.lut_g = np.clip(np.arange(256) * 0.95, 0, 255).astype(np.uint8)
        self.lut_b = np.clip(np.arange(256) * 0.80, 0, 255).astype(np.uint8)
        self._clahe: Optional[cv2.CLAHE] = None
        self._cached_clahe_limit = -1.0

    def get_rtm(self, score_thr: float) -> RTMHandDetector:
        if self.rtm_detector is None or abs(self.rtm_detector.score_thr - score_thr) > 1e-3:
            self.rtm_detector = RTMHandDetector(score_thr=score_thr)
        return self.rtm_detector

    def get_mp(self, region_idx: Optional[int] = None) -> MediaPipeHandDetector:
        if region_idx is None:
            if self.mp_detector is None:
                self.mp_detector = MediaPipeHandDetector()
            return self.mp_detector
        if region_idx not in self.mp_region_detectors:
            self.mp_region_detectors[region_idx] = MediaPipeHandDetector()
        return self.mp_region_detectors[region_idx]

    def close(self) -> None:
        if self.rtm_detector is not None:
            self.rtm_detector.close()
            self.rtm_detector = None
        if self.mp_detector is not None:
            self.mp_detector.close()
            self.mp_detector = None
        for det in self.mp_region_detectors.values():
            det.close()
        self.mp_region_detectors.clear()

    @staticmethod
    def is_duplicate_hand(cand: dict, existing_list: List[dict], dist_thr: float = 40.0) -> bool:
        cx, cy = cand["palm_center"]
        for h in existing_list:
            hx, hy = h["palm_center"]
            if ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5 < dist_thr:
                return True
        return False

    def compute_crop_regions(
        self,
        rois: List[np.ndarray],
        frame_w: int,
        frame_h: int,
        use_crop: bool,
        crop_mode: str,
        padding: int,
    ) -> List[Tuple[int, int, int, int]]:
        if not use_crop or len(rois) == 0:
            return [(0, 0, frame_w, frame_h)]

        pad = max(0, int(padding))
        regions: List[Tuple[int, int, int, int]] = []

        if crop_mode == "individual":
            for poly in rois:
                bx, by, bw, bh = cv2.boundingRect(poly)
                x1 = max(0, bx - pad)
                y1 = max(0, by - pad)
                x2 = min(frame_w, bx + bw + pad)
                y2 = min(frame_h, by + bh + pad)
                if x2 > x1 and y2 > y1:
                    regions.append((x1, y1, x2, y2))
        else:
            min_x = min(int(poly[:, 0].min()) for poly in rois)
            min_y = min(int(poly[:, 1].min()) for poly in rois)
            max_x = max(int(poly[:, 0].max()) for poly in rois)
            max_y = max(int(poly[:, 1].max()) for poly in rois)
            x1 = max(0, min_x - pad)
            y1 = max(0, min_y - pad)
            x2 = min(frame_w, max_x + pad)
            y2 = min(frame_h, max_y + pad)
            if x2 > x1 and y2 > y1:
                regions.append((x1, y1, x2, y2))

        return regions

    def detect_hands(
        self,
        frame: np.ndarray,
        rois: List[np.ndarray],
        config,
    ) -> Tuple[List[dict], List[Tuple[int, int, int, int]]]:
        """Menjalankan deteksi tangan lengkap dengan cropping dan CLAHE preview bersih."""
        fh, fw = frame.shape[:2]
        regions = self.compute_crop_regions(
            rois=rois,
            frame_w=fw,
            frame_h=fh,
            use_crop=config.use_roi_crop,
            crop_mode=config.roi_crop_mode,
            padding=config.roi_crop_padding,
        )

        all_hands: List[dict] = []
        preview_patches: List[Tuple[int, int, int, int, np.ndarray]] = []
        is_rtm = (config.hand_backend == "rtmlib")
        use_per_region = (
            config.use_roi_crop and config.roi_crop_mode == "individual" and len(regions) > 1
        )

        for reg_idx, (x1, y1, x2, y2) in enumerate(regions):
            crop = frame[y1:y2, x1:x2]
            cw, ch = x2 - x1, y2 - y1
            if cw <= 10 or ch <= 10:
                continue

            # Preprocessing sarung tangan jika diaktifkan
            if config.use_clahe:
                limit = float(config.clahe_clip_limit)
                if self._clahe is None or abs(self._cached_clahe_limit - limit) > 1e-3:
                    self._clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
                    self._cached_clahe_limit = limit

                crop_pseudo_rgb, self._clahe = preprocess_glove(
                    crop,
                    clip_limit=limit,
                    invert_dark=config.clahe_invert_dark,
                    lut_r=self.lut_r,
                    lut_g=self.lut_g,
                    lut_b=self.lut_b,
                    cached_clahe=self._clahe,
                )
                crop_feed_bgr = cv2.cvtColor(crop_pseudo_rgb, cv2.COLOR_RGB2BGR)
                if config.show_clahe_preview:
                    preview_patches.append((x1, y1, x2, y2, crop_feed_bgr))
            else:
                crop_feed_bgr = crop

            # Panggil detector aktif
            if is_rtm:
                det = self.get_rtm(float(config.rtm_score_thr))
                detected = det.detect(crop_feed_bgr)
            else:
                det = self.get_mp(region_idx=reg_idx if use_per_region else None)
                detected = det.detect(crop_feed_bgr)

            # Transformasi koordinat dari lokal crop ke frame global
            for h in detected:
                global_hand = {
                    "fingertips": [(x1 + pt[0], y1 + pt[1]) for pt in h["fingertips"]],
                    "palm_center": (x1 + h["palm_center"][0], y1 + h["palm_center"][1]),
                    "dorsum_points": [(x1 + pt[0], y1 + pt[1]) for pt in h["dorsum_points"]],
                    "all_points": [(x1 + pt[0], y1 + pt[1]) for pt in h["all_points"]],
                    "keypoints_21": [
                        (x1 + pt[0], y1 + pt[1]) if pt is not None else None
                        for pt in h["keypoints_21"]
                    ],
                    "hull": (
                        np.array(
                            [[(x1 + pt[0][0], y1 + pt[0][1])] for pt in h["hull"]],
                            dtype=np.int32,
                        )
                        if len(h["hull"]) > 0
                        else h["hull"]
                    ),
                }
                if not self.is_duplicate_hand(global_hand, all_hands):
                    all_hands.append(global_hand)

        # Salin preview CLAHE ke frame SETELAH semua crop selesai agar tidak mencemari crop lain
        for px1, py1, px2, py2, patch in preview_patches:
            frame[py1:py2, px1:px2] = patch

        return all_hands, regions
