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


class YOLOPoseHandDetector(HandDetector):
    """Implementasi deteksi tangan menggunakan Ultralytics YOLO Pose (mis. yolo26n-pose.pt).

    Mendukung secara otomatis:
        1. Model Hand Pose 21-keypoint (custom/fine-tuned 21 titik sendi jari).
        2. Model Pretrained COCO Human Pose 17-keypoint (yolo26n-pose.pt bawaan),
           dengan ekstraksi pergelangan tangan (wrist kpt 9 & 10) serta proyeksi telapak
           dari vektor lengan bawah (elbow -> wrist).
    """

    def __init__(
        self,
        model_path: str = "yolo26n-pose.pt",
        score_thr: float = 0.35,
        device: str = "cpu",
    ) -> None:
        self.model_path = str(model_path or "yolo26n-pose.pt").strip() or "yolo26n-pose.pt"
        self.score_thr = float(score_thr)
        self.device = device
        self.model = None

        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    @staticmethod
    def _build_hand_from_wrist(
        wx: float,
        wy: float,
        ex: Optional[float] = None,
        ey: Optional[float] = None,
    ) -> dict:
        """Membentuk representasi 21 titik & kontur telapak dari koordinat wrist (dan elbow opsional)."""
        if ex is not None and ey is not None and (abs(wx - ex) > 1.0 or abs(wy - ey) > 1.0):
            dx = float(wx - ex)
            dy = float(wy - ey)
            arm_len = float(np.hypot(dx, dy))
            pc_x = int(round(wx + 0.30 * dx))
            pc_y = int(round(wy + 0.30 * dy))
            radius = max(22, min(65, int(round(0.22 * arm_len))))
            base_angle = float(np.arctan2(dy, dx))
        else:
            pc_x = int(round(wx))
            pc_y = int(round(wy))
            radius = 32
            base_angle = -float(np.pi / 2.0)

        wrist_pt = (int(round(wx)), int(round(wy)))
        keypoints_21: List[Optional[Tuple[int, int]]] = [None] * 21
        keypoints_21[0] = wrist_pt

        # Sebarkan 5 jari di sekitar arah hadap lengan (base_angle)
        finger_offsets = [-0.65, -0.32, 0.0, 0.32, 0.65]
        finger_chains = [
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
            (17, 18, 19, 20),
        ]
        for f_angle_off, chain in zip(finger_offsets, finger_chains):
            ang = base_angle + f_angle_off
            for step_idx, k_idx in enumerate(chain, start=1):
                r_step = radius * (0.35 + 0.22 * step_idx)
                px = int(round(pc_x + r_step * np.cos(ang)))
                py = int(round(pc_y + r_step * np.sin(ang)))
                keypoints_21[k_idx] = (px, py)

        all_points = [pt for pt in keypoints_21 if pt is not None]
        fingertips = [keypoints_21[i] for i in UJUNG_JARI if keypoints_21[i] is not None]
        dorsum_points = [keypoints_21[i] for i in SENDI_PUNGGUNG if keypoints_21[i] is not None]
        hull = (
            cv2.convexHull(np.array(all_points, dtype=np.int32))
            if len(all_points) >= 3
            else np.empty((0, 1, 2), dtype=np.int32)
        )
        return {
            "fingertips": fingertips,
            "palm_center": (pc_x, pc_y),
            "dorsum_points": dorsum_points,
            "all_points": all_points,
            "keypoints_21": keypoints_21,
            "hull": hull,
        }

    @classmethod
    def parse_keypoints_array(
        cls,
        kpts_xy: np.ndarray,
        kpts_conf: Optional[np.ndarray],
        score_thr: float,
    ) -> List[dict]:
        """Mengonversi array keypoint hasil prediksi YOLO Pose menjadi list dictionary tangan."""
        results: List[dict] = []
        if kpts_xy is None or len(kpts_xy) == 0:
            return results

        thr = float(score_thr)
        num_instances = kpts_xy.shape[0]

        for inst_idx in range(num_instances):
            pts = kpts_xy[inst_idx]
            num_kpts = pts.shape[0]
            if num_kpts == 0:
                continue

            if kpts_conf is not None and inst_idx < len(kpts_conf) and kpts_conf[inst_idx] is not None:
                scs = [float(s) for s in kpts_conf[inst_idx]]
            else:
                scs = [1.0] * num_kpts

            # KASUS 1: Model Hand Pose 21-Keypoint
            if num_kpts >= 21:
                pts_21 = pts[:21]
                scs_21 = scs[:21]
                valid_mask = [
                    float(s) >= thr and (float(pt[0]) > 0.5 or float(pt[1]) > 0.5)
                    for pt, s in zip(pts_21, scs_21)
                ]
                valid_count = sum(valid_mask)
                mean_score = float(np.mean(scs_21)) if len(scs_21) > 0 else 0.0
                if valid_count < 5 or mean_score < thr:
                    continue

                raw_points = [(int(round(float(pt[0]))), int(round(float(pt[1])))) for pt in pts_21]
                keypoints_21 = [raw_points[i] if valid_mask[i] else None for i in range(21)]
                fingertips = [raw_points[i] for i in UJUNG_JARI if valid_mask[i]]
                dorsum_points = [raw_points[i] for i in SENDI_PUNGGUNG if valid_mask[i]]
                all_points = [raw_points[i] for i in range(21) if valid_mask[i]]
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

            # KASUS 2: Model Pretrained COCO 17-Keypoint (yolo26n-pose.pt)
            elif num_kpts == 17:
                # Pasangan (elbow_idx, wrist_idx): Kiri=(7, 9), Kanan=(8, 10)
                for elbow_idx, wrist_idx in ((7, 9), (8, 10)):
                    wx, wy = float(pts[wrist_idx][0]), float(pts[wrist_idx][1])
                    w_conf = scs[wrist_idx] if wrist_idx < len(scs) else 0.0
                    if w_conf < thr or (wx <= 1.0 and wy <= 1.0):
                        continue

                    ex, ey = float(pts[elbow_idx][0]), float(pts[elbow_idx][1])
                    e_conf = scs[elbow_idx] if elbow_idx < len(scs) else 0.0
                    if e_conf >= thr * 0.5 and (ex > 1.0 or ey > 1.0):
                        hand_dict = cls._build_hand_from_wrist(wx, wy, ex, ey)
                    else:
                        hand_dict = cls._build_hand_from_wrist(wx, wy, None, None)
                    results.append(hand_dict)

            # KASUS 3: Jumlah keypoint kustom lainnya
            else:
                valid_pts = [
                    (int(round(float(pt[0]))), int(round(float(pt[1]))))
                    for pt, s in zip(pts, scs)
                    if float(s) >= thr and (float(pt[0]) > 0.5 or float(pt[1]) > 0.5)
                ]
                if not valid_pts:
                    continue
                pc_x = int(sum(p[0] for p in valid_pts) / len(valid_pts))
                pc_y = int(sum(p[1] for p in valid_pts) / len(valid_pts))
                hand_dict = cls._build_hand_from_wrist(float(pc_x), float(pc_y), None, None)
                results.append(hand_dict)

        return results

    @classmethod
    def parse_boxes(
        cls,
        boxes_obj,
        names_dict: Optional[dict] = None,
        score_thr: float = 0.35,
    ) -> List[dict]:
        """Mengonversi bounding box YOLO Object Detection (mis. model deteksi sarung tangan) ke representasi tangan."""
        results: List[dict] = []
        try:
            xyxy_t = getattr(boxes_obj, "xyxy", None)
            conf_t = getattr(boxes_obj, "conf", None)
            cls_t = getattr(boxes_obj, "cls", None)
            if xyxy_t is None:
                return []

            xyxy_np = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)
            conf_np = (
                (conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t))
                if conf_t is not None
                else np.ones(len(xyxy_np), dtype=np.float32)
            )
            cls_np = (
                (cls_t.cpu().numpy() if hasattr(cls_t, "cpu") else np.asarray(cls_t))
                if cls_t is not None
                else np.zeros(len(xyxy_np), dtype=np.int32)
            )
        except Exception:
            return []

        thr = float(score_thr)
        for i, box in enumerate(xyxy_np):
            score = float(conf_np[i]) if i < len(conf_np) else 1.0
            if score < thr:
                continue

            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
            if x2 <= x1 or y2 <= y1:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Poligon sudut sarung tangan (format kontur cv2 (N, 1, 2))
            hull = cv2.convexHull(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32))

            # Titik-titik persebaran representatif sarung tangan
            all_pts = [
                (x1, y1), (x2, y1), (x2, y2), (x1, y2),
                (cx, cy),
                (x1 + bw // 4, cy), (x2 - bw // 4, cy),
                (cx, y1 + bh // 4), (cx, y2 - bh // 4),
            ]

            # 5 Titik 'fingertips' di tepi depan/atas kotak sarung tangan
            fingertips = [(x1 + int(bw * f), y1) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]

            # Titik punggung/pergelangan di bagian tengah dan bawah kotak
            dorsum = [(cx, cy), (x1, y2), (x2, y2), (cx, y2)]

            c_id = int(cls_np[i]) if i < len(cls_np) else 0
            label = "Sarung Tangan"
            if isinstance(names_dict, dict) and c_id in names_dict:
                label = str(names_dict[c_id])

            results.append({
                "fingertips": fingertips,
                "palm_center": (cx, cy),
                "dorsum_points": dorsum,
                "all_points": all_pts,
                "keypoints_21": all_pts + [(cx, cy)] * (21 - len(all_pts)),
                "hull": hull,
                "bbox": (x1, y1, x2, y2),
                "confidence": score,
                "label": label,
            })

        return results

    def detect(self, crop_bgr: np.ndarray) -> List[dict]:
        if self.model is None:
            return []
        preds = self.model(crop_bgr, verbose=False, conf=self.score_thr)
        if not preds:
            return []

        all_detected: List[dict] = []
        for res in preds:
            # 1. Coba deteksi Keypoints (YOLO Pose)
            kpts_obj = getattr(res, "keypoints", None)
            if kpts_obj is not None:
                xy = getattr(kpts_obj, "xy", None)
                if xy is not None:
                    xy_np = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
                    if len(xy_np) > 0 and xy_np.shape[-1] >= 2:
                        conf = getattr(kpts_obj, "conf", None)
                        conf_np = (
                            (conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf))
                            if conf is not None
                            else None
                        )
                        pose_results = self.parse_keypoints_array(xy_np, conf_np, self.score_thr)
                        if pose_results:
                            all_detected.extend(pose_results)
                            continue

            # 2. Coba deteksi Bounding Boxes (YOLO Object Detection - Sarung Tangan)
            boxes_obj = getattr(res, "boxes", None)
            if boxes_obj is not None and len(boxes_obj) > 0:
                box_results = self.parse_boxes(boxes_obj, getattr(res, "names", None), self.score_thr)
                all_detected.extend(box_results)

        return all_detected

    def close(self) -> None:
        self.model = None


class HandDetectionPipeline:
    """Mengelola cropping region ROI, filter sarung tangan, deteksi AI, dan pencegahan deteksi ganda."""

    def __init__(self) -> None:
        self.rtm_detector: Optional[RTMHandDetector] = None
        self.mp_detector: Optional[MediaPipeHandDetector] = None
        self.mp_region_detectors: Dict[int, MediaPipeHandDetector] = {}
        self.yolo_detector: Optional[YOLOPoseHandDetector] = None

        self.lut_r = np.clip(np.arange(256) * 1.15, 0, 255).astype(np.uint8)
        self.lut_g = np.clip(np.arange(256) * 0.95, 0, 255).astype(np.uint8)
        self.lut_b = np.clip(np.arange(256) * 0.80, 0, 255).astype(np.uint8)
        self._clahe: Optional[cv2.CLAHE] = None
        self._cached_clahe_limit = -1.0

    def get_rtm(self, score_thr: float) -> RTMHandDetector:
        if self.rtm_detector is None or abs(self.rtm_detector.score_thr - score_thr) > 1e-3:
            self.rtm_detector = RTMHandDetector(score_thr=score_thr)
        return self.rtm_detector

    def get_yolo(self, model_path: str, score_thr: float) -> YOLOPoseHandDetector:
        norm_path = str(model_path or "yolo26n-pose.pt").strip() or "yolo26n-pose.pt"
        if (
            self.yolo_detector is None
            or self.yolo_detector.model_path != norm_path
            or abs(self.yolo_detector.score_thr - score_thr) > 1e-3
        ):
            self.yolo_detector = YOLOPoseHandDetector(model_path=norm_path, score_thr=score_thr)
        return self.yolo_detector

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
        if self.yolo_detector is not None:
            self.yolo_detector.close()
            self.yolo_detector = None
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
        backend_str = str(config.hand_backend)
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
            if backend_str == "rtmlib":
                det = self.get_rtm(float(config.rtm_score_thr))
                detected = det.detect(crop_feed_bgr)
            elif backend_str in ("yolo26n", "yolo_detect", "yolo"):
                yolo_path = str(getattr(config, "yolo_model_path", "yolo26n-pose.pt"))
                det = self.get_yolo(yolo_path, float(config.rtm_score_thr))
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
                if "bbox" in h:
                    bx1, by1, bx2, by2 = h["bbox"]
                    global_hand["bbox"] = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
                if "confidence" in h:
                    global_hand["confidence"] = h["confidence"]
                if "label" in h:
                    global_hand["label"] = h["label"]

                if not self.is_duplicate_hand(global_hand, all_hands):
                    all_hands.append(global_hand)

        # Salin preview CLAHE ke frame SETELAH semua crop selesai agar tidak mencemari crop lain
        for px1, py1, px2, py2, patch in preview_patches:
            frame[py1:py2, px1:px2] = patch

        return all_hands, regions


# Kompatibilitas nama alias
YOLOHandDetector = YOLOPoseHandDetector

__all__ = [
    "UJUNG_JARI",
    "SENDI_PUNGGUNG",
    "preprocess_glove",
    "HandDetector",
    "RTMHandDetector",
    "MediaPipeHandDetector",
    "YOLOPoseHandDetector",
    "YOLOHandDetector",
    "HandDetectionPipeline",
]
