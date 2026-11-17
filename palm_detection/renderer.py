"""Renderer khusus untuk menggambar visual overlay pada citra OpenCV.

Bertanggung jawab murni pada penggambaran grafis (OverlayRenderer), tanpa
mencampurkan logika state machine ataupun komputasi inferensi deteksi.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from palm_detection.detector import SENDI_PUNGGUNG, UJUNG_JARI
from palm_detection.states import RoiState, TrackingMode

WARNA_ROI_BGR = [
    (255, 128, 0),    # Biru-Oranye
    (0, 200, 255),    # Kuning-Emas
    (200, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (128, 255, 0),    # Hijau Muda
    (255, 0, 128),    # Ungu
]
WARNA_KUNCI_BGR = (130, 130, 130)

SAMBUNGAN_TANGAN = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ibu Jari
    (0, 5), (5, 6), (6, 7), (7, 8),        # Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),   # Jari Tengah
    (9, 13), (13, 14), (14, 15), (15, 16), # Jari Manis
    (13, 17), (17, 18), (18, 19), (19, 20),# Kelingking
    (0, 17),                               # Telapak
]


class OverlayRenderer:
    """Menggambar overlay grafis ROI, kerangka tangan, dan bounding box crop."""

    @staticmethod
    def isi_transparan(
        img: np.ndarray,
        poly: np.ndarray,
        warna: Tuple[int, int, int],
        alpha: float = 0.28,
    ) -> None:
        """Mengisi poligon dengan warna transparan."""
        if poly is None or len(poly) < 3:
            return
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], warna)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_setup_mode(
        self,
        frame: np.ndarray,
        rois: List[np.ndarray],
        titik_klik: List[Tuple[int, int]],
        kursor: Tuple[int, int],
        redo_roi_idx: Optional[int],
        get_roi_name: Callable[[int], str],
    ) -> None:
        """Menggambar poligon ROI dan garis panduan mouse saat pembuatan/pengeditan ROI."""
        for i, poly in enumerate(rois):
            warna = WARNA_ROI_BGR[i % len(WARNA_ROI_BGR)]
            self.isi_transparan(frame, poly, warna, alpha=0.18)
            cv2.polylines(frame, [poly], True, warna, 2)
            for pt in poly:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, warna, 1, cv2.LINE_AA)

            r_name = get_roi_name(i)
            lbl_text = f"{r_name} (Diedit...)" if redo_roi_idx == i else f"{r_name} (Siap)"
            cv2.putText(
                frame,
                lbl_text,
                (int(poly[:, 0].min()), max(int(poly[:, 1].min()) - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                warna,
                2,
                cv2.LINE_AA,
            )

        if titik_klik:
            for a, b in zip(titik_klik, titik_klik[1:] + [kursor]):
                cv2.line(frame, a, b, (0, 255, 255), 2)
            for p in titik_klik:
                cv2.circle(frame, p, 6, (0, 255, 255), -1)

    def draw_roi_card_overlay(
        self,
        frame: np.ndarray,
        poly: np.ndarray,
        idx: int,
        state: RoiState,
        label_text: str,
        roi_name: str,
    ) -> None:
        """Menggambar satu poligon ROI pada mode monitoring berdasarkan RoiState."""
        if state == RoiState.DONE:
            warna = (0, 200, 0)
            self.isi_transparan(frame, poly, warna, alpha=0.28)
            ketebalan = 3
        elif state == RoiState.HAND_IN:
            warna = (0, 165, 255)
            self.isi_transparan(frame, poly, warna, alpha=0.20)
            ketebalan = 2
        elif state == RoiState.ACTIVE:
            warna = WARNA_ROI_BGR[idx % len(WARNA_ROI_BGR)]
            ketebalan = 2
        elif state == RoiState.ERROR:
            warna = (0, 0, 255)
            self.isi_transparan(frame, poly, warna, alpha=0.30)
            ketebalan = 2
        else:
            warna = WARNA_KUNCI_BGR
            ketebalan = 2

        cv2.polylines(frame, [poly], True, warna, ketebalan)
        for pt in poly:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, warna, 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"{roi_name}: {label_text}",
            (int(poly[:, 0].min()), max(int(poly[:, 1].min()) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            warna,
            2,
            cv2.LINE_AA,
        )

    def draw_hands(
        self,
        frame: np.ndarray,
        hands: List[dict],
        tracking_mode: TrackingMode | str,
    ) -> None:
        """Menggambar visualisasi tangan (tulang, sendi, hull) sesuai mode tracking aktif."""
        mode_str = tracking_mode.value if isinstance(tracking_mode, TrackingMode) else tracking_mode

        for h in hands:
            # Jika objek dideteksi sebagai bounding box (YOLO Object Detection - Sarung Tangan)
            if "bbox" in h:
                x1, y1, x2, y2 = h["bbox"]
                conf = float(h.get("confidence", 0.0))
                label = str(h.get("label", "Sarung Tangan"))

                # 1. Gambar kotak & isi transparan
                if "hull" in h and len(h["hull"]) >= 3:
                    cv2.polylines(frame, [h["hull"]], True, (0, 255, 170), 2, cv2.LINE_AA)
                    self.isi_transparan(frame, h["hull"], (0, 255, 170), alpha=0.15)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 170), 2, cv2.LINE_AA)

                # 2. Titik pusat telapak / sarung tangan
                pc = h["palm_center"]
                cv2.circle(frame, pc, 6, (0, 255, 255), -1)
                cv2.circle(frame, pc, 8, (0, 0, 0), 2)

                # 3. Label tag
                tag_text = f"{label} {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(th + 8, y1)), (0, 255, 170), -1)
                cv2.putText(
                    frame,
                    tag_text,
                    (x1 + 3, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                continue

            if mode_str == TrackingMode.FULL_HAND:
                # 1. Selubung convex hull
                if "hull" in h and len(h["hull"]) >= 3:
                    cv2.polylines(frame, [h["hull"]], True, (0, 255, 170), 1, cv2.LINE_AA)
                    self.isi_transparan(frame, h["hull"], (0, 255, 170), alpha=0.12)

                # 2. Garis kerangka sendi
                kpts_21 = h.get("keypoints_21", h["all_points"])
                for i_a, i_b in SAMBUNGAN_TANGAN:
                    if (
                        i_a < len(kpts_21)
                        and i_b < len(kpts_21)
                        and kpts_21[i_a] is not None
                        and kpts_21[i_b] is not None
                    ):
                        cv2.line(frame, kpts_21[i_a], kpts_21[i_b], (0, 230, 255), 2, cv2.LINE_AA)

                # 3. Titik-titik sendi
                for idx_pt, p in enumerate(kpts_21):
                    if p is None:
                        continue
                    if idx_pt in UJUNG_JARI:
                        cv2.circle(frame, p, 5, (0, 255, 255), -1)
                        cv2.circle(frame, p, 6, (0, 180, 220), 1)
                    elif idx_pt == 0:
                        cv2.circle(frame, p, 6, (255, 120, 0), -1)
                    else:
                        cv2.circle(frame, p, 3, (0, 255, 128), -1)

                # 4. Titik pusat telapak / dorsum
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

            elif mode_str == TrackingMode.PALM_CENTER:
                if h["dorsum_points"]:
                    for p in h["dorsum_points"]:
                        cv2.circle(frame, p, 4, (0, 200, 255), -1)
                    wrist = h["dorsum_points"][0]
                    for kn in h["dorsum_points"][1:]:
                        cv2.line(frame, wrist, kn, (0, 200, 255), 1, cv2.LINE_AA)

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

            else:  # FINGERTIPS
                for p in h["fingertips"]:
                    cv2.circle(frame, p, 6, (255, 255, 0), -1)
                    cv2.circle(frame, p, 6, (0, 0, 0), 1)

    def draw_crop_boxes(
        self, frame: np.ndarray, crop_boxes: List[Tuple[int, int, int, int]]
    ) -> None:
        """Menggambar garis kotak crop area ROI."""
        for (cx1, cy1, cx2, cy2) in crop_boxes:
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 128), 1, cv2.LINE_AA)
