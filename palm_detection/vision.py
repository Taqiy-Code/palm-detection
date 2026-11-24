"""Utilitas visi komputer untuk pengolahan citra ROI, diff mask, dan geometri poligon."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def siapkan_gray(img: np.ndarray) -> np.ndarray:
    """Mengonversi BGR ke grayscale dengan Gaussian Blur 7x7."""
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 0)


def normalize_and_validate_roi(pts, min_area: float = 100.0) -> Optional[np.ndarray]:
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


def hull_roi_intersection_area(poly: np.ndarray, hull: np.ndarray) -> float:
    """Menghitung luas irisan absolut (dalam piksel^2) antara convex hull tangan dan poligon ROI konveks."""
    if poly is None or hull is None or len(poly) < 3 or len(hull) < 3:
        return 0.0
    poly_f = poly.reshape(-1, 2).astype(np.float32)
    hull_f = hull.reshape(-1, 2).astype(np.float32)
    if float(cv2.contourArea(poly_f)) <= 1.0 or float(cv2.contourArea(hull_f)) <= 1.0:
        return 0.0
    try:
        inter_area, _ = cv2.intersectConvexConvex(poly_f, hull_f)
    except cv2.error:
        return 0.0
    return float(max(0.0, inter_area))


def hull_roi_overlap_ratio(poly: np.ndarray, hull: np.ndarray) -> float:
    """Menghitung rasio luas irisan convex hull tangan dan poligon ROI konveks terhadap luas ROI."""
    if poly is None or hull is None or len(poly) < 3 or len(hull) < 3:
        return 0.0
    poly_f = poly.reshape(-1, 2).astype(np.float32)
    roi_area = float(cv2.contourArea(poly_f))
    if roi_area <= 1.0:
        return 0.0
    inter_area = hull_roi_intersection_area(poly, hull)
    return float(inter_area / roi_area)


def assign_primary_roi_per_hand(
    rois: List[np.ndarray],
    hands: List[dict],
    tracking_mode: str = "full_hand",
    overlap_thr: float = 0.15,
    active_step_idx: Optional[int] = None,
) -> Tuple[List[bool], List[bool]]:
    """Menentukan (tangan_masuk_per_roi, ada_tangan_model_per_roi) dengan Winner-Takes-All per tangan.

    Setiap objek tangan `h` hanya dapat memicu `tangan_masuk_per_roi = True` pada maksimal SATU ROI
    pemenang (Primary ROI), sehingga 1 tangan yang berada di ROI 1 namun tepinya menyenggol ROI 2
    tidak memicu `URUTAN SALAH` palsu, sementara 2 tangan berbeda di ROI 1 & ROI 2 tetap terdeteksi.

    Prioritas pemilihan Primary ROI untuk 1 tangan:
        1. Titik pusat tangan (`palm_center`) berada di dalam poligon ROI,
        2. Luas irisan absolut (`inter_area`) terbesar (bukan dibagi luas ROI),
        3. Jumlah titik tangan yang berada di dalam ROI,
        4. Tie-breaker: langkah aktif (`active_step_idx`).
    """
    n = len(rois)
    entered = [False] * n
    present = [False] * n
    if n == 0 or not hands:
        return entered, present

    mode_str = str(getattr(tracking_mode, "value", tracking_mode))
    roi_areas = [
        float(cv2.contourArea(poly.reshape(-1, 2).astype(np.float32))) if poly is not None and len(poly) >= 3 else 0.0
        for poly in rois
    ]

    for h in hands:
        best_idx: Optional[int] = None
        best_score: Optional[Tuple[int, float, int, int]] = None

        pc = h.get("palm_center")
        pc_pt = (float(pc[0]), float(pc[1])) if pc is not None else None
        hull = h.get("hull")

        for idx, poly in enumerate(rois):
            if poly is None or len(poly) < 3:
                continue

            pc_in = (cv2.pointPolygonTest(poly, pc_pt, False) >= 0) if pc_pt is not None else False
            qualifies_entry = False
            inter_area = 0.0
            pts_in_roi = 0

            if mode_str == "full_hand":
                inter_area = hull_roi_intersection_area(poly, hull)
                roi_area = roi_areas[idx]
                ov_ratio = (inter_area / roi_area) if roi_area > 1.0 else 0.0
                pts_in_roi = sum(
                    1
                    for p in h.get("all_points", ())
                    if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
                )
                if ov_ratio >= overlap_thr * 0.5 or (pc_in and pts_in_roi >= 3):
                    present[idx] = True
                if ov_ratio >= overlap_thr or (pc_in and pts_in_roi >= 5):
                    qualifies_entry = True

            elif mode_str == "palm_center":
                pts_in_roi = sum(
                    1
                    for p in h.get("dorsum_points", ())
                    if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
                )
                if pc_in or pts_in_roi >= 2:
                    present[idx] = True
                if pc_in:
                    inter_area = hull_roi_intersection_area(poly, hull)
                    qualifies_entry = True

            else:  # fingertips
                pts_in_roi = sum(
                    1
                    for p in h.get("fingertips", ())
                    if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
                )
                if pts_in_roi >= 2:
                    present[idx] = True
                if pts_in_roi == 5:
                    inter_area = hull_roi_intersection_area(poly, hull)
                    qualifies_entry = True

            if qualifies_entry:
                score = (
                    1 if pc_in else 0,
                    inter_area,
                    pts_in_roi,
                    1 if idx == active_step_idx else 0,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx is not None:
            entered[best_idx] = True

    return entered, present


def normalize_brightness(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Menormalisasi perubahan pencahayaan global sebelum frame differencing."""
    delta = float(np.mean(ref)) - float(np.mean(img))
    if abs(delta) < 0.5:
        return img
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def clean_diff_mask(
    img_a: np.ndarray,
    img_b: np.ndarray,
    roi_mask: Optional[np.ndarray],
    pixel_thr: int,
    morph_kernel: np.ndarray,
) -> np.ndarray:
    """Menghitung mask piksel berubah dengan normalisasi brightness dan morphological opening."""
    norm_a = normalize_brightness(img_a, img_b)
    diff = cv2.absdiff(norm_a, img_b)
    if roi_mask is not None:
        binary = ((diff > pixel_thr) & (roi_mask > 0)).astype(np.uint8) * 255
    else:
        binary = (diff > pixel_thr).astype(np.uint8) * 255
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel)


def rasio_perubahan(
    gray_eval: np.ndarray,
    referensi: Optional[np.ndarray],
    mask: np.ndarray,
    pixel_thr: int,
    morph_kernel: np.ndarray,
) -> float:
    """Menghitung persentase luas piksel yang berubah terhadap total luas ROI."""
    if referensi is None or mask is None:
        return 0.0
    cleaned = clean_diff_mask(gray_eval, referensi, mask, pixel_thr, morph_kernel)
    berubah = np.count_nonzero(cleaned)
    luas = max(np.count_nonzero(mask), 1)
    return float(berubah / luas)


def rasio_gerak_roi(
    gray_now: np.ndarray,
    prev_gray: Optional[np.ndarray],
    mask: np.ndarray,
    motion_thr: int,
    morph_kernel: np.ndarray,
) -> float:
    """Mendeteksi pergerakan objek di dalam ROI antar frame berturut-turut."""
    if prev_gray is None or prev_gray.shape != gray_now.shape or mask is None:
        return 0.0
    cleaned = clean_diff_mask(gray_now, prev_gray, mask, motion_thr, morph_kernel)
    bergerak = np.count_nonzero(cleaned)
    luas = max(np.count_nonzero(mask), 1)
    return float(bergerak / luas)
