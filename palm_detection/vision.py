"""Utilitas visi komputer untuk pengolahan citra ROI, diff mask, dan geometri poligon."""

from __future__ import annotations

from typing import List, Optional

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
