"""Komponen antarmuka grafis (PyQt6) untuk Palm Detection."""

from __future__ import annotations

import datetime
import sys
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
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

from palm_detection.config import AppConfig, CONFIG_FILE
from palm_detection.states import BannerState, LogLevel, RoiState
from palm_detection.worker import CameraWorker


class VideoCanvas(QLabel):
    """Widget penampil video interaktif yang memetakan klik/gerakan mouse."""

    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal()
    mouse_moved = pyqtSignal(int, int)
    vertex_dragged = pyqtSignal(int, int, int, int)
    vertex_drag_finished = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "background-color: #0d1117; border: 2px solid #30363d; border-radius: 8px;"
        )
        self._frame_w = 1280
        self._frame_h = 720
        self._pixmap_rect: Optional[Tuple[int, int, int, int]] = None
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
    """Kartu indikator status per-ROI."""

    redo_requested = pyqtSignal(int)
    name_changed = pyqtSignal(int, str)

    def __init__(self, idx: int, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.idx = idx
        self.config = config
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        self.name_edit = QLineEdit(self.config.get_roi_name(idx))
        self.name_edit.setPlaceholderText(f"Nama ROI #{idx + 1}")
        self.name_edit.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px; "
            "color: #58a6ff; font-weight: bold; font-size: 11px; padding: 2px 6px;"
        )
        self.name_edit.editingFinished.connect(self._on_name_edited)

        self.btn_redo = QPushButton("↺")
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

        self.update_card(RoiState.SETUP, "Menunggu dibuat", 0.0, "", 0.08)

    def _on_name_edited(self) -> None:
        new_name = self.name_edit.text().strip()
        if not new_name:
            new_name = f"ROI #{self.idx + 1}"
            self.name_edit.setText(new_name)
        self.config.set_roi_name(self.idx, new_name)
        self.config.save(CONFIG_FILE)
        self.name_changed.emit(self.idx, new_name)

    def update_card(
        self, state: RoiState | str, label: str, ratio: float, debounce: str, ambang_rasio: float
    ) -> None:
        state_key = state.value if isinstance(state, RoiState) else str(state)
        styles = {
            RoiState.DONE.value: ("#122d1f", "#2ea043", "#3fb950"),
            RoiState.ACTIVE.value: ("#2b230f", "#d29922", "#e3b341"),
            RoiState.HAND_IN.value: ("#311b10", "#db6d28", "#f0883e"),
            RoiState.ERROR.value: ("#361417", "#f85149", "#ff7b72"),
            RoiState.READY.value: ("#13233a", "#388bfd", "#58a6ff"),
            RoiState.WAIT_REF.value: ("#13233a", "#388bfd", "#58a6ff"),
            RoiState.SETUP.value: ("#161b22", "#484f58", "#8b949e"),
            RoiState.LOCKED.value: ("#161b22", "#30363d", "#6e7681"),
        }
        bg, border, badge_color = styles.get(state_key, styles[RoiState.LOCKED.value])
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
    """Dialog pengaturan konfigurasi sistem."""

    def __init__(self, config: AppConfig, worker: CameraWorker, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.worker = worker
        self.setWindowTitle("⚙ Pengaturan Sistem dan Kalibrasi Kamera")

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
        btn_save_cfg.setStyleSheet(
            "background-color: #1f5f35; border: 1px solid #3fb950; font-weight: bold; padding: 7px 16px;"
        )
        btn_save_cfg.clicked.connect(self.save_configuration)

        btn_load_cfg = QPushButton("📂 Muat dari config.json")
        btn_load_cfg.setStyleSheet(
            "background-color: #21262d; border: 1px solid #30363d; padding: 7px 16px;"
        )
        btn_load_cfg.clicked.connect(self.load_configuration)

        bottom_row.addWidget(btn_save_cfg)
        bottom_row.addWidget(btn_load_cfg)
        bottom_row.addStretch()

        btn_close = QPushButton("Tutup Pengaturan")
        btn_close.setStyleSheet(
            "background-color: #21262d; border: 1px solid #30363d; padding: 7px 22px; font-weight: bold;"
        )
        btn_close.clicked.connect(self.accept)
        bottom_row.addWidget(btn_close)
        layout.addLayout(bottom_row)

    def _create_engine_param_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

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

        prof_group = QGroupBox("Ekspor Kode OpenCV dan Info Konfigurasi")
        prof_layout = QVBoxLayout(prof_group)
        btn_copy_code = QPushButton("Cetak dan Salin Kode cap.set(...) ke Clipboard")
        btn_copy_code.clicked.connect(self.export_cap_set_snippet)
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
        self.config.hand_backend = self.combo_backend.currentData()

    def on_tracking_mode_changed(self) -> None:
        self.config.hand_tracking_mode = self.combo_tracking_mode.currentData()

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
        self.config.save(CONFIG_FILE)
        QMessageBox.information(
            self,
            "Konfigurasi Tersimpan",
            f"Konfigurasi dan {len(self.config.saved_rois)} ROI berhasil disimpan ke {CONFIG_FILE}.",
        )

    def load_configuration(self) -> None:
        loaded = AppConfig.load(CONFIG_FILE)
        self.config.update_from(loaded)

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
        QApplication.clipboard().setText(snippet)
        QMessageBox.information(
            self,
            "Kode Kalibrasi Disalin",
            f"Konfigurasi kamera telah disalin ke Clipboard:\n\n{snippet}",
        )


class MainWindow(QMainWindow):
    """Jendela utama aplikasi Palm Detection."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Palm Detection")
        self.resize(1380, 840)

        self.config = AppConfig.load()
        self.roi_cards: List[ROICardWidget] = []
        self.worker = CameraWorker(self.config, self)

        self._apply_dark_theme()
        self._build_ui()
        self._setup_shortcuts()

        # Koneksi sinyal worker -> GUI
        self.worker.frame_ready.connect(self.canvas.update_frame)
        self.worker.telemetry_updated.connect(self.on_telemetry_updated)
        self.worker.camera_info_ready.connect(self.on_camera_info)
        self.worker.log_event.connect(self.add_log_entry)

        # Koneksi interaksi kanvas -> worker
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

        # Header Bar
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

        # Konten Utama
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # Panel Kiri
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        self.banner_lbl = QLabel("Memuat kamera dan menyiapkan sistem...")
        self.banner_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner_lbl.setFixedHeight(44)
        self.banner_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self._set_banner_style(BannerState.SETUP)

        self.canvas = VideoCanvas()
        left_layout.addWidget(self.banner_lbl)
        left_layout.addWidget(self.canvas, stretch=1)

        # Panel Kanan
        right_panel = QWidget()
        right_panel.setFixedWidth(380)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Box Ringkasan Produksi
        prod_group = QGroupBox("Ringkasan Produksi")
        prod_layout = QVBoxLayout(prod_group)
        prod_layout.setContentsMargins(10, 8, 10, 8)
        prod_layout.setSpacing(8)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)

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

        # Status Urutan Langkah
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

        # Log Aktivitas
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

    def _set_banner_style(self, state: BannerState | str) -> None:
        key = state.value if isinstance(state, BannerState) else str(state)
        styles = {
            BannerState.SETUP.value: ("#13233a", "#388bfd", "#79c0ff"),
            BannerState.ACTIVE.value: ("#2b230f", "#d29922", "#ffd33d"),
            BannerState.ERROR.value: ("#491217", "#f85149", "#ffa198"),
            BannerState.DONE.value: ("#122d1f", "#2ea043", "#56d364"),
        }
        bg, border, text_color = styles.get(key, styles[BannerState.SETUP.value])
        self.banner_lbl.setStyleSheet(
            f"background-color: {bg}; border: 2px solid {border}; "
            f"color: {text_color}; border-radius: 6px; padding: 4px 12px;"
        )

    def open_settings_dialog(self) -> None:
        dlg = SettingsDialog(self.config, self.worker, self)
        dlg.exec()

    @pyqtSlot(str, str)
    def add_log_entry(self, level: str, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}] {message}")

        colors = {
            LogLevel.SUCCESS.value: "#3fb950",
            LogLevel.PART.value: "#58a6ff",
            LogLevel.WARN.value: "#f85149",
            LogLevel.INFO.value: "#8b949e",
        }
        item.setForeground(QColor(colors.get(level, "#c9d1d9")))
        if level in (LogLevel.PART.value, LogLevel.SUCCESS.value):
            item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    @pyqtSlot(dict)
    def on_telemetry_updated(self, data: dict) -> None:
        self.banner_lbl.setText(data["banner_text"])
        self._set_banner_style(data["banner_state"])

        coords = data.get("rois_coords", [])
        self.canvas.set_rois_coords(coords)
        self.config.saved_rois = coords

        parts = data.get("parts_completed", 0)
        steps = data.get("steps_completed", 0)
        total_steps = data.get("roi_total", self.config.jumlah_roi)

        self.part_count_lbl.setText(str(parts))
        self.step_count_lbl.setText(f"{steps} / {total_steps}")
        if total_steps > 0:
            self.progress_bar.setValue(int((steps / total_steps) * 100))

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
        self.status_bar.showMessage("Menutup aplikasi dan mematikan kamera...", 1000)
        self.worker.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.worker.stop)
    window.show()
    sys.exit(app.exec())
