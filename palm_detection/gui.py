"""Komponen antarmuka grafis (PyQt5) untuk Palm Detection."""

from __future__ import annotations

import datetime
import os
import sys
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QMouseEvent, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
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
from palm_detection.worker import CameraWorker, release_all_cameras


class VideoCanvas(QLabel):
    """Widget penampil video interaktif yang memetakan klik/gerakan mouse."""

    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal()
    mouse_moved = pyqtSignal(int, int)
    vertex_dragged = pyqtSignal(int, int, int, int)
    vertex_drag_finished = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
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
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
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
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            mapped = self._map_to_frame(int(pos.x()), int(pos.y()))
            if mapped:
                fx, fy = mapped
                target = self._find_vertex_near(fx, fy)
                if target is not None:
                    self._drag_target = target
                    self.setCursor(Qt.ClosedHandCursor)
                    return
                self.left_clicked.emit(fx, fy)
        elif event.button() == Qt.RightButton:
            if self._drag_target is not None:
                self._drag_target = None
                self.setCursor(Qt.ArrowCursor)
            else:
                self.right_clicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        mapped = self._map_to_frame(int(pos.x()), int(pos.y()))
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
                    self.setCursor(Qt.PointingHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            self.mouse_moved.emit(fx, fy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_target is not None:
            self._drag_target = None
            self.setCursor(Qt.ArrowCursor)
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
        self.setFrameShape(QFrame.StyledPanel)

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

        self.btn_redo = QPushButton("Reset")
        self.btn_redo.setFixedSize(45, 22)
        self.btn_redo.setStyleSheet(
            "background-color: #21262d; border: 1px solid #30363d; border-radius: 4px; "
            "color: #e6edf3; font-weight: normal; font-size: 11px; padding: 0px;"
        )
        self.btn_redo.clicked.connect(lambda: self.redo_requested.emit(self.idx))

        self.badge_lbl = QLabel("SETUP")
        self.badge_lbl.setAlignment(Qt.AlignCenter)
        self.badge_lbl.setFixedHeight(22)
        self.badge_lbl.setMinimumWidth(105)

        top_row.addWidget(self.name_edit, stretch=1)
        top_row.addWidget(self.btn_redo)
        top_row.addWidget(self.badge_lbl)

        bottom_row = QHBoxLayout()
        self.ratio_lbl = QLabel("Diff: 0.0%")
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
        self.ratio_lbl.setText(f"Diff: {ratio * 100:.1f}% (Min {ambang_rasio * 100:.1f}%)")
        self.debounce_lbl.setText(debounce)


class SettingsDialog(QDialog):
    """Dialog pengaturan konfigurasi sistem."""

    def __init__(self, config: AppConfig, worker: CameraWorker, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.worker = worker
        self.setWindowTitle("Pengaturan Sistem")

        screen = (
            getattr(self, "screen", lambda: None)()
            or (getattr(parent, "screen", lambda: None)() if parent else None)
            or QApplication.primaryScreen()
        )
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
        self.tabs.addTab(self._create_engine_param_tab(), "Deteksi ROI")
        self.tabs.addTab(self._create_glove_crop_tab(), "Filter dan Crop")
        self.tabs.addTab(self._create_camera_tab(), "Kamera")
        layout.addWidget(self.tabs, stretch=1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        btn_save_cfg = QPushButton("Simpan Konfigurasi")
        btn_save_cfg.setStyleSheet(
            "background-color: #1f5f35; border: 1px solid #3fb950; font-weight: bold; padding: 7px 16px;"
        )
        btn_save_cfg.clicked.connect(self.save_configuration)

        btn_load_cfg = QPushButton("Muat Konfigurasi")
        btn_load_cfg.setStyleSheet(
            "background-color: #21262d; border: 1px solid #30363d; padding: 7px 16px;"
        )
        btn_load_cfg.clicked.connect(self.load_configuration)

        bottom_row.addWidget(btn_save_cfg)
        bottom_row.addWidget(btn_load_cfg)
        bottom_row.addStretch()

        btn_close = QPushButton("Tutup")
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

        engine_group = QGroupBox("Engine Deteksi")
        engine_layout = QGridLayout(engine_group)
        engine_layout.setVerticalSpacing(8)

        engine_layout.addWidget(QLabel("Backend Deteksi:"), 0, 0)
        self.combo_backend = QComboBox()
        self.combo_backend.addItem("RTMlib (RTMPose)", "rtmlib")
        self.combo_backend.addItem("MediaPipe Hands", "mediapipe")
        self.combo_backend.addItem("YOLO26n-Pose (Ultralytics)", "yolo26n")
        backend_idx_map = {"rtmlib": 0, "mediapipe": 1, "yolo26n": 2}
        self.combo_backend.setCurrentIndex(backend_idx_map.get(self.config.hand_backend, 0))
        self.combo_backend.currentIndexChanged.connect(self.on_backend_changed)
        engine_layout.addWidget(self.combo_backend, 0, 1)

        engine_layout.addWidget(QLabel("Model YOLO Pose:"), 1, 0)
        yolo_row = QHBoxLayout()
        yolo_row.setContentsMargins(0, 0, 0, 0)
        yolo_row.setSpacing(6)
        self.edit_yolo_model = QLineEdit(self.config.yolo_model_path)
        self.edit_yolo_model.setPlaceholderText("yolo26n-pose.pt atau path model .pt/.onnx")
        self.edit_yolo_model.editingFinished.connect(
            lambda: setattr(
                self.config,
                "yolo_model_path",
                self.edit_yolo_model.text().strip() or "yolo26n-pose.pt",
            )
        )
        self.btn_browse_yolo = QPushButton("Pilih...")
        self.btn_browse_yolo.setFixedWidth(68)
        self.btn_browse_yolo.clicked.connect(self.on_browse_yolo_model)
        yolo_row.addWidget(self.edit_yolo_model, 1)
        yolo_row.addWidget(self.btn_browse_yolo)
        engine_layout.addLayout(yolo_row, 1, 1)

        engine_layout.addWidget(QLabel("Threshold Skor Sendi:"), 2, 0)
        self.spin_rtm_score = QDoubleSpinBox()
        self.spin_rtm_score.setRange(0.10, 0.95)
        self.spin_rtm_score.setSingleStep(0.05)
        self.spin_rtm_score.setValue(self.config.rtm_score_thr)
        self.spin_rtm_score.valueChanged.connect(
            lambda v: setattr(self.config, "rtm_score_thr", float(v))
        )
        engine_layout.addWidget(self.spin_rtm_score, 2, 1)

        engine_layout.addWidget(QLabel("Mode Pelacakan:"), 3, 0)
        self.combo_tracking_mode = QComboBox()
        self.combo_tracking_mode.addItem("Seluruh Tangan (21 Titik)", "full_hand")
        self.combo_tracking_mode.addItem("Pusat Tangan", "palm_center")
        self.combo_tracking_mode.addItem("5 Ujung Jari", "fingertips")
        mode_idx_map = {"full_hand": 0, "palm_center": 1, "fingertips": 2}
        self.combo_tracking_mode.setCurrentIndex(mode_idx_map.get(self.config.hand_tracking_mode, 0))
        self.combo_tracking_mode.currentIndexChanged.connect(self.on_tracking_mode_changed)
        engine_layout.addWidget(self.combo_tracking_mode, 3, 1)

        is_yolo = self.config.hand_backend == "yolo26n"
        self.edit_yolo_model.setEnabled(is_yolo)
        self.btn_browse_yolo.setEnabled(is_yolo)

        layout.addWidget(engine_group)

        param_group = QGroupBox("Parameter Deteksi ROI")
        grid = QGridLayout(param_group)
        grid.setVerticalSpacing(10)

        grid.addWidget(QLabel("Jumlah ROI:"), 0, 0)
        self.spin_jumlah_roi = QSpinBox()
        self.spin_jumlah_roi.setRange(1, 6)
        self.spin_jumlah_roi.setValue(self.config.jumlah_roi)
        self.spin_jumlah_roi.valueChanged.connect(self.on_jumlah_roi_changed)
        grid.addWidget(self.spin_jumlah_roi, 0, 1)

        grid.addWidget(QLabel("Ambang Piksel:"), 1, 0)
        self.spin_ambang_piksel = QSpinBox()
        self.spin_ambang_piksel.setRange(5, 200)
        self.spin_ambang_piksel.setValue(self.config.ambang_piksel)
        self.spin_ambang_piksel.setSuffix(" / 255")
        self.spin_ambang_piksel.valueChanged.connect(
            lambda v: setattr(self.config, "ambang_piksel", v)
        )
        grid.addWidget(self.spin_ambang_piksel, 1, 1)

        grid.addWidget(QLabel("Ambang Rasio Area:"), 2, 0)
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

        grid.addWidget(QLabel("Debounce Masuk:"), 3, 0)
        self.spin_debounce_masuk = QSpinBox()
        self.spin_debounce_masuk.setRange(1, 60)
        self.spin_debounce_masuk.setSuffix(" frame")
        self.spin_debounce_masuk.setValue(self.config.debounce_masuk)
        self.spin_debounce_masuk.valueChanged.connect(
            lambda v: setattr(self.config, "debounce_masuk", v)
        )
        grid.addWidget(self.spin_debounce_masuk, 3, 1)

        grid.addWidget(QLabel("Debounce Keluar:"), 4, 0)
        self.spin_debounce_keluar = QSpinBox()
        self.spin_debounce_keluar.setRange(1, 90)
        self.spin_debounce_keluar.setSuffix(" frame")
        self.spin_debounce_keluar.setValue(self.config.debounce_keluar)
        self.spin_debounce_keluar.valueChanged.connect(
            lambda v: setattr(self.config, "debounce_keluar", v)
        )
        grid.addWidget(self.spin_debounce_keluar, 4, 1)

        self.chk_debug_diff = QCheckBox("Tampilkan Highlight Piksel Berubah")
        self.chk_debug_diff.setChecked(self.config.debug_diff)
        self.chk_debug_diff.toggled.connect(lambda c: setattr(self.config, "debug_diff", c))
        grid.addWidget(self.chk_debug_diff, 5, 0, 1, 2)

        layout.addWidget(param_group)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(tab)
        return scroll

    def _create_glove_crop_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        glove_group = QGroupBox("Filter Sarung Tangan (CLAHE)")
        glove_layout = QGridLayout(glove_group)
        glove_layout.setVerticalSpacing(8)

        self.chk_use_clahe = QCheckBox("Aktifkan Filter Sarung Tangan")
        self.chk_use_clahe.setChecked(self.config.use_clahe)
        self.chk_use_clahe.toggled.connect(self.on_use_clahe_toggled)
        glove_layout.addWidget(self.chk_use_clahe, 0, 0, 1, 2)

        self.chk_clahe_invert = QCheckBox("Invert Warna (Sarung Tangan Gelap)")
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

        self.chk_clahe_preview = QCheckBox("Tampilkan Preview Citra")
        self.chk_clahe_preview.setChecked(self.config.show_clahe_preview)
        self.chk_clahe_preview.toggled.connect(lambda c: setattr(self.config, "show_clahe_preview", c))
        glove_layout.addWidget(self.chk_clahe_preview, 3, 0, 1, 2)
        layout.addWidget(glove_group)

        crop_group = QGroupBox("ROI Cropping")
        crop_layout = QGridLayout(crop_group)
        crop_layout.setVerticalSpacing(8)

        self.chk_use_roi_crop = QCheckBox("Aktifkan ROI Cropping")
        self.chk_use_roi_crop.setChecked(self.config.use_roi_crop)
        self.chk_use_roi_crop.toggled.connect(self.on_use_roi_crop_toggled)
        crop_layout.addWidget(self.chk_use_roi_crop, 0, 0, 1, 2)

        crop_layout.addWidget(QLabel("Mode Pemotongan:"), 1, 0)
        self.combo_crop_mode = QComboBox()
        self.combo_crop_mode.addItem("Gabungan Semua ROI", "union")
        self.combo_crop_mode.addItem("Per Kotak ROI", "individual")
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

        self.chk_show_crop_box = QCheckBox("Tampilkan Garis Pembatas Crop")
        self.chk_show_crop_box.setChecked(self.config.show_crop_box)
        self.chk_show_crop_box.toggled.connect(
            lambda c: setattr(self.config, "show_crop_box", c)
        )
        crop_layout.addWidget(self.chk_show_crop_box, 3, 0, 1, 2)
        layout.addWidget(crop_group)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(tab)
        return scroll

    def _create_camera_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        cam_group = QGroupBox("Perangkat Kamera")
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

        auto_group = QGroupBox("Kontrol Otomatis Kamera")
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

        slider_group = QGroupBox("Kalibrasi Kamera")
        s_grid = QGridLayout(slider_group)
        s_grid.setVerticalSpacing(8)

        s_grid.addWidget(QLabel("Focus:"), 0, 0)
        self.slider_focus, self.spin_focus = self._make_slider_spin(
            0, 255, self.config.focus, lambda v: self.worker.update_camera_hardware("focus", v)
        )
        self.slider_focus.setEnabled(not self.config.auto_focus)
        self.spin_focus.setEnabled(not self.config.auto_focus)
        s_grid.addWidget(self.slider_focus, 0, 1)
        s_grid.addWidget(self.spin_focus, 0, 2)

        s_grid.addWidget(QLabel("Exposure:"), 1, 0)
        self.slider_exp, self.spin_exp = self._make_slider_spin(
            -13, 0, self.config.exposure, lambda v: self.worker.update_camera_hardware("exposure", v)
        )
        self.slider_exp.setEnabled(not self.config.auto_exposure)
        self.spin_exp.setEnabled(not self.config.auto_exposure)
        s_grid.addWidget(self.slider_exp, 1, 1)
        s_grid.addWidget(self.spin_exp, 1, 2)

        s_grid.addWidget(QLabel("White Balance:"), 2, 0)
        self.slider_wb, self.spin_wb = self._make_slider_spin(
            2800, 6500, self.config.wb_temp, lambda v: self.worker.update_camera_hardware("wb_temp", v), step=50
        )
        self.slider_wb.setEnabled(not self.config.auto_wb)
        self.spin_wb.setEnabled(not self.config.auto_wb)
        s_grid.addWidget(self.slider_wb, 2, 1)
        s_grid.addWidget(self.spin_wb, 2, 2)

        s_grid.addWidget(QLabel("Gain:"), 3, 0)
        self.slider_gain, self.spin_gain = self._make_slider_spin(
            0, 255, self.config.gain, lambda v: self.worker.update_camera_hardware("gain", v)
        )
        s_grid.addWidget(self.slider_gain, 3, 1)
        s_grid.addWidget(self.spin_gain, 3, 2)
        layout.addWidget(slider_group)

        prof_group = QGroupBox("Ekspor Kode")
        prof_layout = QVBoxLayout(prof_group)
        btn_copy_code = QPushButton("Salin Kode cap.set(...)")
        btn_copy_code.clicked.connect(self.export_cap_set_snippet)
        prof_layout.addWidget(btn_copy_code)
        layout.addWidget(prof_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(tab)
        return scroll

    @staticmethod
    def _make_slider_spin(min_val: int, max_val: int, init_val: int, callback, step: int = 1):
        slider = QSlider(Qt.Horizontal)
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
        is_yolo = self.config.hand_backend == "yolo26n"
        self.edit_yolo_model.setEnabled(is_yolo)
        self.btn_browse_yolo.setEnabled(is_yolo)

    def on_browse_yolo_model(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih Model YOLO Pose",
            "",
            "YOLO Models (*.pt *.onnx *.engine);;All Files (*)",
        )
        if filepath:
            self.edit_yolo_model.setText(filepath)
            self.config.yolo_model_path = filepath

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

        backend_idx_map = {"rtmlib": 0, "mediapipe": 1, "yolo26n": 2}
        self.combo_backend.setCurrentIndex(backend_idx_map.get(self.config.hand_backend, 0))
        self.edit_yolo_model.setText(self.config.yolo_model_path)
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

        # State UI untuk kontrol frame video & perekaman
        self._is_scrubbing_slider: bool = False
        self._last_video_path: str = ""
        self._last_video_total: int = 0
        self._is_video_mode_ui: bool = False
        self._is_recording_ui: bool = False

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
        title_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title_lbl.setStyleSheet("color: #58a6ff; letter-spacing: 1px;")

        header_bar.addWidget(title_lbl)
        header_bar.addStretch()

        self.btn_open_video = QPushButton("Buka Video (Ctrl+O)")
        self.btn_open_video.setStyleSheet(
            "background-color: #1f5f35; border-color: #3fb950; font-weight: bold; padding: 6px 14px;"
        )
        self.btn_open_video.clicked.connect(self.action_open_video)

        self.btn_live_cam = QPushButton("Kamera Live")
        self.btn_live_cam.setStyleSheet(
            "background-color: #21262d; border-color: #30363d; font-weight: bold; padding: 6px 12px;"
        )
        self.btn_live_cam.clicked.connect(self.action_switch_to_camera)
        self.btn_live_cam.setVisible(False)

        btn_settings = QPushButton("Pengaturan (Ctrl+,)")
        btn_settings.setStyleSheet("background-color: #1f4f82; border-color: #58a6ff; font-weight: bold; padding: 6px 14px;")
        btn_settings.clicked.connect(self.open_settings_dialog)

        btn_clear_cycle = QPushButton("Reset Siklus (Ctrl+Shift+C)")
        btn_clear_cycle.clicked.connect(self.action_clear_status)

        btn_recapture_ref = QPushButton("Ambil Ulang Ref (Ctrl+B)")
        btn_recapture_ref.clicked.connect(self.action_recapture_ref)

        btn_reset_roi = QPushButton("Buat Ulang ROI (Ctrl+R)")
        btn_reset_roi.setStyleSheet("background-color: #491217; border-color: #f85149;")
        btn_reset_roi.clicked.connect(self.action_reset_roi)

        header_bar.addWidget(self.btn_open_video)
        header_bar.addWidget(self.btn_live_cam)
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
        self.banner_lbl.setAlignment(Qt.AlignCenter)
        self.banner_lbl.setFixedHeight(44)
        self.banner_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self._set_banner_style(BannerState.SETUP)

        self.canvas = VideoCanvas()
        left_layout.addWidget(self.banner_lbl)
        left_layout.addWidget(self.canvas, stretch=1)

        # Panel Kontrol Frame & Video Input/Output
        self.video_ctrl_frame = self._build_video_control_panel()
        left_layout.addWidget(self.video_ctrl_frame)

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
        self.part_count_lbl.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.part_count_lbl.setStyleSheet("color: #3fb950;")
        self.part_count_lbl.setAlignment(Qt.AlignCenter)
        lbl_p_desc = QLabel("Part Selesai")
        lbl_p_desc.setStyleSheet("color: #8b949e; font-size: 11px;")
        lbl_p_desc.setAlignment(Qt.AlignCenter)
        cp_layout.addWidget(self.part_count_lbl)
        cp_layout.addWidget(lbl_p_desc)

        card_step = QFrame()
        card_step.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px;")
        cs_layout = QVBoxLayout(card_step)
        cs_layout.setContentsMargins(8, 6, 8, 6)
        cs_layout.setSpacing(2)
        self.step_count_lbl = QLabel("0 / 3")
        self.step_count_lbl.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.step_count_lbl.setStyleSheet("color: #58a6ff;")
        self.step_count_lbl.setAlignment(Qt.AlignCenter)
        lbl_s_desc = QLabel("Progres Langkah")
        lbl_s_desc.setStyleSheet("color: #8b949e; font-size: 11px;")
        lbl_s_desc.setAlignment(Qt.AlignCenter)
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
        roi_group = QGroupBox("Urutan Langkah")
        roi_group_layout = QVBoxLayout(roi_group)
        roi_group_layout.setContentsMargins(8, 8, 8, 8)
        roi_scroll = QScrollArea()
        roi_scroll.setWidgetResizable(True)
        roi_scroll.setFrameShape(QFrame.NoFrame)
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
        log_group = QGroupBox("Log Aktivitas")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        log_layout.setSpacing(6)

        self.log_list = QListWidget()
        log_layout.addWidget(self.log_list, stretch=1)

        log_btn_row = QHBoxLayout()
        btn_clear_log = QPushButton("Bersihkan Log")
        btn_clear_log.setStyleSheet("font-size: 11px; padding: 3px 8px;")
        btn_clear_log.clicked.connect(self.log_list.clear)

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

    def _build_video_control_panel(self) -> QGroupBox:
        group = QGroupBox("Kontrol Frame & Video (Input / Output)")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Baris 1: Navigasi & Pemilihan Frame oleh Pengguna
        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)

        self.btn_play_pause = QPushButton("▶ Putar")
        self.btn_play_pause.setFixedWidth(90)
        self.btn_play_pause.setStyleSheet(
            "background-color: #1f4f82; border-color: #58a6ff; font-weight: bold; padding: 5px 8px;"
        )
        self.btn_play_pause.clicked.connect(self.action_toggle_video_play)
        self.btn_play_pause.setEnabled(False)

        self.btn_prev_frame = QPushButton("◀ -1")
        self.btn_prev_frame.setFixedWidth(52)
        self.btn_prev_frame.setToolTip("Mundur 1 frame")
        self.btn_prev_frame.clicked.connect(lambda: self.worker.step_video_frame(-1))
        self.btn_prev_frame.setEnabled(False)

        self.btn_next_frame = QPushButton("+1 ▶")
        self.btn_next_frame.setFixedWidth(52)
        self.btn_next_frame.setToolTip("Maju 1 frame")
        self.btn_next_frame.clicked.connect(lambda: self.worker.step_video_frame(1))
        self.btn_next_frame.setEnabled(False)

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.setEnabled(False)
        self.slider_frame.sliderPressed.connect(self._on_slider_pressed)
        self.slider_frame.sliderReleased.connect(self._on_slider_released)
        self.slider_frame.valueChanged.connect(self._on_slider_frame_changed)

        lbl_frame_title = QLabel("Frame:")
        lbl_frame_title.setStyleSheet("color: #8b949e; font-size: 12px;")

        self.spin_current_frame = QSpinBox()
        self.spin_current_frame.setRange(0, 0)
        self.spin_current_frame.setFixedWidth(85)
        self.spin_current_frame.setEnabled(False)
        self.spin_current_frame.valueChanged.connect(self._on_spin_current_frame_changed)

        self.lbl_total_frames = QLabel("/ 0 (00:00.0)")
        self.lbl_total_frames.setStyleSheet("color: #8b949e; font-size: 11px; min-width: 95px;")

        nav_row.addWidget(self.btn_play_pause)
        nav_row.addWidget(self.btn_prev_frame)
        nav_row.addWidget(self.btn_next_frame)
        nav_row.addWidget(self.slider_frame, stretch=1)
        nav_row.addWidget(lbl_frame_title)
        nav_row.addWidget(self.spin_current_frame)
        nav_row.addWidget(self.lbl_total_frames)
        layout.addLayout(nav_row)

        # Baris 2: Penentuan Frame Referensi & Rentang Frame Ekspor Video Output
        range_row = QHBoxLayout()
        range_row.setSpacing(6)

        self.btn_lock_ref_frame = QPushButton("Jadikan Frame Ini Referensi")
        self.btn_lock_ref_frame.setToolTip(
            "Kunci frame yang sedang tampil saat ini sebagai referensi background bersih"
        )
        self.btn_lock_ref_frame.setStyleSheet(
            "background-color: #21262d; border-color: #58a6ff; color: #58a6ff; font-size: 11px; padding: 5px 10px;"
        )
        self.btn_lock_ref_frame.clicked.connect(self.action_lock_current_frame_ref)

        lbl_start = QLabel("Frame Awal:")
        lbl_start.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.spin_start_frame = QSpinBox()
        self.spin_start_frame.setRange(0, 0)
        self.spin_start_frame.setFixedWidth(78)
        self.spin_start_frame.setEnabled(False)
        self.spin_start_frame.valueChanged.connect(self._on_frame_range_changed)

        self.btn_set_start_curr = QPushButton("Set Saat Ini")
        self.btn_set_start_curr.setStyleSheet("font-size: 11px; padding: 4px 6px;")
        self.btn_set_start_curr.setToolTip("Gunakan nomor frame saat ini sebagai Frame Awal")
        self.btn_set_start_curr.clicked.connect(self._set_start_from_current)
        self.btn_set_start_curr.setEnabled(False)

        lbl_end = QLabel("Frame Akhir:")
        lbl_end.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.spin_end_frame = QSpinBox()
        self.spin_end_frame.setRange(0, 0)
        self.spin_end_frame.setFixedWidth(78)
        self.spin_end_frame.setEnabled(False)
        self.spin_end_frame.valueChanged.connect(self._on_frame_range_changed)

        self.btn_set_end_curr = QPushButton("Set Saat Ini")
        self.btn_set_end_curr.setStyleSheet("font-size: 11px; padding: 4px 6px;")
        self.btn_set_end_curr.setToolTip("Gunakan nomor frame saat ini sebagai Frame Akhir")
        self.btn_set_end_curr.clicked.connect(self._set_end_from_current)
        self.btn_set_end_curr.setEnabled(False)

        self.btn_record_output = QPushButton("Simpan Video Output...")
        self.btn_record_output.setStyleSheet(
            "background-color: #1f5f35; border-color: #3fb950; font-weight: bold; padding: 5px 12px;"
        )
        self.btn_record_output.clicked.connect(self.action_toggle_recording)

        range_row.addWidget(self.btn_lock_ref_frame)
        range_row.addStretch()
        range_row.addWidget(lbl_start)
        range_row.addWidget(self.spin_start_frame)
        range_row.addWidget(self.btn_set_start_curr)
        range_row.addWidget(lbl_end)
        range_row.addWidget(self.spin_end_frame)
        range_row.addWidget(self.btn_set_end_curr)
        range_row.addWidget(self.btn_record_output)
        layout.addLayout(range_row)

        return group

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
        act_o = QAction(self)
        act_o.setShortcut(QKeySequence("Ctrl+O"))
        act_o.triggered.connect(self.action_open_video)
        self.addAction(act_o)

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
        dlg.exec_()

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
            item.setFont(QFont("Segoe UI", 9, QFont.Bold))

        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    @staticmethod
    def _format_sec(sec: float) -> str:
        sec = max(0.0, float(sec))
        m = int(sec // 60)
        s = sec - m * 60
        return f"{m:02d}:{s:04.1f}"

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

        # Sinkronisasi kontrol video & penentuan frame oleh pengguna
        is_video = bool(data.get("is_video_mode", False))
        v_path = str(data.get("video_path", ""))
        v_total = int(data.get("video_total_frames", 0))
        v_curr = int(data.get("video_current_frame", 0))
        v_start = int(data.get("video_start_frame", 0))
        v_end = int(data.get("video_end_frame", max(0, v_total - 1)))
        v_paused = bool(data.get("video_paused", True))
        v_fps = float(data.get("video_fps", 30.0) or 30.0)
        is_rec = bool(data.get("is_recording", False))
        rec_count = int(data.get("recorded_frames_count", 0))

        if is_video != self._is_video_mode_ui:
            self._is_video_mode_ui = is_video
            self.btn_live_cam.setVisible(is_video)
            self.btn_play_pause.setEnabled(is_video)
            self.btn_prev_frame.setEnabled(is_video)
            self.btn_next_frame.setEnabled(is_video)
            self.slider_frame.setEnabled(is_video)
            self.spin_current_frame.setEnabled(is_video)
            self.spin_start_frame.setEnabled(is_video)
            self.spin_end_frame.setEnabled(is_video)
            self.btn_set_start_curr.setEnabled(is_video)
            self.btn_set_end_curr.setEnabled(is_video)
            if not is_video:
                self.lbl_total_frames.setText("/ 0 (Live)")
                self._last_video_path = ""
                self._last_video_total = 0

        if is_video:
            max_idx = max(0, v_total - 1)
            if v_path != self._last_video_path or v_total != self._last_video_total:
                self._last_video_path = v_path
                self._last_video_total = v_total
                for w in (self.slider_frame, self.spin_current_frame, self.spin_start_frame, self.spin_end_frame):
                    w.blockSignals(True)
                    w.setRange(0, max_idx)
                self.spin_start_frame.setValue(v_start)
                self.spin_end_frame.setValue(v_end)
                for w in (self.slider_frame, self.spin_current_frame, self.spin_start_frame, self.spin_end_frame):
                    w.blockSignals(False)

            self.btn_play_pause.setText("▶ Putar" if v_paused else "⏸ Jeda")

            if not self._is_scrubbing_slider:
                self.slider_frame.blockSignals(True)
                self.slider_frame.setValue(v_curr)
                self.slider_frame.blockSignals(False)

            if not self.spin_current_frame.hasFocus():
                self.spin_current_frame.blockSignals(True)
                self.spin_current_frame.setValue(v_curr)
                self.spin_current_frame.blockSignals(False)

            curr_sec = v_curr / v_fps if v_fps > 0 else 0.0
            tot_sec = max_idx / v_fps if v_fps > 0 else 0.0
            self.lbl_total_frames.setText(
                f"/ {max_idx} ({self._format_sec(curr_sec)} / {self._format_sec(tot_sec)})"
            )

        if is_rec != self._is_recording_ui or is_rec:
            self._is_recording_ui = is_rec
            if is_rec:
                self.btn_record_output.setText(f"⏹ Stop & Simpan ({rec_count} f)")
                self.btn_record_output.setStyleSheet(
                    "background-color: #6e1920; border-color: #f85149; font-weight: bold; padding: 5px 12px;"
                )
            else:
                self.btn_record_output.setText("Simpan Video Output...")
                self.btn_record_output.setStyleSheet(
                    "background-color: #1f5f35; border-color: #3fb950; font-weight: bold; padding: 5px 12px;"
                )

        if self.config.hand_backend == "rtmlib":
            engine_str = "RTMlib (RTMPose)"
        elif self.config.hand_backend == "yolo26n":
            model_name = os.path.basename(self.config.yolo_model_path or "yolo26n-pose.pt")
            engine_str = f"YOLO26n-Pose ({model_name})"
        else:
            engine_str = "Google MediaPipe"
        rec_str = f" | [REC: {rec_count} frame]" if is_rec else ""
        if data["has_reference"]:
            self.ref_status_lbl.setText(
                f"Engine: {engine_str} | Referensi: Tersimpan | FPS: {data['fps']:.1f}{rec_str}"
            )
        else:
            self.ref_status_lbl.setText(
                f"Engine: {engine_str} | Referensi: Menunggu ({data['roi_defined']}/{data['roi_total']} ROI) | FPS: {data['fps']:.1f}{rec_str}"
            )

    @pyqtSlot(str)
    def on_camera_info(self, info: str) -> None:
        self.cam_status_lbl.setText(info)

    # =========================================================================
    # Handler Kontrol Video, Penentuan Frame, dan Perekaman Output
    # =========================================================================
    def action_open_video(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih File Video Input",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*)",
        )
        if filepath:
            self.worker.open_video_file(filepath)

    def action_switch_to_camera(self) -> None:
        self.worker.request_camera_switch(self.config.camera_index)

    def action_toggle_video_play(self) -> None:
        self.worker.toggle_video_play()

    def _on_slider_pressed(self) -> None:
        self._is_scrubbing_slider = True
        self.worker.set_video_paused(True)

    def _on_slider_released(self) -> None:
        self._is_scrubbing_slider = False
        val = self.slider_frame.value()
        self.spin_current_frame.blockSignals(True)
        self.spin_current_frame.setValue(val)
        self.spin_current_frame.blockSignals(False)
        self.worker.seek_video_frame(val)

    def _on_slider_frame_changed(self, val: int) -> None:
        self.spin_current_frame.blockSignals(True)
        self.spin_current_frame.setValue(val)
        self.spin_current_frame.blockSignals(False)
        self.worker.seek_video_frame(val)

    def _on_spin_current_frame_changed(self, val: int) -> None:
        self.slider_frame.blockSignals(True)
        self.slider_frame.setValue(val)
        self.slider_frame.blockSignals(False)
        self.worker.set_video_paused(True)
        self.worker.seek_video_frame(val)

    def _on_frame_range_changed(self) -> None:
        sf = self.spin_start_frame.value()
        ef = self.spin_end_frame.value()
        if sf > ef:
            if self.sender() == self.spin_start_frame:
                self.spin_end_frame.blockSignals(True)
                self.spin_end_frame.setValue(sf)
                self.spin_end_frame.blockSignals(False)
                ef = sf
            else:
                self.spin_start_frame.blockSignals(True)
                self.spin_start_frame.setValue(ef)
                self.spin_start_frame.blockSignals(False)
                sf = ef
        self.worker.set_video_frame_range(sf, ef)

    def _set_start_from_current(self) -> None:
        curr = self.spin_current_frame.value()
        self.spin_start_frame.setValue(curr)
        self.add_log_entry("INFO", f"Frame Awal diatur ke Frame #{curr}.")

    def _set_end_from_current(self) -> None:
        curr = self.spin_current_frame.value()
        self.spin_end_frame.setValue(curr)
        self.add_log_entry("INFO", f"Frame Akhir diatur ke Frame #{curr}.")

    def action_lock_current_frame_ref(self, roi_mode: Optional[str] = None) -> None:
        if isinstance(roi_mode, str) and roi_mode in ("keep", "add", "reset"):
            self.worker.lock_current_frame_as_ref(roi_mode=roi_mode)
            return

        if self._is_video_mode_ui:
            self.worker.set_video_paused(True)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Jadikan Frame Ini Referensi")
        msg_box.setIcon(QMessageBox.Question)
        curr_roi_count = len(self.config.saved_rois)
        msg_box.setText(
            "Frame saat ini akan dikunci sebagai referensi background bersih.\n\n"
            f"Pilih pengaturan ROI yang ingin digunakan (ROI aktif saat ini: {curr_roi_count}):"
        )

        btn_keep = msg_box.addButton("Gunakan ROI Saat Ini", QMessageBox.AcceptRole)
        btn_add = msg_box.addButton("Tambah ROI Baru", QMessageBox.ActionRole)
        btn_reset = msg_box.addButton("Buat Ulang Semua ROI", QMessageBox.ResetRole)
        btn_cancel = msg_box.addButton("Batal", QMessageBox.RejectRole)
        msg_box.setDefaultButton(btn_keep)

        msg_box.exec_()
        clicked = msg_box.clickedButton()
        if clicked == btn_cancel or clicked is None:
            return
        if clicked == btn_add:
            self.worker.lock_current_frame_as_ref(roi_mode="add")
        elif clicked == btn_reset:
            self.worker.lock_current_frame_as_ref(roi_mode="reset")
        else:
            self.worker.lock_current_frame_as_ref(roi_mode="keep")

    def action_toggle_recording(self) -> None:
        if self._is_recording_ui:
            self.worker.stop_video_recording()
            return

        default_name = "output_annotated.mp4"
        if self._last_video_path:
            base, _ = os.path.splitext(self._last_video_path)
            default_name = f"{base}_annotated.mp4"

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Simpan Video Output Teranotasi",
            default_name,
            "MP4 Video (*.mp4);;AVI Video (*.avi)",
        )
        if not out_path:
            return

        if self._is_video_mode_ui:
            sf = self.spin_start_frame.value()
            ef = self.spin_end_frame.value()
            self.worker.start_video_recording(out_path, start_frame=sf, end_frame=ef)
        else:
            self.worker.start_video_recording(out_path)

    def action_reset_roi(self) -> None:
        reply = QMessageBox.question(
            self,
            "Konfirmasi Buat Ulang ROI",
            "Apakah Anda yakin ingin menghapus semua ROI hasil kalibrasi dan membuat ulang dari awal?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.worker.trigger_reset_roi()

    def action_recapture_ref(self) -> None:
        self.worker.trigger_recapture_ref()

    def action_clear_status(self) -> None:
        self.worker.trigger_clear_status()

    def action_reset_parts(self) -> None:
        self.worker.trigger_reset_parts()

    def shutdown_all(self) -> None:
        """Mematikan thread worker dan memastikan seluruh kamera dilepas."""
        try:
            self.worker.stop()
        finally:
            release_all_cameras()

    def closeEvent(self, event) -> None:
        reply = QMessageBox.question(
            self,
            "Konfirmasi Keluar",
            "Apakah Anda yakin ingin menutup aplikasi?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return
        self.status_bar.showMessage("Menutup aplikasi dan mematikan seluruh kamera...", 1000)
        self.shutdown_all()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.shutdown_all)
    window.show()
    try:
        exit_code = app.exec_()
    finally:
        window.shutdown_all()
    sys.exit(exit_code)
