"""web.py: Antarmuka Web Dashboard & REST API untuk Palm Detection pada mode Headless.

Dirancang khusus untuk pengoperasian di perangkat edge seperti Nvidia Jetson
tanpa memerlukan layar monitor fisik atau server tampilan X11/GUI.
Mendukung Python 3.10+ secara penuh dengan framework Flask yang ringan.

Penggunaan:
    python web.py
    python web.py --host 0.0.0.0 --port 5000
    python web.py --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import threading
import time
from typing import Optional

# Pre-load MediaPipe sebelum modul lain yang memuat PyQt5 (seperti CameraWorker)
# untuk mencegah konflik inisialisasi DLL C++ (_framework_bindings) di Windows.
try:
    import mediapipe as _mp  # noqa: F401
except Exception:
    pass

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from palm_detection.config import AppConfig, CONFIG_FILE
from palm_detection.detector import MODEL_DIR, ensure_model_dir, list_available_yolo_models
from palm_detection.states import BannerState, HandBackend, LogLevel, RoiState, TrackingMode
from palm_detection.worker import CameraWorker, release_all_cameras

# Inisialisasi Flask App
app = Flask(__name__)

# State Global Worker
_worker_instance: Optional[CameraWorker] = None
_worker_lock = threading.Lock()
_global_config_path: str = CONFIG_FILE


def get_worker() -> CameraWorker:
    """Mengambil atau menginisialisasi instance tunggal CameraWorker."""
    global _worker_instance
    with _worker_lock:
        if _worker_instance is None:
            cfg = AppConfig.load(_global_config_path)
            _worker_instance = CameraWorker(cfg)
            _worker_instance.start()
        return _worker_instance


def cleanup_worker() -> None:
    """Membersihkan worker dan melepas seluruh kamera saat server berhenti."""
    global _worker_instance
    with _worker_lock:
        if _worker_instance is not None:
            try:
                _worker_instance.stop()
            except Exception:
                pass
            _worker_instance = None
    release_all_cameras()


atexit.register(cleanup_worker)

# =============================================================================
# Embedded Single-Page HTML Dashboard (Mandiri, Tanpa Dependensi CDN Luar)
# =============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Palm Detection - Headless Web Console</title>
    <style>
        :root {
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-hover: #21262d;
            --border: #30363d;
            --text-main: #c9d1d9;
            --text-muted: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-amber: #d29922;
            --accent-red: #f85149;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            user-select: none;
        }

        header {
            background-color: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .header-title h1 {
            font-size: 1.15rem;
            color: var(--accent-blue);
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }

        .badge-live { background-color: #1f3d2b; color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-video { background-color: #2e2817; color: var(--accent-amber); border: 1px solid var(--accent-amber); }

        .btn-group { display: flex; gap: 8px; flex-wrap: wrap; }
        .btn {
            background-color: var(--bg-hover);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.82rem;
            font-weight: 600;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn:hover { background-color: #30363d; border-color: #8b949e; }
        .btn-primary { background-color: #1f5f35; border-color: var(--accent-green); color: #fff; }
        .btn-primary:hover { background-color: #2ea043; }
        .btn-danger { background-color: #491217; border-color: var(--accent-red); color: #ffa198; }
        .btn-danger:hover { background-color: #b62324; color: #fff; }

        .main-container {
            display: flex;
            flex: 1;
            padding: 14px;
            gap: 14px;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        @media (max-width: 1024px) {
            .main-container { flex-direction: column; }
        }

        .left-col {
            flex: 1.4;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .banner-bar {
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 700;
            text-align: center;
            border: 2px solid var(--accent-blue);
            background-color: #13233a;
            color: #79c0ff;
            transition: all 0.25s ease;
        }

        .canvas-container {
            position: relative;
            background-color: #05070a;
            border: 2px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 400px;
            cursor: crosshair;
        }

        .canvas-container img {
            width: 100%;
            height: auto;
            max-height: 70vh;
            object-fit: contain;
            display: block;
        }

        .canvas-hint {
            position: absolute;
            bottom: 8px;
            left: 8px;
            background: rgba(13, 17, 23, 0.85);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            color: var(--text-muted);
            border: 1px solid var(--border);
            pointer-events: none;
        }

        .video-ctrl-panel {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .video-ctrl-panel input[type="range"] {
            flex: 1;
            accent-color: var(--accent-blue);
            cursor: pointer;
        }

        .right-col {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
        }

        .card-header {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .counters-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .counter-box {
            background-color: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px;
            text-align: center;
        }

        .counter-val {
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .val-parts { color: var(--accent-green); }
        .val-steps { color: var(--accent-blue); }

        .counter-lbl {
            font-size: 0.7rem;
            color: var(--text-muted);
            font-weight: 700;
            margin-top: 4px;
            letter-spacing: 0.5px;
        }

        .roi-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 250px;
            overflow-y: auto;
        }

        .roi-item {
            background-color: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }

        .roi-item input {
            background: transparent;
            border: 1px solid transparent;
            color: var(--accent-blue);
            font-size: 0.82rem;
            font-weight: 700;
            padding: 2px 4px;
            border-radius: 4px;
            width: 140px;
        }
        .roi-item input:focus {
            border-color: var(--accent-blue);
            background: #161b22;
            outline: none;
        }

        .roi-metrics {
            font-size: 0.72rem;
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 2px;
        }

        .badge-roi {
            font-size: 0.68rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .log-box {
            background-color: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px;
            font-family: monospace;
            font-size: 0.76rem;
            height: 180px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .log-entry { line-height: 1.3; }
        .log-INFO { color: var(--text-muted); }
        .log-SUCCESS { color: var(--accent-green); font-weight: bold; }
        .log-WARN { color: var(--accent-red); font-weight: bold; }
        .log-PART { color: var(--accent-blue); font-weight: bold; }

        /* Modal Settings */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 999;
        }

        .modal {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            width: 90%;
            max-width: 600px;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .modal-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 700;
        }

        .modal-body {
            padding: 16px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .form-group label {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .form-group input, .form-group select {
            background: #0d1117;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 0.82rem;
        }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .modal-footer {
            padding: 10px 16px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: flex-end;
            gap: 8px;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>Palm Detection Web Console</h1>
            <span id="badgeMode" class="badge badge-live">Live Kamera</span>
            <span id="fpsVal" style="font-size: 0.75rem; color: var(--text-muted);">0.0 FPS</span>
        </div>
        <div class="btn-group">
            <button class="btn" onclick="triggerAction('recapture_ref')">Ambil Ulang Ref</button>
            <button class="btn" onclick="triggerAction('clear_status')">Reset Status</button>
            <button class="btn btn-danger" onclick="confirmResetRoi()">Buat Ulang ROI</button>
            <button class="btn btn-primary" onclick="openSettings()">Pengaturan</button>
        </div>
    </header>

    <div class="main-container">
        <!-- Kolom Kiri: Video & Kontrol Canvas -->
        <div class="left-col">
            <div id="bannerBar" class="banner-bar">Menghubungkan ke sistem inspeksi...</div>

            <div class="canvas-container" id="canvasBox">
                <img id="streamImg" src="/video_feed" alt="Video Stream" onclick="onCanvasClick(event)" oncontextmenu="onCanvasRightClick(event)">
                <div class="canvas-hint">Klik 4 titik pada video untuk membuat sudut ROI. Klik kanan untuk Undo.</div>
            </div>

            <!-- Panel Kontrol Mode Video (Muncul jika sedang buka file video) -->
            <div id="videoControls" class="video-ctrl-panel" style="display: none;">
                <button class="btn" id="btnPlayPause" onclick="togglePlayPause()">Putar</button>
                <button class="btn" onclick="stepFrame(-1)">-1 Frame</button>
                <button class="btn" onclick="stepFrame(1)">+1 Frame</button>
                <input type="range" id="frameSlider" min="0" max="100" value="0" oninput="onSliderSeek(this.value)">
                <span id="frameCounter" style="font-size: 0.78rem; font-family: monospace;">0 / 0</span>
                <button class="btn btn-primary" onclick="lockCurrentFrame()">Kunci sbg Ref</button>
            </div>
        </div>

        <!-- Kolom Kanan: Telemetri, Status ROI, dan Log -->
        <div class="right-col">
            <div class="counters-row">
                <div class="counter-box">
                    <div id="partsVal" class="counter-val val-parts">0</div>
                    <div class="counter-lbl">PART SELESAI</div>
                    <button class="btn" style="margin-top: 6px; padding: 2px 8px; font-size: 0.68rem;" onclick="triggerAction('reset_parts')">Reset Part</button>
                </div>
                <div class="counter-box">
                    <div id="stepsVal" class="counter-val val-steps">0 / 0</div>
                    <div class="counter-lbl">PROGRES LANGKAH</div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span>Status Area Kerja (ROI)</span>
                    <span id="roiCountSummary" style="font-size: 0.75rem;">0 Terdefinisi</span>
                </div>
                <div id="roiList" class="roi-list">
                    <!-- Kartu ROI dinamis di-render via Javascript -->
                </div>
            </div>

            <div class="card" style="flex: 1;">
                <div class="card-header">
                    <span>Log Aktivitas Sistem</span>
                    <button class="btn" style="padding: 2px 6px; font-size: 0.65rem;" onclick="clearLogUi()">Bersihkan</button>
                </div>
                <div id="logBox" class="log-box"></div>
            </div>
        </div>
    </div>

    <!-- Modal Pengaturan -->
    <div id="settingsModal" class="modal-overlay">
        <div class="modal">
            <div class="modal-header">
                <span>Pengaturan Sistem Inspeksi</span>
                <button class="btn" style="padding: 2px 8px;" onclick="closeSettings()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label>AI Hand Tracking Backend</label>
                    <select id="cfgBackend">
                        <option value="rtmlib">RTMlib (RTMPose Hand)</option>
                        <option value="mediapipe">Google MediaPipe Hands</option>
                        <option value="yolo26n">YOLO Pose (Keypoints)</option>
                        <option value="yolo_detect">YOLO Object Detection (Deteksi Sarung Tangan)</option>
                    </select>
                </div>
                <div class="form-group" id="groupYoloModel">
                    <label>Model YOLO (palm_detection/model)</label>
                    <select id="cfgYoloModelSelect" onchange="if(this.value) document.getElementById('cfgYoloModel').value = this.value;">
                        <option value="">-- Pilih dari palm_detection/model --</option>
                    </select>
                    <input type="text" id="cfgYoloModel" placeholder="Nama file di palm_detection/model atau path .pt/.onnx" style="margin-top: 4px;">
                </div>
                <div class="form-group" id="groupYoloClass">
                    <label>Filter Nama Kelas YOLO Object Detection</label>
                    <input type="text" id="cfgYoloClass" placeholder="gloves" value="gloves">
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Tracking Mode</label>
                        <select id="cfgTrackMode">
                            <option value="full_hand">Seluruh Tangan (21 Sendi)</option>
                            <option value="palm_center">Pusat Telapak</option>
                            <option value="fingertips">5 Ujung Jari</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Jumlah ROI (Langkah)</label>
                        <input type="number" id="cfgJumlahRoi" min="1" max="6">
                    </div>
                </div>
                <div class="form-group">
                    <label>Perbandingan Piksel (Sebelum vs Sesudah)</label>
                    <select id="cfgUsePixelDiff">
                        <option value="false">Nonaktif (Hanya Debounce)</option>
                        <option value="true">Aktif (Validasi Diff Piksel)</option>
                    </select>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Ambang Piksel (0-255)</label>
                        <input type="number" id="cfgAmbangPiksel" min="5" max="250">
                    </div>
                    <div class="form-group">
                        <label>Ambang Rasio (%)</label>
                        <input type="number" id="cfgAmbangRasio" min="0.5" max="95" step="0.5">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Debounce Masuk (Frame)</label>
                        <input type="number" id="cfgDebounceIn" min="1" max="60">
                    </div>
                    <div class="form-group">
                        <label>Debounce Keluar (Frame)</label>
                        <input type="number" id="cfgDebounceOut" min="1" max="90">
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Pangkas Area (ROI Crop Mode)</label>
                        <select id="cfgCropMode">
                            <option value="union">Union (Gabungan ROI)</option>
                            <option value="individual">Individual Per-ROI</option>
                            <option value="none">Tanpa Crop (Layar Penuh)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Optimasi Sarung Tangan (CLAHE)</label>
                        <select id="cfgGloveEnhance">
                            <option value="true">Aktif (Kontras Lipatan Sendi)</option>
                            <option value="false">Nonaktif</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn" onclick="closeSettings()">Batal</button>
                <button class="btn btn-primary" onclick="saveSettings()">Simpan Konfigurasi</button>
            </div>
        </div>
    </div>

    <script>
        const BANNER_STYLES = {
            "setup": { bg: "#13233a", border: "#388bfd", text: "#79c0ff" },
            "active": { bg: "#2b230f", border: "#d29922", text: "#ffd33d" },
            "error": { bg: "#491217", border: "#f85149", text: "#ffa198" },
            "done": { bg: "#122d1f", border: "#2ea043", text: "#56d364" }
        };

        const ROI_STYLES = {
            "SETUP": { bg: "#161b22", border: "#484f58", badge: "#8b949e" },
            "READY": { bg: "#13233a", border: "#388bfd", badge: "#58a6ff" },
            "WAIT_REF": { bg: "#13233a", border: "#388bfd", badge: "#58a6ff" },
            "LOCKED": { bg: "#161b22", border: "#30363d", badge: "#6e7681" },
            "ACTIVE": { bg: "#2b230f", border: "#d29922", badge: "#e3b341" },
            "HAND_IN": { bg: "#311b10", border: "#db6d28", badge: "#f0883e" },
            "DONE": { bg: "#122d1f", border: "#2ea043", badge: "#3fb950" },
            "ERROR": { bg: "#361417", border: "#f85149", badge: "#ff7b72" }
        };

        // Polling Telemetri & Log secara periodik
        let isSeeking = false;

        async function pollTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                if (!res.ok) return;
                const data = await res.json();
                updateDashboard(data);
            } catch (err) {
                // Abaikan jika network drop sesaat
            }
        }

        async function pollLogs() {
            try {
                const res = await fetch('/api/logs');
                if (!res.ok) return;
                const logs = await res.json();
                renderLogs(logs);
            } catch (err) {}
        }

        setInterval(pollTelemetry, 150);
        setInterval(pollLogs, 800);

        function updateDashboard(t) {
            // Update Banner
            const banner = document.getElementById('bannerBar');
            banner.innerText = t.banner_text || "Memuat...";
            const st = BANNER_STYLES[t.banner_state] || BANNER_STYLES["setup"];
            banner.style.backgroundColor = st.bg;
            banner.style.borderColor = st.border;
            banner.style.color = st.text;

            // Update FPS & Mode
            document.getElementById('fpsVal').innerText = (t.fps || 0).toFixed(1) + " FPS";
            const badgeMode = document.getElementById('badgeMode');
            if (t.is_video_mode) {
                badgeMode.className = "badge badge-video";
                badgeMode.innerText = "Mode Video";
                document.getElementById('videoControls').style.display = "flex";
            } else {
                badgeMode.className = "badge badge-live";
                badgeMode.innerText = "Live Kamera";
                document.getElementById('videoControls').style.display = "none";
            }

            // Update Counters
            document.getElementById('partsVal').innerText = t.parts_completed || 0;
            document.getElementById('stepsVal').innerText = (t.steps_completed || 0) + " / " + (t.roi_total || 0);
            document.getElementById('roiCountSummary').innerText = (t.roi_defined || 0) + " / " + (t.roi_total || 0) + " Terdefinisi";

            // Update Video Slider
            if (t.is_video_mode && !isSeeking) {
                const slider = document.getElementById('frameSlider');
                slider.max = t.video_total_frames || 100;
                slider.value = t.video_current_frame || 0;
                document.getElementById('frameCounter').innerText = (t.video_current_frame || 0) + " / " + (t.video_total_frames || 0);
                document.getElementById('btnPlayPause').innerText = t.video_paused ? "Putar" : "Jeda";
            }

            // Render ROI List
            renderRoiCards(t.rois || []);
        }

        function renderRoiCards(rois) {
            const list = document.getElementById('roiList');
            if (rois.length === 0) {
                list.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 12px;">Belum ada area ROI yang dibuat. Klik pada layar video untuk menambahkan.</div>';
                return;
            }

            let html = '';
            rois.forEach((r, idx) => {
                const st = ROI_STYLES[r.state] || ROI_STYLES["LOCKED"];
                const ratioPct = ((r.diff_ratio || 0) * 100).toFixed(1);
                const ambangPct = ((r.ambang_rasio || 0.08) * 100).toFixed(1);
                const diffSubtext = (r.use_pixel_diff === false)
                    ? 'Diff: Nonaktif (Debounce)'
                    : `Diff: ${ratioPct}% (Min ${ambangPct}%)`;
                html += `
                    <div class="roi-item" style="border-left: 4px solid ${st.badge};">
                        <div>
                            <input type="text" value="${r.name || ('ROI #' + (idx+1))}" onchange="changeRoiName(${idx}, this.value)">
                            <div style="font-size: 0.68rem; color: var(--text-muted); margin-left: 4px;">${diffSubtext}</div>
                        </div>
                        <div class="roi-metrics">
                            <span class="badge-roi" style="background: ${st.bg}; color: ${st.badge}; border: 1px solid ${st.border};">${r.state}</span>
                            <span style="font-size: 0.68rem; color: var(--accent-blue); font-weight: bold;">${r.debounce || ''}</span>
                        </div>
                        <button class="btn" style="padding: 2px 6px; font-size: 0.68rem;" onclick="redoRoi(${idx})">Reset</button>
                    </div>
                `;
            });
            list.innerHTML = html;
        }

        let lastLogsCount = 0;
        function renderLogs(logs) {
            if (!logs || logs.length === lastLogsCount) return;
            lastLogsCount = logs.length;
            const box = document.getElementById('logBox');
            box.innerHTML = logs.map(l => `
                <div class="log-entry log-${l.level}">
                    <span style="color: var(--text-muted); font-size: 0.7rem;">[${l.time}]</span> ${l.message}
                </div>
            `).join('');
            box.scrollTop = box.scrollHeight;
        }

        function clearLogUi() {
            document.getElementById('logBox').innerHTML = '';
            lastLogsCount = 0;
        }

        // Interaksi Klik Video untuk Memetakan Koordinat ROI
        function onCanvasClick(e) {
            const img = document.getElementById('streamImg');
            const rect = img.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;

            // Skalakan ke resolusi asli frame video (misal 1280x720 atau 640x480)
            const natW = img.naturalWidth || 1280;
            const natH = img.naturalHeight || 720;
            const scaleX = natW / rect.width;
            const scaleY = natH / rect.height;

            const clickX = Math.round((e.clientX - rect.left) * scaleX);
            const clickY = Math.round((e.clientY - rect.top) * scaleY);

            fetch('/api/click', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: clickX, y: clickY })
            });
        }

        function onCanvasRightClick(e) {
            e.preventDefault();
            triggerAction('undo_roi_point');
        }

        async function triggerAction(actionName) {
            await fetch('/api/action/' + actionName, { method: 'POST' });
        }

        function confirmResetRoi() {
            if (confirm("Apakah Anda yakin ingin menghapus semua ROI hasil kalibrasi dan membuat ulang dari awal?")) {
                triggerAction('reset_roi');
            }
        }

        async function redoRoi(idx) {
            await fetch('/api/action/redo_roi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ roi_idx: idx })
            });
        }

        async function changeRoiName(idx, newName) {
            await fetch('/api/action/set_roi_name', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ roi_idx: idx, name: newName })
            });
        }

        // Kontrol Video Playback
        async function togglePlayPause() {
            await fetch('/api/video/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'toggle' })
            });
        }

        async function stepFrame(delta) {
            await fetch('/api/video/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'step', value: delta })
            });
        }

        function onSliderSeek(val) {
            isSeeking = true;
            fetch('/api/video/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'seek', value: parseInt(val) })
            }).finally(() => { setTimeout(() => { isSeeking = false; }, 200); });
        }

        async function lockCurrentFrame() {
            const mode = prompt("Pilih mode ROI untuk frame referensi ini:\\n'keep' (Gunakan ROI saat ini)\\n'add' (Tambah ROI baru)\\n'reset' (Buat ulang semua)", "keep");
            if (mode && ['keep', 'add', 'reset'].includes(mode.toLowerCase())) {
                await fetch('/api/video/lock_ref', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ roi_mode: mode.toLowerCase() })
                });
            }
        }

        // Modal Settings
        async function openSettings() {
            const [res, resModels] = await Promise.all([
                fetch('/api/config'),
                fetch('/api/models')
            ]);
            if (resModels.ok) {
                const mData = await resModels.json();
                const sel = document.getElementById('cfgYoloModelSelect');
                sel.innerHTML = '<option value="">-- Pilih dari palm_detection/model --</option>';
                (mData.models || []).forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.innerText = m;
                    sel.appendChild(opt);
                });
            }
            if (res.ok) {
                const cfg = await res.json();
                document.getElementById('cfgBackend').value = cfg.hand_backend || 'rtmlib';
                document.getElementById('cfgYoloModel').value = cfg.yolo_model_path || 'yolo26n-pose.pt';
                document.getElementById('cfgYoloModelSelect').value = cfg.yolo_model_path || '';
                document.getElementById('cfgYoloClass').value = cfg.yolo_target_class || 'gloves';
                document.getElementById('cfgTrackMode').value = cfg.hand_tracking_mode || 'full_hand';
                document.getElementById('cfgJumlahRoi').value = cfg.jumlah_roi || 3;
                document.getElementById('cfgUsePixelDiff').value = cfg.use_pixel_diff ? "true" : "false";
                document.getElementById('cfgAmbangPiksel').value = cfg.ambang_piksel || 30;
                document.getElementById('cfgAmbangRasio').value = ((cfg.ambang_rasio || 0.08) * 100).toFixed(1);
                document.getElementById('cfgDebounceIn').value = cfg.debounce_masuk || 6;
                document.getElementById('cfgDebounceOut').value = cfg.debounce_keluar || 12;
                document.getElementById('cfgCropMode').value = (cfg.use_roi_crop === false) ? 'none' : (cfg.roi_crop_mode || 'union');
                const useClahe = (cfg.use_clahe !== undefined) ? cfg.use_clahe : Boolean(cfg.enhance_glove);
                document.getElementById('cfgGloveEnhance').value = useClahe ? "true" : "false";
            }
            document.getElementById('settingsModal').style.display = 'flex';
        }

        function closeSettings() {
            document.getElementById('settingsModal').style.display = 'none';
        }

        async function saveSettings() {
            const cropVal = document.getElementById('cfgCropMode').value;
            const useClaheVal = document.getElementById('cfgGloveEnhance').value === "true";
            const payload = {
                hand_backend: document.getElementById('cfgBackend').value,
                yolo_model_path: document.getElementById('cfgYoloModel').value,
                yolo_target_class: document.getElementById('cfgYoloClass').value || 'gloves',
                hand_tracking_mode: document.getElementById('cfgTrackMode').value,
                jumlah_roi: parseInt(document.getElementById('cfgJumlahRoi').value) || 3,
                use_pixel_diff: document.getElementById('cfgUsePixelDiff').value === "true",
                ambang_piksel: parseInt(document.getElementById('cfgAmbangPiksel').value) || 30,
                ambang_rasio: (parseFloat(document.getElementById('cfgAmbangRasio').value) || 8.0) / 100.0,
                debounce_masuk: parseInt(document.getElementById('cfgDebounceIn').value) || 6,
                debounce_keluar: parseInt(document.getElementById('cfgDebounceOut').value) || 12,
                use_roi_crop: cropVal !== "none",
                roi_crop_mode: cropVal === "none" ? "union" : cropVal,
                use_clahe: useClaheVal,
                enhance_glove: useClaheVal
            };

            await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            closeSettings();
        }
    </script>
</body>
</html>
"""

# =============================================================================
# Route Handlers Flask
# =============================================================================

@app.route("/")
def index():
    """Halaman konsol web dashboard interaktif."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    """Streaming frame video langsung via multipart MJPEG."""
    worker = get_worker()

    def generate_frames():
        blank_frame: Optional[bytes] = None
        while True:
            jpeg = worker.get_latest_jpeg(quality=75)
            if jpeg is None:
                if blank_frame is None:
                    # Buat placeholder hitam dengan pesan status jika kamera belum aktif
                    img = 20 * np.ones((360, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        img,
                        "Menghubungkan ke kamera...",
                        (140, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (120, 160, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    ok, buf = cv2.imencode(".jpg", img)
                    blank_frame = buf.tobytes() if ok else b""

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + blank_frame + b"\r\n"
                )
                time.sleep(0.08)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(0.03)

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry")
def api_telemetry():
    """Endpoint status telemetri real-time (FPS, banner, urutan langkah, status ROI)."""
    worker = get_worker()
    return jsonify(worker.get_latest_telemetry())


@app.route("/api/logs")
def api_logs():
    """Endpoint riwayat event log sistem terbaru."""
    worker = get_worker()
    return jsonify(worker.get_recent_logs(limit=50))


@app.route("/api/models")
def api_models():
    """Mengembalikan daftar file model YOLO yang ada di dalam folder palm_detection/model."""
    ensure_model_dir()
    return jsonify({
        "model_dir": MODEL_DIR,
        "models": list_available_yolo_models(),
    })


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Membaca dan memperbarui parameter konfigurasi AppConfig."""
    worker = get_worker()
    if request.method == "GET":
        from dataclasses import asdict
        cfg_dict = asdict(worker.config)
        cfg_dict["available_models"] = list_available_yolo_models()
        return jsonify(cfg_dict)

    data = request.get_json(force=True, silent=True) or {}
    if "enhance_glove" in data and "use_clahe" not in data:
        data["use_clahe"] = bool(data["enhance_glove"])
    if data.get("roi_crop_mode") == "none":
        data["use_roi_crop"] = False
        data["roi_crop_mode"] = "union"

    for key, val in data.items():
        if hasattr(worker.config, key):
            setattr(worker.config, key, val)

    # Sinkronisasi parameter ke sequencer dan pipeline
    if "jumlah_roi" in data:
        worker.set_jumlah_roi(int(data["jumlah_roi"]))
    if "debounce_masuk" in data:
        worker.sequencer.debounce_masuk = int(data["debounce_masuk"])
    if "debounce_keluar" in data:
        worker.sequencer.debounce_keluar = int(data["debounce_keluar"])
    if "ambang_rasio" in data:
        worker.sequencer.ambang_rasio = float(data["ambang_rasio"])
    if "use_pixel_diff" in data:
        worker.sequencer.use_pixel_diff = bool(data["use_pixel_diff"])

    worker.config.save(_global_config_path)
    worker.log_event.emit("SUCCESS", "Pengaturan sistem berhasil diperbarui dan disimpan.")
    return jsonify({"status": "ok"})


@app.route("/api/click", methods=["POST"])
def api_click():
    """Simulasi klik mouse pada kanvas video untuk menambahkan titik sudut ROI."""
    worker = get_worker()
    data = request.get_json(force=True, silent=True) or {}
    x = int(data.get("x", 0))
    y = int(data.get("y", 0))
    worker.add_roi_point(x, y)
    return jsonify({"status": "ok", "x": x, "y": y})


@app.route("/api/action/<action_name>", methods=["POST"])
def api_action(action_name: str):
    """Memicu aksi kontrol cepat sistem."""
    worker = get_worker()
    data = request.get_json(force=True, silent=True) or {}

    if action_name == "undo_roi_point":
        worker.undo_roi_point()
    elif action_name == "reset_roi":
        worker.trigger_reset_roi()
    elif action_name == "recapture_ref":
        worker.trigger_recapture_ref()
    elif action_name == "clear_status":
        worker.trigger_clear_status()
    elif action_name == "reset_parts":
        worker.trigger_reset_parts()
    elif action_name == "redo_roi":
        roi_idx = int(data.get("roi_idx", 0))
        worker.trigger_redo_roi(roi_idx)
    elif action_name == "set_roi_name":
        roi_idx = int(data.get("roi_idx", 0))
        name = str(data.get("name", "")).strip()
        worker.config.set_roi_name(roi_idx, name)
        worker.config.save(_global_config_path)
        worker.log_event.emit("INFO", f"Nama ROI #{roi_idx + 1} diubah menjadi '{name}'.")
    else:
        return jsonify({"status": "error", "message": f"Aksi tidak dikenal: {action_name}"}), 400

    return jsonify({"status": "ok", "action": action_name})


@app.route("/api/video/control", methods=["POST"])
def api_video_control():
    """Kontrol playback video (play/pause/seek/step)."""
    worker = get_worker()
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")
    val = data.get("value")

    if action == "toggle":
        worker.toggle_video_play()
    elif action == "play":
        worker.set_video_paused(False)
    elif action == "pause":
        worker.set_video_paused(True)
    elif action == "seek" and val is not None:
        worker.seek_video_frame(int(val))
    elif action == "step" and val is not None:
        worker.step_video_frame(int(val))
    else:
        return jsonify({"status": "error", "message": "Perintah video tidak valid"}), 400

    return jsonify({"status": "ok"})


@app.route("/api/video/lock_ref", methods=["POST"])
def api_video_lock_ref():
    """Mengunci frame video saat ini sebagai referensi background."""
    worker = get_worker()
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("roi_mode", "keep")
    worker.lock_current_frame_as_ref(roi_mode=mode)
    return jsonify({"status": "ok", "mode": mode})


# =============================================================================
# CLI Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Palm Detection Headless Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port HTTP (default: 5000)")
    parser.add_argument("--config", type=str, default=CONFIG_FILE, help="Path ke file konfigurasi JSON")
    parser.add_argument("--video", type=str, default=None, help="Buka file video secara otomatis saat startup")
    return parser.parse_args()


def main() -> None:
    global _global_config_path
    args = parse_args()
    _global_config_path = args.config

    worker = get_worker()
    if args.video and os.path.exists(args.video):
        worker.open_video_file(args.video)

    print("=" * 60)
    print(" Palm Detection - Headless Web Server Aktif")
    print("=" * 60)
    print(f" Akses Dashboard Web di browser:")
    print(f"   -> Lokal  : http://localhost:{args.port}")
    print(f"   -> Jaringan: http://<IP_JETSON>:{args.port}")
    print(f" Port: {args.port} | Host: {args.host}")
    print(" Tekan Ctrl+C untuk mematikan server.")
    print("=" * 60)

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        cleanup_worker()


if __name__ == "__main__":
    main()
