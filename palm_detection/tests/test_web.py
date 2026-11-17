"""Unit & Integration Test Suite untuk modul web.py (Headless Web Console)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import web
from palm_detection.config import AppConfig


class TestWebHeadlessConsole(unittest.TestCase):
    """Pengujian komprehensif endpoint Web Console Flask."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_cfg_path = os.path.join(cls.temp_dir, "test_config.json")
        web._global_config_path = cls.test_cfg_path
        cls.client = web.app.test_client()
        cls.worker = web.get_worker()

    @classmethod
    def tearDownClass(cls):
        web.cleanup_worker()
        if os.path.exists(cls.test_cfg_path):
            try:
                os.remove(cls.test_cfg_path)
            except Exception:
                pass

    def test_index_page_returns_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Palm Detection Web Console", res.get_data(as_text=True))
        self.assertIn("/video_feed", res.get_data(as_text=True))

    def test_video_feed_stream(self):
        res = self.client.get("/video_feed")
        self.assertEqual(res.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", res.headers.get("Content-Type", ""))
        # Baca satu chunk awal untuk memastikan generator berjalan tanpa error
        chunk = next(res.response)
        self.assertIn(b"--frame", chunk)
        self.assertIn(b"Content-Type: image/jpeg", chunk)

    def test_api_telemetry_endpoint(self):
        res = self.client.get("/api/telemetry")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsInstance(data, dict)

    def test_api_logs_endpoint(self):
        self.worker.log_event.emit("INFO", "Log uji web console")
        res = self.client.get("/api/logs")
        self.assertEqual(res.status_code, 200)
        logs = res.get_json()
        self.assertIsInstance(logs, list)
        self.assertTrue(any("Log uji web console" in l.get("message", "") for l in logs))

    def test_api_config_get_and_post(self):
        # 1. GET Config
        res_get = self.client.get("/api/config")
        self.assertEqual(res_get.status_code, 200)
        cfg_data = res_get.get_json()
        self.assertIn("ambang_piksel", cfg_data)
        self.assertIn("jumlah_roi", cfg_data)

        # 2. POST Config update
        res_post = self.client.post("/api/config", json={
            "ambang_piksel": 48,
            "debounce_masuk": 9,
            "enhance_glove": True,
            "roi_crop_mode": "none",
        })
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(self.worker.config.ambang_piksel, 48)
        self.assertEqual(self.worker.config.debounce_masuk, 9)
        self.assertTrue(self.worker.config.use_clahe)
        self.assertFalse(self.worker.config.use_roi_crop)

        # 3. GET Models list from palm_detection/model
        res_models = self.client.get("/api/models")
        self.assertEqual(res_models.status_code, 200)
        m_data = res_models.get_json()
        self.assertIn("models", m_data)
        self.assertIsInstance(m_data["models"], list)

    def test_api_click_endpoint(self):
        res = self.client.post("/api/click", json={"x": 120, "y": 240})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("x"), 120)
        self.assertEqual(data.get("y"), 240)

    def test_api_actions_and_reset_parts(self):
        # Uji reset parts counter
        self.worker.sequencer.parts_completed = 5
        res = self.client.post("/api/action/reset_parts")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.worker.sequencer.parts_completed, 0)

        # Uji clear status
        res = self.client.post("/api/action/clear_status")
        self.assertEqual(res.status_code, 200)

        # Uji aksi tidak valid -> 400
        res_err = self.client.post("/api/action/unknown_action_xyz")
        self.assertEqual(res_err.status_code, 400)


if __name__ == "__main__":
    unittest.main()
