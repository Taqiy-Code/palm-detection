# Deteksi Komponen Bulb Kendaraan

Sistem otomasi inspeksi dan panduan perakitan komponen bulb kendaraan berbasis visi komputer (*computer vision*) dan pelacakan tangan cerdas (*hand tracking*).

Aplikasi ini menggunakan antarmuka grafis desktop modern berbasis **PyQt5** yang mengintegrasikan akuisisi kamera real-time, segmentasi area kerja (ROI Polygon), deteksi perubahan piksel (*background subtraction*), serta algoritma AI untuk melacak posisi tangan pekerja (Dual-Backend: RTMlib RTMPose dan Google MediaPipe).

---

## 🚀 Fitur Utama

1. **Dual-Backend AI Hand Tracking**:
   - **RTMlib (RTMPose Hand)**: Pelacakan sendi tangan berperforma tinggi dengan backend ONNX Runtime, sangat akurat mendeteksi kontur tangan dan posisi sendi.
   - **Google MediaPipe Hands**: Pilihan alternatif untuk pelacakan 21 keypoint tangan secara efisien.
2. **Optimasi Sarung Tangan Industri (CLAHE & Pseudo-Skin Tone)**:
   - Penggunaan CLAHE (*Contrast Limited Adaptive Histogram Equalization*) dan *Look-Up Table (LUT)* untuk mempertegas lipatan sendi dan mengatasi sarung tangan gelap/berwarna.
3. **ROI Poligon Fleksibel & Interaktif**:
   - Pengguna dapat membuat area inspeksi dengan mengklik 4 titik sudut di video.
   - Sudut ROI dapat digeser (*drag-and-drop*) langsung di antarmuka kanvas secara live.
   - Pangkas area deteksi (*ROI Cropping*) untuk akselerasi FPS.
4. **Validasi Perakitan & Urutan Kerja**:
   - Fitur *debounce* masuk dan keluar untuk memastikan tangan benar-benar menyelesaikan pemasangan sebelum lanjut ke langkah berikutnya.
   - Pengecekan urutan kerja dengan alarm visual jika urutan salah (*Poka-Yoke*).
   - Penghitung jumlah part yang telah selesai (*Parts Completed*).
5. **Kontrol Hardware Kamera**:
   - Kontrol langsung untuk exposure, focus, white balance, dan gain melalui OpenCV DirectShow.

---

## 🛠️ Instalasi & Persiapan Lingkungan (Virtual Environment)

Proyek ini mendukung Python >= 3.10 (termasuk kompatibel dengan Python 3.10 pada Nvidia Jetson / Ubuntu 22.04).

### 1. Aktivasi Virtual Environment
Di terminal PowerShell / Command Prompt:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Atau di CMD biasa: `.venv\Scripts\activate.bat`)*

### 2. Menjalankan Aplikasi

#### A. Mode Desktop GUI (PyQt5)
Jika perangkat terhubung ke monitor / display X11:
```bash
python main.py
```
Atau di Windows:
```cmd
run.bat
```

#### B. Mode Headless Web Console (Nvidia Jetson / Tanpa Monitor)
Jika perangkat berjalan tanpa layar monitor (*headless edge server*):
```bash
python3 web.py --host 0.0.0.0 --port 5000
```
Atau via skrip peluncur:
```bash
chmod +x run_web.sh
./run_web.sh
```
Buka browser di laptop atau HP yang terhubung ke jaringan lokal:
```text
http://<IP_JETSON>:5000
```
Fitur Web Console:
- Live streaming video MJPEG dengan overlay ROI & deteksi tangan real-time.
- Kalibrasi & pembuatan poligon ROI langsung dengan mengklik 4 titik pada layar video di browser.
- Monitoring telemetri: Counter Part Selesai, Progres Langkah Poka-Yoke, Status Banner, dan Log Aktivitas.
- Modal Pengaturan lengkap untuk pergantian backend AI (RTMlib, MediaPipe, YOLO), threshold piksel, debounce, dan hardware kamera.

---

## 📦 Dependensi Utama

- `PyQt5`: Antarmuka GUI desktop
- `Flask`: Antarmuka Web Dashboard & Streaming video untuk pengoperasian Headless
- `opencv-contrib-python-headless`: Pengolahan citra dan akuisisi kamera (bebas konflik GUI Qt pada Linux/Jetson)
- `rtmlib`: Framework pelacakan tangan RTMPose berbasis deep learning
- `onnxruntime`: Engine inferensi model AI
- `mediapipe`: Framework pelacakan tangan alternatif dari Google
- `numpy`: Komputasi matriks dan array numerik
