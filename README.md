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

Proyek ini telah dikonfigurasi dengan virtual environment `.venv` terisolasi berbasis Python 3.12.

### 1. Aktivasi Virtual Environment
Di terminal PowerShell / Command Prompt:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Atau di CMD biasa: `.venv\Scripts\activate.bat`)*

### 2. Menjalankan Aplikasi
Setelah venv aktif:
```bash
python main.py
```
Atau langsung jalankan skrip peluncur:
```cmd
run.bat
```

---

## 📦 Dependensi Utama

- `PyQt5`: Antarmuka GUI desktop
- `opencv-contrib-python`: Pengolahan citra dan akuisisi kamera
- `rtmlib`: Framework pelacakan tangan RTMPose berbasis deep learning
- `onnxruntime`: Engine inferensi model AI
- `mediapipe`: Framework pelacakan tangan alternatif dari Google
- `numpy`: Komputasi matriks dan array numerik
