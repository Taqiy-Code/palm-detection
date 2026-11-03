"""Palm Detection - sistem verifikasi urutan pengambilan komponen berbasis pelacakan tangan.

Struktur modul:
    config     : AppConfig (dataclass murni)
    states     : Enum status/state (murni)
    sequencer  : StepSequencer, state machine urutan langkah (murni, tanpa OpenCV/Qt)
    vision     : Geometri ROI, analisis perubahan piksel, referensi background (OpenCV)
    roi        : RoiManager, penyuntingan poligon ROI (OpenCV)
    detection  : HandDetector (RTMlib / MediaPipe) dan pipeline crop + preprocessing sarung tangan
    renderer   : OverlayRenderer, hanya menggambar overlay
    camera     : CameraSource, pemilik tunggal cv2.VideoCapture di thread grabber sendiri
    worker     : CameraWorker (QThread), orkestrasi seluruh komponen di atas
    gui        : Widget PyQt6 dan MainWindow
"""
