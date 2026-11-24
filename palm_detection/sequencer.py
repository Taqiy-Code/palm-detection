"""State machine urutan langkah perakitan (murni, tanpa OpenCV / PyQt5).

Kelas `StepSequencer` mudah diuji secara modular dengan unit-test tanpa membutuhkan
lingkungan grafis atau perangkat keras kamera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from palm_detection.states import BannerState, LogLevel, RoiState, StepStatus


@dataclass
class SequencerEvent:
    level: LogLevel
    message: str


@dataclass
class StepTelemetryItem:
    index: int
    state: RoiState
    label: str
    ratio: float
    debounce: str


class StepSequencer:
    """State machine urutan verifikasi langkah (Poka-Yoke)."""

    def __init__(
        self,
        num_steps: int = 3,
        debounce_masuk: int = 3,
        debounce_keluar: int = 8,
        ambang_rasio: float = 0.08,
        use_pixel_diff: bool = True,
    ) -> None:
        self.num_steps = max(1, num_steps)
        self.debounce_masuk = max(1, debounce_masuk)
        self.debounce_keluar = max(1, debounce_keluar)
        self.ambang_rasio = float(ambang_rasio)
        self.use_pixel_diff = bool(use_pixel_diff)

        self.statuses: List[StepStatus] = []
        self.masuk_counts: List[int] = []
        self.keluar_counts: List[int] = []
        self.error_counts: List[int] = []
        self.reentry_counts: List[int] = []
        self.last_ratios: List[float] = []

        self.parts_completed: int = 0
        self.cycle_cooldown_until: Optional[float] = None
        self.last_error_log_time: float = 0.0

        self.reset(self.num_steps)

    def reset(self, num_steps: Optional[int] = None) -> None:
        """Mereset seluruh status langkah dan hitungan debounce."""
        if num_steps is not None:
            self.num_steps = max(1, num_steps)
        n = self.num_steps
        self.statuses = [StepStatus.WAITING] * n
        self.masuk_counts = [0] * n
        self.keluar_counts = [0] * n
        self.error_counts = [0] * n
        self.reentry_counts = [0] * n
        self.last_ratios = [0.0] * n
        self.cycle_cooldown_until = None

    def clear_statuses(self) -> None:
        """Mereset status langkah ke WAITING (memulai siklus baru) tanpa mereset parts_completed."""
        n = self.num_steps
        self.statuses = [StepStatus.WAITING] * n
        self.masuk_counts = [0] * n
        self.keluar_counts = [0] * n
        self.error_counts = [0] * n
        self.reentry_counts = [0] * n
        self.cycle_cooldown_until = None

    def reset_parts(self) -> None:
        """Mereset counter part selesai ke 0."""
        self.parts_completed = 0

    def ensure_size(self, size: int) -> None:
        """Memastikan panjang list status mencukupi jika jumlah ROI bertambah."""
        needed = max(self.num_steps, size, 1)
        curr = len(self.statuses)
        if curr < needed:
            diff = needed - curr
            self.statuses.extend([StepStatus.WAITING] * diff)
            self.masuk_counts.extend([0] * diff)
            self.keluar_counts.extend([0] * diff)
            self.error_counts.extend([0] * diff)
            self.reentry_counts.extend([0] * diff)
            self.last_ratios.extend([0.0] * diff)

    def get_active_step(self) -> Optional[int]:
        """Mengembalikan index langkah yang saat ini harus dikerjakan, atau None jika semua DONE."""
        total = min(len(self.statuses), self.num_steps)
        for i in range(total):
            if self.statuses[i] != StepStatus.DONE:
                return i
        return None

    def get_steps_completed(self) -> int:
        """Menghitung jumlah langkah yang sudah berstatus DONE dalam siklus saat ini."""
        total = min(len(self.statuses), self.num_steps)
        return sum(1 for i in range(total) if self.statuses[i] == StepStatus.DONE)

    def is_cycle_completed(self) -> bool:
        """Mengembalikan True jika seluruh langkah dalam siklus telah selesai."""
        return self.get_active_step() is None

    def process_step_entry(self, step_idx: int, hand_in_roi: bool) -> Tuple[bool, str]:
        """Memproses debounce saat tangan masuk ke ROI aktif."""
        if step_idx >= len(self.statuses):
            return False, ""
        debounce_info = ""
        if self.statuses[step_idx] == StepStatus.WAITING:
            if hand_in_roi:
                self.masuk_counts[step_idx] += 1
            else:
                self.masuk_counts[step_idx] = 0

            if self.masuk_counts[step_idx] > 0:
                debounce_info = f"Masuk: {self.masuk_counts[step_idx]}/{self.debounce_masuk}"

            if self.masuk_counts[step_idx] >= self.debounce_masuk:
                self.statuses[step_idx] = StepStatus.HAND_IN
                self.keluar_counts[step_idx] = 0
                return True, debounce_info

        return False, debounce_info

    def process_step_exit(
        self, step_idx: int, hand_or_motion_present: bool
    ) -> Tuple[bool, str]:
        """Memproses debounce saat tangan keluar dari ROI aktif.

        Mengembalikan (should_evaluate_diff, debounce_info).
        """
        if step_idx >= len(self.statuses):
            return False, ""
        debounce_info = ""
        if self.statuses[step_idx] == StepStatus.HAND_IN:
            if not hand_or_motion_present:
                self.keluar_counts[step_idx] += 1
            else:
                self.keluar_counts[step_idx] = 0

            debounce_info = f"Keluar: {self.keluar_counts[step_idx]}/{self.debounce_keluar}"
            if self.keluar_counts[step_idx] >= self.debounce_keluar:
                return True, debounce_info

        return False, debounce_info

    def finalize_step_exit(
        self, step_idx: int, diff_ratio: Optional[float], step_name: str
    ) -> Tuple[bool, SequencerEvent]:
        """Memvalidasi penyelesaian langkah setelah debounce keluar tercapai."""
        if not self.use_pixel_diff or diff_ratio is None:
            self.last_ratios[step_idx] = 0.0
            self.statuses[step_idx] = StepStatus.DONE
            self.masuk_counts[step_idx] = 0
            event = SequencerEvent(
                LogLevel.SUCCESS,
                f"Langkah {step_idx + 1} ({step_name}) Selesai (Debounce OK)",
            )
            return True, event

        self.last_ratios[step_idx] = diff_ratio
        if diff_ratio >= self.ambang_rasio:
            self.statuses[step_idx] = StepStatus.DONE
            self.masuk_counts[step_idx] = 0
            event = SequencerEvent(
                LogLevel.SUCCESS,
                f"Langkah {step_idx + 1} ({step_name}) Selesai: Diff {diff_ratio * 100:.1f}%",
            )
            return True, event
        else:
            self.statuses[step_idx] = StepStatus.WAITING
            self.masuk_counts[step_idx] = 0
            event = SequencerEvent(
                LogLevel.WARN,
                f"Langkah {step_idx + 1} ({step_name}) ditolak: Diff {diff_ratio * 100:.1f}% < target {self.ambang_rasio * 100:.1f}%",
            )
            return False, event

    def process_locked_step(
        self, step_idx: int, hand_in_roi: bool, debounce_err: int = 2
    ) -> bool:
        """Memproses debounce urutan salah (tangan di ROI terkunci)."""
        if step_idx >= len(self.error_counts):
            return False
        if hand_in_roi:
            self.error_counts[step_idx] += 1
        else:
            self.error_counts[step_idx] = 0
        return self.error_counts[step_idx] >= max(1, debounce_err)

    def process_done_step_reentry(
        self, step_idx: int, hand_in_roi: bool, debounce_reentry: int = 3
    ) -> bool:
        """Memproses debounce akses ulang (double-pick) pada ROI yang sudah berstatus DONE."""
        if step_idx >= len(self.reentry_counts):
            return False
        if self.statuses[step_idx] != StepStatus.DONE:
            self.reentry_counts[step_idx] = 0
            return False
        if hand_in_roi:
            self.reentry_counts[step_idx] += 1
        else:
            self.reentry_counts[step_idx] = 0
        return self.reentry_counts[step_idx] >= max(1, debounce_reentry)

    def clear_error_counts(self) -> None:
        """Mengosongkan counter error dan counter akses ulang seluruh ROI."""
        for i in range(len(self.error_counts)):
            self.error_counts[i] = 0
        for i in range(len(self.reentry_counts)):
            self.reentry_counts[i] = 0

    def reset_active_step_debounce(self, step_idx: Optional[int] = None) -> None:
        """Mereset debounce dan status langkah aktif ke WAITING (0) saat terjadi jumping proses."""
        target = self.get_active_step() if step_idx is None else step_idx
        if target is None or not (0 <= target < len(self.statuses)):
            return
        if self.statuses[target] != StepStatus.DONE:
            self.statuses[target] = StepStatus.WAITING
            self.masuk_counts[target] = 0
            self.keluar_counts[target] = 0

    def handle_cycle_completion(
        self, now: float, cooldown_seconds: float = 1.2
    ) -> Tuple[bool, bool, Optional[SequencerEvent]]:
        """Mengelola transisi auto-increment part dan auto-cycle cooldown.

        Mengembalikan:
            (is_done_cycle, cooldown_expired, event)
        """
        if not self.is_cycle_completed():
            self.cycle_cooldown_until = None
            return False, False, None

        if self.cycle_cooldown_until is None:
            self.cycle_cooldown_until = now + cooldown_seconds
            self.parts_completed += 1
            event = SequencerEvent(
                LogLevel.PART,
                f"PART #{self.parts_completed} SELESAI ({self.num_steps}/{self.num_steps} Langkah)",
            )
            return True, False, event

        if now >= self.cycle_cooldown_until:
            return True, True, None

        return True, False, None
