"""Peningkatan citra real-time dari webcam.

Tombol: ``n`` / ``p`` ganti operasi, ``+`` / ``-`` atur parameter utama, ``r`` kembalikan
parameter bawaan, ``h`` tampil / sembunyikan histogram, ``c`` bandingkan dengan gambar asli,
``s`` simpan, ``q`` atau ESC keluar.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2

from ._types import Image
from .errors import CameraError
from .imageio import write_image
from .operations import EnhanceResult, Operation, get_operation, list_operations
from .render import draw_live_overlay, side_by_side

logger = logging.getLogger(__name__)

WINDOW_TITLE = "ImgEnhance - Webcam"
KEY_ESC = 27
HINT = "n/p operasi  +/- atur  r reset  h histogram  c banding  s simpan  q keluar"


@dataclass
class LiveState:
    """Keadaan tampilan webcam: operasi aktif, parameternya, dan opsi tampilan."""

    operations: list[Operation] = field(default_factory=list_operations)
    index: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    show_histogram: bool = True
    compare: bool = False

    def __post_init__(self) -> None:
        if not self.params:
            self.params = self.operation.defaults()

    @classmethod
    def for_operation(cls, name: str) -> LiveState:
        """Mulai dari operasi ``name`` (``OperationError`` bila tidak dikenal)."""
        target = get_operation(name)
        operations = list_operations()
        return cls(operations=operations, index=operations.index(target))

    @property
    def operation(self) -> Operation:
        return self.operations[self.index]

    def cycle(self, direction: int) -> None:
        """Pindah ke operasi berikutnya (+1) atau sebelumnya (-1); parameter kembali bawaan."""
        self.index = (self.index + direction) % len(self.operations)
        self.params = self.operation.defaults()

    def reset(self) -> None:
        self.params = self.operation.defaults()

    def adjust(self, direction: int) -> None:
        """Ubah parameter utama satu langkah (``direction`` = +1 / -1) dalam batas rentangnya."""
        param = self.operation.primary
        if param is None:
            return
        step = param.step or 1
        value = self.params[param.name] + direction * step
        if param.minimum is not None:
            value = max(param.minimum, value)
        if param.maximum is not None:
            value = min(param.maximum, value)
        self.params[param.name] = int(value) if param.kind == "int" else round(float(value), 4)

    def detail(self) -> str:
        shown = [
            f"{p.name}={self.params[p.name]}"
            for p in self.operation.params
            if p.when is None or self.params.get(p.when[0]) == p.when[1]
        ]
        return ", ".join(shown) if shown else "(tanpa parameter)"


def process_frame(frame: Image, state: LiveState) -> tuple[Image, EnhanceResult]:
    """Terapkan operasi aktif pada ``frame``; kembalikan (tampilan beranotasi, hasil). Tanpa I/O."""
    result = state.operation.apply(frame, state.params)
    shown = side_by_side(frame, result.image) if state.compare else result.image
    title = f"{state.operation.title}  [{state.index + 1}/{len(state.operations)}]"
    display = draw_live_overlay(
        shown,
        title,
        state.detail(),
        result.image if state.show_histogram else None,
        HINT,
    )
    return display, result


def run_webcam(
    operation: str = "equalize",
    camera_index: int = 0,
    output_dir: Path = Path("output"),
) -> None:
    """Jalankan loop webcam sampai pengguna menekan q / ESC.

    Raises
    ------
    CameraError : kamera tidak dapat dibuka atau frame gagal dibaca.
    OperationError : ``operation`` tidak dikenal.
    """
    state = LiveState.for_operation(operation)  # validasi nama sebelum kamera dibuka
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened():
        capture.release()
        raise CameraError(f"Kamera dengan indeks {camera_index} tidak bisa dibuka.")

    logger.info("Webcam aktif. %s", HINT)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise CameraError("Gagal membaca frame dari kamera.")
            mirrored = cast(Image, cv2.flip(frame, 1))  # efek cermin agar terasa natural
            display, _ = process_frame(mirrored, state)

            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), KEY_ESC):
                break
            if key == ord("s"):
                target = (
                    output_dir
                    / f"webcam_{state.operation.name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                )
                write_image(target, display)
                logger.info("Disimpan: %s", target)
            elif key == ord("n"):
                state.cycle(+1)
            elif key == ord("p"):
                state.cycle(-1)
            elif key in (ord("+"), ord("=")):
                state.adjust(+1)
            elif key in (ord("-"), ord("_")):
                state.adjust(-1)
            elif key == ord("r"):
                state.reset()
            elif key == ord("h"):
                state.show_histogram = not state.show_histogram
            elif key == ord("c"):
                state.compare = not state.compare
    finally:
        capture.release()
        with contextlib.suppress(cv2.error):  # OpenCV tanpa dukungan GUI (build headless)
            cv2.destroyAllWindows()
