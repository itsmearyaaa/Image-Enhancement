"""Alias tipe yang dipakai bersama oleh seluruh modul."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Gambar 8-bit: BGR berbentuk (tinggi, lebar, 3) atau grayscale (tinggi, lebar), sesuai OpenCV.
Image = npt.NDArray[np.uint8]
#: Gambar grayscale 8-bit berbentuk (tinggi, lebar).
Gray = npt.NDArray[np.uint8]
#: Tabel pemetaan intensitas (look-up table) 256 elemen: ``s = lut[r]``.
Lut = npt.NDArray[np.uint8]
#: Histogram: jumlah piksel untuk tiap tingkat keabuan.
Histogram = npt.NDArray[np.int64]
