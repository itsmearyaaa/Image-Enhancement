"""Operasi aras lokal (*mask operation*): nilai baru piksel dihitung dari tetangganya.

Pelengkap operasi titik pada :mod:`imgenhance.point`. Materi P03 membedakan aras titik, lokal, dan
global (hlm. 3-4); filter di bawah ini mewakili aras lokal: penghalusan (mengurangi *noise*),
*median filtering* (noise garam-lada), dan penajaman (mengurangi *blur*).
"""

from __future__ import annotations

from typing import cast

import cv2

from ._types import Image
from .errors import ParameterError
from .point import check_image

SMOOTH_METHODS = ("mean", "gaussian")


def _check_kernel(ksize: int) -> None:
    if not isinstance(ksize, int) or ksize < 3 or ksize % 2 == 0:
        raise ParameterError(f"Ukuran kernel harus bilangan ganjil >= 3, bukan {ksize!r}")


def smooth(image: Image, ksize: int = 5, method: str = "gaussian") -> Image:
    """Penghalusan (*smoothing*): rata-rata (``mean``) atau Gaussian terhadap tetangga ``ksize``."""
    check_image(image)
    _check_kernel(ksize)
    if method == "mean":
        return cast(Image, cv2.blur(image, (ksize, ksize)))
    if method == "gaussian":
        return cast(Image, cv2.GaussianBlur(image, (ksize, ksize), 0))
    raise ParameterError(f"Metode smoothing '{method}' tidak dikenal; pilih: mean, gaussian")


def median_filter(image: Image, ksize: int = 3) -> Image:
    """*Median filtering*: piksel diganti median tetangganya (efektif untuk noise garam-lada)."""
    check_image(image)
    _check_kernel(ksize)
    return cast(Image, cv2.medianBlur(image, ksize))


def sharpen(image: Image, amount: float = 1.0, ksize: int = 5) -> Image:
    """Penajaman *unsharp mask*: ``hasil = asli + amount * (asli - blur(asli))``."""
    check_image(image)
    _check_kernel(ksize)
    if amount < 0:
        raise ParameterError(f"Besar penajaman tidak boleh negatif: {amount}")
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return cast(Image, cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0))
