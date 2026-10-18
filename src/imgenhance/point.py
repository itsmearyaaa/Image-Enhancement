"""Operasi titik (*point operation*): setiap piksel diubah sendiri-sendiri, tanpa tetangga.

Hampir semua operasi pada materi P03 berbentuk ``s = T(r)``: nilai keluaran ``s`` hanya bergantung
pada nilai masukan ``r`` milik piksel itu sendiri. Karena itu setiap operasi di sini dinyatakan
sebagai **look-up table (LUT)** 256 elemen, lalu diterapkan sekaligus ke seluruh gambar oleh
``cv2.LUT``. Keuntungannya: rumus ``T`` terlihat jelas (dan bisa digambar sebagai kurva), hasilnya
konsisten untuk gambar grayscale maupun berwarna, dan cukup cepat untuk webcam real-time.

Rumus (r = intensitas masukan 0..255, s = intensitas keluaran):

==============  =======================================  ==========================
Operasi         Rumus                                    Fungsi
==============  =======================================  ==========================
Negasi          ``s = 255 - r``                          :func:`negative_lut`
Brightness      ``s = r + t``                            :func:`brightness_lut`
Kontras         ``s = G * (r - P) + P``                  :func:`contrast_lut`
Gamma (power)   ``s = 255 * (r / 255) ** gamma``         :func:`gamma_lut`
Log             ``s = c * log(1 + r)``                   :func:`log_lut`
Stretching      linear per bagian (r1,s1)-(r2,s2)        :func:`stretch_lut`
Threshold       ``s = 255 jika r >= T, selain itu 0``    :func:`threshold_lut`
m-bit           ``s = langkah * floor(r / langkah)``     :func:`quantize_lut`
Gray-level      rentang [A, B] disorot                   :func:`slice_lut`
Bit-plane       ``s = 255 jika bit ke-k bernilai 1``     :func:`bitplane_lut`
==============  =======================================  ==========================
"""

from __future__ import annotations

import math
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from ._types import Gray, Image, Lut
from .errors import ParameterError

LEVELS = 256
MAX_LEVEL = LEVELS - 1

#: Bobot (a_r, a_g, a_b) untuk RGB -> grayscale ``x = a_r*r + a_g*g + a_b*b`` dengan jumlah 1.
GRAY_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "average": (1 / 3, 1 / 3, 1 / 3),  # x = (r + g + b) / 3, rumus pada materi
    "luminance": (0.299, 0.587, 0.114),  # bobot ITU-R BT.601 (sama dengan cv2.COLOR_BGR2GRAY)
}

_R: npt.NDArray[np.float64] = np.arange(LEVELS, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Utilitas
# --------------------------------------------------------------------------- #
def _lut(values: npt.ArrayLike) -> Lut:
    """Bulatkan ke bilangan bulat terdekat dan batasi ke 0..255 (clipping)."""
    rounded = np.rint(np.asarray(values, dtype=np.float64))
    return cast(Lut, np.clip(rounded, 0, MAX_LEVEL).astype(np.uint8))


def _check_level(name: str, value: float) -> None:
    if not 0 <= value <= MAX_LEVEL:
        raise ParameterError(f"{name} harus di antara 0 dan {MAX_LEVEL}, bukan {value}")


def is_gray(image: Image) -> bool:
    """True bila ``image`` grayscale (2 dimensi)."""
    return image.ndim == 2


def check_image(image: Image) -> None:
    if image.dtype != np.uint8:
        raise ParameterError(f"Gambar harus 8-bit (uint8), bukan {image.dtype}")
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 3)):
        return
    raise ParameterError(f"Bentuk gambar tidak didukung: {image.shape}")


# --------------------------------------------------------------------------- #
# Konversi warna
# --------------------------------------------------------------------------- #
def to_grayscale(
    image: Image,
    method: str = "luminance",
    weights: tuple[float, float, float] | None = None,
) -> Gray:
    """Ubah RGB ke grayscale: ``x = a_r*r + a_g*g + a_b*b`` dengan ``a_r + a_g + a_b = 1``.

    Parameters
    ----------
    method : ``"average"`` (``(r+g+b)/3``) atau ``"luminance"`` (0.299 / 0.587 / 0.114).
    weights : bobot kustom ``(a_r, a_g, a_b)``; bila diberikan, ``method`` diabaikan.
    """
    check_image(image)
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 1:
        return image[:, :, 0].copy()

    if weights is None:
        if method not in GRAY_WEIGHTS:
            raise ParameterError(
                f"Metode grayscale '{method}' tidak dikenal; pilih: {', '.join(GRAY_WEIGHTS)}"
            )
        weights = GRAY_WEIGHTS[method]
    if min(weights) < 0 or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ParameterError(f"Bobot grayscale harus tidak negatif dan berjumlah 1: {weights}")

    a_r, a_g, a_b = weights
    b, g, r = (image[:, :, i].astype(np.float64) for i in range(3))
    return _lut(a_r * r + a_g * g + a_b * b)


def luminance(image: Image) -> Gray:
    """Kanal luminans (Y pada YCrCb) dari gambar berwarna; grayscale dikembalikan apa adanya.

    Luminans dipakai untuk operasi yang bergantung pada isi gambar (histogram equalization,
    contrast stretching) supaya *hue* warna tidak bergeser.
    """
    check_image(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cast(Gray, cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0])


def as_bgr(image: Image) -> Image:
    """Pastikan gambar 3 kanal BGR (grayscale diperluas menjadi tiga kanal sama)."""
    check_image(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    gray = image if image.ndim == 2 else image[:, :, 0]
    return cast(Image, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


# --------------------------------------------------------------------------- #
# Penerapan LUT
# --------------------------------------------------------------------------- #
def apply_lut(image: Image, lut: Lut, *, luminance_only: bool = False) -> Image:
    """Terapkan ``lut`` ke setiap piksel.

    Gambar berwarna: LUT yang sama dipakai untuk tiap kanal B, G, R, atau - bila
    ``luminance_only`` - hanya untuk luminans (Y) sehingga warna (hue) tidak bergeser.
    """
    check_image(image)
    if lut.shape != (LEVELS,) or lut.dtype != np.uint8:
        raise ParameterError("LUT harus berupa array uint8 berisi 256 elemen")
    if image.ndim == 2 or image.shape[2] == 1 or not luminance_only:
        return cast(Image, cv2.LUT(image, lut))
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.LUT(np.ascontiguousarray(ycrcb[:, :, 0]), lut)
    return cast(Image, cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR))


# --------------------------------------------------------------------------- #
# Pembangun LUT
# --------------------------------------------------------------------------- #
def identity_lut() -> Lut:
    """Transformasi identitas ``s = r`` (gambar tidak berubah)."""
    return _lut(_R)


def negative_lut() -> Lut:
    """Negasi citra: ``s = 255 - r`` (hasilnya seperti klise foto)."""
    return _lut(MAX_LEVEL - _R)


def brightness_lut(offset: float) -> Lut:
    """Pengaturan brightness: ``s = r + t``; ``t`` boleh positif (lebih terang) atau negatif."""
    if not -MAX_LEVEL <= offset <= MAX_LEVEL:
        raise ParameterError(f"Perubahan brightness harus di antara -255 dan 255, bukan {offset}")
    return _lut(_R + offset)


def contrast_lut(gain: float, center: float = 128.0) -> Lut:
    """Pengaturan kontras: ``s = G * (r - P) + P``.

    ``G`` adalah koefisien penguatan kontras (``G > 1`` menaikkan, ``G < 1`` menurunkan kontras)
    dan ``P`` adalah pusat pengontrasan. Contoh pada materi: ``G = 2``, ``P = 10``.
    """
    if gain < 0:
        raise ParameterError(f"Penguatan kontras tidak boleh negatif: {gain}")
    _check_level("Pusat kontras", center)
    return _lut(gain * (_R - center) + center)


def gamma_lut(gamma: float) -> Lut:
    """Koreksi gamma / power-law: ``s = 255 * (r / 255) ** gamma``.

    ``gamma < 1`` mencerahkan (memekarkan area gelap), ``gamma > 1`` menggelapkan.
    """
    if gamma <= 0:
        raise ParameterError(f"Gamma harus lebih besar dari 0, bukan {gamma}")
    return _lut(MAX_LEVEL * (_R / MAX_LEVEL) ** gamma)


def log_lut() -> Lut:
    """Transformasi logaritma: ``s = c * log(1 + r)`` dengan ``c = 255 / log(256)``.

    Memekarkan nilai gelap dan memampatkan nilai terang; ``c`` dipilih agar 255 tetap 255.
    """
    return _lut((MAX_LEVEL / math.log(LEVELS)) * np.log1p(_R))


def stretch_lut(r1: float, s1: float, r2: float, s2: float) -> Lut:
    """Peregangan kontras linear per bagian lewat titik ``(r1, s1)`` dan ``(r2, s2)``.

    Fungsi menghubungkan ``(0, 0) - (r1, s1) - (r2, s2) - (255, 255)`` dengan garis lurus.
    Kasus khusus (materi hlm. 24):

    * ``r1 = s1`` dan ``r2 = s2``: fungsi linear (gambar tidak berubah);
    * ``r1 = r2``, ``s1 = 0``, ``s2 = 255``: fungsi ambang batas (thresholding);
    * ``(r1, s1) = (rmin, 0)`` dan ``(r2, s2) = (rmax, 255)``: peregangan Min-Max.
    """
    for name, value in (("r1", r1), ("s1", s1), ("r2", r2), ("s2", s2)):
        _check_level(name, value)
    if r1 > r2:
        raise ParameterError(f"r1 ({r1}) tidak boleh lebih besar dari r2 ({r2})")
    if s1 > s2:
        raise ParameterError(f"s1 ({s1}) tidak boleh lebih besar dari s2 ({s2})")

    out = np.empty(LEVELS, dtype=np.float64)
    low, high = r1 > _R, r2 < _R
    mid = ~(low | high)
    out[low] = s1 / r1 * _R[low] if r1 > 0 else 0.0
    if r2 > r1:
        out[mid] = s1 + (s2 - s1) * (_R[mid] - r1) / (r2 - r1)
    else:  # r1 == r2: fungsi ambang batas
        out[mid] = s2
    out[high] = s2 + (MAX_LEVEL - s2) * (_R[high] - r2) / (MAX_LEVEL - r2) if r2 < MAX_LEVEL else 0
    return _lut(out)


def threshold_lut(threshold: float) -> Lut:
    """Thresholding: ``s = 255`` jika ``r >= T``, selain itu ``s = 0`` (citra biner)."""
    _check_level("Threshold", threshold)
    return _lut(np.where(threshold <= _R, MAX_LEVEL, 0))


def quantize_lut(bits: int, *, rescale: bool = False) -> Lut:
    """Grayscale ke *m-bit*: kuantisasi menjadi ``2**m`` tingkat keabuan.

    Dengan ``langkah = 256 / 2**m``, ``s = langkah * floor(r / langkah)`` (tiap piksel dibulatkan
    ke bawah ke tingkat terdekat). Bila ``rescale``, tingkat disebar rata dari 0 sampai 255
    sehingga putih tetap putih.
    """
    if not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ParameterError(f"Jumlah bit harus bilangan bulat 1..8, bukan {bits!r}")
    levels = 2**bits
    step = LEVELS // levels
    index = _R // step
    if rescale:
        return _lut(index * MAX_LEVEL / max(1, levels - 1))
    return _lut(index * step)


def slice_lut(low: float, high: float, level: float = 255, *, preserve: bool = False) -> Lut:
    """Gray-level slicing: sorot rentang intensitas ``[low, high]``.

    Piksel di dalam rentang diberi nilai ``level``. Piksel di luar rentang menjadi 0
    (``preserve=False``) atau dibiarkan seperti semula (``preserve=True``).
    """
    _check_level("Batas bawah", low)
    _check_level("Batas atas", high)
    _check_level("Level sorotan", level)
    if low > high:
        raise ParameterError(f"Batas bawah ({low}) tidak boleh melebihi batas atas ({high})")
    inside = (low <= _R) & (high >= _R)
    return _lut(np.where(inside, level, _R if preserve else 0))


def bitplane_lut(bit: int) -> Lut:
    """Bit-plane slicing: ``s = 255`` bila bit ke-``bit`` (0 = LSB ... 7 = MSB) bernilai 1."""
    if not isinstance(bit, int) or not 0 <= bit <= 7:
        raise ParameterError(f"Bit-plane harus bilangan bulat 0..7, bukan {bit!r}")
    return _lut(((_R.astype(np.uint8) >> bit) & 1) * MAX_LEVEL)
