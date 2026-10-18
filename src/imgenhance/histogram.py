"""Histogram grayscale dan *histogram equalization* (materi P03 hlm. 14-21).

Histogram menyatakan distribusi derajat keabuan: ``n_k`` adalah jumlah piksel dengan intensitas
``r_k``. Dalam bentuk peluang ``P(r_k) = n_k / n``. *Histogram equalization* meratakan distribusi
itu lewat distribusi kumulatif::

    s_k = T(r_k) = sum_{j=0..k} P(r_j)

lalu ``s_k`` (0..1) dibulatkan ke tingkat keabuan terdekat: ``round(s_k * (L - 1))``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from ._types import Gray, Histogram, Image, Lut
from .errors import ParameterError
from .point import LEVELS, apply_lut, identity_lut, luminance


def histogram(gray: Gray, levels: int = LEVELS) -> Histogram:
    """Hitung histogram: ``hist[k]`` = jumlah piksel berintensitas ``k`` (panjang ``levels``)."""
    if gray.ndim != 2:
        raise ParameterError("Histogram dihitung dari gambar grayscale (2 dimensi)")
    if not 2 <= levels <= LEVELS:
        raise ParameterError(f"Jumlah level harus 2..{LEVELS}, bukan {levels}")
    return cast(Histogram, np.bincount(gray.ravel(), minlength=levels)[:levels].astype(np.int64))


def channel_histograms(image: Image) -> dict[str, Histogram]:
    """Histogram per kanal: ``{"gray": ...}`` atau ``{"blue": ..., "green": ..., "red": ...}``."""
    if image.ndim == 2:
        return {"gray": histogram(image)}
    if image.shape[2] == 1:
        return {"gray": histogram(image[:, :, 0])}
    names = ("blue", "green", "red")  # urutan kanal OpenCV: B, G, R
    return {name: histogram(np.ascontiguousarray(image[:, :, i])) for i, name in enumerate(names)}


def probabilities(hist: Histogram) -> npt.NDArray[np.float64]:
    """Peluang tiap level: ``P(r_k) = n_k / n``."""
    total = int(hist.sum())
    if total <= 0:
        raise ParameterError("Histogram kosong (tidak ada piksel)")
    return hist.astype(np.float64) / total


def cumulative_distribution(hist: Histogram) -> npt.NDArray[np.float64]:
    """Distribusi kumulatif ``s_k = sum_{j<=k} P(r_j)``; elemen terakhir selalu 1.0."""
    return cast(npt.NDArray[np.float64], np.cumsum(probabilities(hist)))


def equalization_map(hist: Histogram) -> npt.NDArray[np.int64]:
    """Peta equalization: indeks level baru untuk tiap level lama, ``round(s_k * (L - 1))``.

    ``L`` adalah panjang histogram, jadi fungsi ini juga cocok untuk contoh kecil di materi
    (``L = 8``).
    """
    levels = len(hist)
    cdf = cumulative_distribution(hist)
    return cast(npt.NDArray[np.int64], np.rint(cdf * (levels - 1)).astype(np.int64))


def equalize_lut(hist: Histogram) -> Lut:
    """LUT 256 elemen untuk histogram equalization dari ``hist`` (harus 256 level).

    Gambar yang hanya berisi satu intensitas tidak punya distribusi untuk diratakan; rumus
    kumulatif akan memutihkannya seluruhnya, jadi untuk kasus itu dikembalikan LUT identitas.
    """
    if len(hist) != LEVELS:
        raise ParameterError(f"LUT equalization butuh histogram {LEVELS} level, bukan {len(hist)}")
    if np.count_nonzero(hist) <= 1:
        return identity_lut()
    return equalization_map(hist).astype(np.uint8)


def equalize(image: Image, *, luminance_only: bool = True) -> Image:
    """Histogram equalization untuk gambar grayscale atau berwarna.

    Gambar berwarna diratakan pada luminansnya (``luminance_only=True``, hue terjaga). Dengan
    ``luminance_only=False`` tiap kanal B, G, R diratakan sendiri-sendiri; kontras naik tetapi
    warna bisa bergeser.
    """
    if image.ndim == 2 or image.shape[2] == 1 or luminance_only:
        lut = equalize_lut(histogram(luminance(image)))
        return apply_lut(image, lut, luminance_only=True)
    channels = [
        apply_lut(
            np.ascontiguousarray(image[:, :, i]),
            equalize_lut(histogram(np.ascontiguousarray(image[:, :, i]))),
        )
        for i in range(image.shape[2])
    ]
    return cast(Image, np.dstack(channels))
