"""Penggambaran hasil: gambar perbandingan (asli vs hasil + histogram + kurva), lembar kontak, dan
overlay webcam. Semuanya digambar dengan OpenCV/NumPy (tanpa matplotlib).

Teks memakai font Hershey bawaan OpenCV yang hanya mendukung ASCII; karakter lain diganti
otomatis oleh :func:`_safe`.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np

from ._types import Image, Lut
from .analysis import ImageStats
from .histogram import channel_histograms
from .point import LEVELS, MAX_LEVEL, as_bgr

Size = tuple[int, int]  #: (lebar, tinggi) dalam piksel
Color = tuple[int, int, int]  #: BGR

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAGE: Color = (243, 243, 243)
CARD: Color = (255, 255, 255)
BORDER: Color = (214, 214, 214)
INK: Color = (48, 48, 48)
MUTED: Color = (128, 128, 128)
GRID: Color = (232, 232, 232)
BEFORE_COLOR: Color = (150, 105, 55)  # biru tua (BGR)
AFTER_COLOR: Color = (35, 115, 235)  # oranye (BGR)
CHANNEL_COLORS: dict[str, Color] = {
    "gray": (110, 110, 110),
    "blue": (210, 110, 30),
    "green": (60, 160, 50),
    "red": (50, 50, 215),
}

DEFAULT_TILE: Size = (400, 300)


# --------------------------------------------------------------------------- #
# Primitif gambar
# --------------------------------------------------------------------------- #
def _safe(text: str) -> str:
    """Ubah ke ASCII agar tidak muncul '?' acak pada font Hershey."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return folded or "?"


def _text(
    canvas: Image,
    text: str,
    org: tuple[int, int],
    scale: float = 0.5,
    color: Color = INK,
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, _safe(text), org, FONT, scale, color, thickness, cv2.LINE_AA)


def _text_width(text: str, scale: float = 0.5, thickness: int = 1) -> int:
    (width, _), _ = cv2.getTextSize(_safe(text), FONT, scale, thickness)
    return int(width)


def _card(size: Size, color: Color = CARD) -> Image:
    """Kartu putih berbingkai tipis."""
    width, height = size
    tile = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), BORDER, 1)
    return cast(Image, tile)


def _caption(tile: Image, text: str) -> None:
    _text(tile, text, (12, 22), 0.5, INK, 1)


def _blend_fill(tile: Image, points: np.ndarray, color: Color, alpha: float) -> None:
    overlay = tile.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, alpha, tile, 1 - alpha, 0, dst=tile)


def fit_image(
    image: Image, size: Size, background: Color = CARD, max_upscale: float = 4.0
) -> Image:
    """Letakkan ``image`` (BGR/gray) di tengah kanvas ``size`` tanpa mengubah rasio aspek."""
    width, height = size
    bgr = as_bgr(image)
    src_h, src_w = bgr.shape[:2]
    scale = min(width / src_w, height / src_h, max_upscale)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST if scale >= 3 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=method)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return cast(Image, canvas)


# --------------------------------------------------------------------------- #
# Kartu (tile)
# --------------------------------------------------------------------------- #
def image_tile(image: Image, size: Size, caption: str) -> Image:
    """Kartu berisi gambar + keterangan di bagian atas."""
    tile = _card(size)
    top = 32
    inner = fit_image(image, (size[0] - 2, size[1] - top - 2))
    tile[top : top + inner.shape[0], 1 : 1 + inner.shape[1]] = inner
    _caption(tile, caption)
    return tile


def histogram_tile(image: Image, size: Size, caption: str, color: Color = BEFORE_COLOR) -> Image:
    """Kartu histogram: area terisi untuk grayscale, tiga garis B/G/R untuk gambar berwarna."""
    tile = _card(size)
    width, height = size
    x0, y0, x1, y1 = 16, 40, width - 16, height - 30
    hists = channel_histograms(image)
    all_bins = np.concatenate(list(hists.values()))
    peak = int(all_bins.max()) or 1
    # Puncak tunggal yang ekstrem (mis. 40% piksel hitam) tidak boleh meratakan seluruh grafik,
    # jadi tinggi grafik dibatasi dan puncaknya dipotong; angka puncak sebenarnya tetap dicantumkan.
    cap = min(peak, max(float(np.percentile(all_bins, 99)) * 1.25, peak * 0.1, 1.0))

    for tick in (0, 64, 128, 192, 255):
        x = x0 + round(tick * (x1 - x0) / MAX_LEVEL)
        cv2.line(tile, (x, y0), (x, y1), GRID, 1)
        label = str(tick)
        _text(tile, label, (x - _text_width(label, 0.38) // 2, y1 + 18), 0.38, MUTED)

    xs = np.linspace(x0, x1, LEVELS)
    for name, hist in hists.items():
        ys = y1 - np.minimum(hist, cap) / cap * (y1 - y0)
        points = np.stack([xs, ys], axis=1).round().astype(np.int32)
        line_color = color if name == "gray" else CHANNEL_COLORS[name]
        if name == "gray":
            area = np.vstack([[x0, y1], points, [x1, y1]]).astype(np.int32)
            _blend_fill(tile, area, line_color, 0.35)
        cv2.polylines(tile, [points.reshape(-1, 1, 2)], False, line_color, 1, cv2.LINE_AA)

    cv2.line(tile, (x0, y1), (x1, y1), MUTED, 1)
    _caption(tile, caption)
    label = f"puncak {peak:,} px".replace(",", ".") + (" (dipotong)" if cap < peak else "")
    _text(tile, label, (width - 12 - _text_width(label, 0.4), 22), 0.4, MUTED)
    return tile


def curve_tile(lut: Lut | None, size: Size, caption: str = "Kurva transformasi s = T(r)") -> Image:
    """Kartu kurva transformasi ``s = T(r)``; garis putus-putus = identitas (tanpa perubahan)."""
    tile = _card(size)
    width, height = size
    _caption(tile, caption)
    if lut is None:
        message = "Tidak ada kurva T(r):"
        _text(tile, message, (12, height // 2 - 8), 0.5, MUTED)
        _text(tile, "operasi aras lokal atau per kanal warna", (12, height // 2 + 14), 0.45, MUTED)
        return tile

    side = min(width - 80, height - 74)
    x0, y1 = (width - side) // 2 + 8, height - 32
    x1, y0 = x0 + side, y1 - side

    def px(r: float) -> int:
        return round(x0 + r * side / MAX_LEVEL)

    def py(s: float) -> int:
        return round(y1 - s * side / MAX_LEVEL)

    for tick in (0, 64, 128, 192, 255):
        cv2.line(tile, (px(tick), y0), (px(tick), y1), GRID, 1)
        cv2.line(tile, (x0, py(tick)), (x1, py(tick)), GRID, 1)
        _text(tile, str(tick), (px(tick) - 8, y1 + 16), 0.36, MUTED)
        _text(tile, str(tick), (x0 - 30, py(tick) + 4), 0.36, MUTED)
    for k in range(0, LEVELS - 1, 12):  # diagonal identitas putus-putus
        cv2.line(
            tile, (px(k), py(k)), (px(min(k + 6, MAX_LEVEL)), py(min(k + 6, MAX_LEVEL))), MUTED, 1
        )
    cv2.rectangle(tile, (x0, y0), (x1, y1), MUTED, 1)

    points = np.array([[px(r), py(int(lut[r]))] for r in range(LEVELS)], dtype=np.int32)
    cv2.polylines(tile, [points.reshape(-1, 1, 2)], False, AFTER_COLOR, 2, cv2.LINE_AA)
    _text(tile, "r (masukan)", (x1 - 78, y1 + 28), 0.38, MUTED)
    _text(tile, "s (keluaran)", (x0 - 38, y0 - 8), 0.38, MUTED)
    return tile


def stats_tile(before: ImageStats, after: ImageStats, size: Size, note: str = "") -> Image:
    """Kartu tabel statistik histogram sebelum / sesudah."""
    tile = _card(size)
    width, _ = size
    _caption(tile, "Statistik (luminans)")
    col_before, col_after = width - 190, width - 90
    _text(tile, "Asli", (col_before, 50), 0.5, BEFORE_COLOR, 2)
    _text(tile, "Hasil", (col_after, 50), 0.5, AFTER_COLOR, 2)
    rows = (
        ("Rata-rata", f"{before.mean:.1f}", f"{after.mean:.1f}"),
        ("Simpangan baku", f"{before.std:.1f}", f"{after.std:.1f}"),
        ("Min - Maks", f"{before.minimum}-{before.maximum}", f"{after.minimum}-{after.maximum}"),
        ("Sebaran p2-p98", str(before.spread), str(after.spread)),
        ("Entropi (bit)", f"{before.entropy:.2f}", f"{after.entropy:.2f}"),
    )
    for i, (label, a, b) in enumerate(rows):
        y = 84 + i * 30
        cv2.line(tile, (12, y - 18), (width - 12, y - 18), GRID, 1)
        _text(tile, label, (14, y), 0.5, INK)
        _text(tile, a, (col_before, y), 0.5, INK)
        _text(tile, b, (col_after, y), 0.5, INK)
    if note:
        _text(tile, note, (14, 84 + len(rows) * 30 + 12), 0.45, MUTED)
    return tile


# --------------------------------------------------------------------------- #
# Komposisi
# --------------------------------------------------------------------------- #
def _header(width: int, title: str, subtitle: str = "") -> Image:
    bar = np.full((52, width, 3), PAGE, dtype=np.uint8)
    _text(bar, title, (16, 26), 0.75, INK, 2)
    if subtitle:
        _text(bar, subtitle, (16, 45), 0.45, MUTED)
    return cast(Image, bar)


def comparison(
    before: Image,
    after: Image,
    title: str,
    *,
    lut: Lut | None = None,
    subtitle: str = "",
    note: str = "",
    tile: Size = DEFAULT_TILE,
) -> Image:
    """Gambar perbandingan 3 x 2: asli | hasil | statistik, lalu histogram asli | histogram hasil
    | kurva transformasi."""
    pad, gap = 16, 12
    tw, th = tile
    width = 2 * pad + 3 * tw + 2 * gap
    top = [
        image_tile(before, tile, "Citra asli"),
        image_tile(after, tile, "Hasil"),
        stats_tile(ImageStats.from_image(before), ImageStats.from_image(after), tile, note),
    ]
    bottom = [
        histogram_tile(before, tile, "Histogram asli", BEFORE_COLOR),
        histogram_tile(after, tile, "Histogram hasil", AFTER_COLOR),
        curve_tile(lut, tile),
    ]
    page = np.full((52 + 2 * th + gap + pad, width, 3), PAGE, dtype=np.uint8)
    page[:52] = _header(width, title, subtitle)
    for row, tiles in enumerate((top, bottom)):
        y = 52 + row * (th + gap)
        for col, t in enumerate(tiles):
            x = pad + col * (tw + gap)
            page[y : y + th, x : x + tw] = t
    return cast(Image, page)


def contact_sheet(
    items: Sequence[tuple[str, Image]],
    columns: int = 4,
    tile: Size = (300, 240),
    title: str = "",
) -> Image:
    """Lembar kontak: kisi gambar berlabel (mis. hasil semua operasi pada satu foto)."""
    if not items:
        raise ValueError("contact_sheet butuh minimal satu gambar")
    columns = max(1, min(columns, len(items)))
    rows = -(-len(items) // columns)
    pad, gap = 16, 10
    tw, th = tile
    head = 52 if title else 0
    page = np.full(
        (
            head + 2 * pad + rows * th + (rows - 1) * gap,
            2 * pad + columns * tw + (columns - 1) * gap,
            3,
        ),
        PAGE,
        dtype=np.uint8,
    )
    if title:
        page[:head] = _header(page.shape[1], title)
    for index, (label, image) in enumerate(items):
        row, col = divmod(index, columns)
        x, y = pad + col * (tw + gap), head + pad + row * (th + gap)
        page[y : y + th, x : x + tw] = image_tile(image, tile, label)
    return cast(Image, page)


# --------------------------------------------------------------------------- #
# Webcam
# --------------------------------------------------------------------------- #
def side_by_side(left: Image, right: Image) -> Image:
    """Susun dua gambar horizontal dengan tinggi sama (gambar grayscale dijadikan BGR)."""
    a, b = as_bgr(left), as_bgr(right)
    if a.shape[0] != b.shape[0]:
        scale = a.shape[0] / b.shape[0]
        size = (max(1, round(b.shape[1] * scale)), a.shape[0])
        b = cast(Image, cv2.resize(b, size, interpolation=cv2.INTER_AREA))
    return cast(Image, np.hstack([a, b]))


def draw_live_overlay(
    display: Image,
    title: str,
    detail: str,
    histogram_source: Image | None = None,
    hint: str = "",
) -> Image:
    """Tambahkan pita judul, petunjuk tombol, dan (opsional) histogram mini pada frame webcam."""
    out = as_bgr(display).copy()
    height, width = out.shape[:2]
    band = out[:52].copy()
    cv2.rectangle(out, (0, 0), (width, 52), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(out[:52], 0.7, band, 0.3, 0, dst=out[:52])
    _text(out, title, (12, 22), 0.65, (255, 255, 255), 2)
    _text(out, detail, (12, 42), 0.45, (210, 210, 210))

    if hint:
        _text(out, hint, (12, height - 12), 0.42, (0, 0, 0), 3)  # tepi hitam agar terbaca
        _text(out, hint, (12, height - 12), 0.42, (255, 255, 255), 1)

    if histogram_source is not None and height > 130 + 60 and width > 260 + 20:
        mini = histogram_tile(histogram_source, (260, 130), "Histogram hasil", AFTER_COLOR)
        y, x = height - 130 - 28, width - 260 - 10
        region = out[y : y + 130, x : x + 260]
        cv2.addWeighted(mini, 0.85, region, 0.15, 0, dst=region)
    return out
