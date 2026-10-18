from __future__ import annotations

import numpy as np
import pytest

from imgenhance import ImageStats, negative_lut
from imgenhance.render import (
    _safe,
    comparison,
    contact_sheet,
    curve_tile,
    draw_live_overlay,
    fit_image,
    histogram_tile,
    image_tile,
    side_by_side,
    stats_tile,
)


def test_comparison_layout_and_dtype(color_image: np.ndarray) -> None:
    after = 255 - color_image
    panel = comparison(
        color_image, after, "Negasi", lut=negative_lut(), subtitle="x.png | negative"
    )
    assert panel.dtype == np.uint8 and panel.ndim == 3 and panel.shape[2] == 3
    assert panel.shape[1] == 2 * 16 + 3 * 400 + 2 * 12
    assert panel.shape[0] == 52 + 2 * 300 + 12 + 16


def test_comparison_accepts_gray_and_missing_curve(gray_image: np.ndarray) -> None:
    panel = comparison(gray_image, gray_image, "Tanpa kurva", lut=None, note="catatan")
    assert panel.shape[2] == 3


def test_comparison_is_not_blank_and_draws_something_per_cell(color_image: np.ndarray) -> None:
    panel = comparison(color_image, 255 - color_image, "t", lut=negative_lut())
    assert panel.std() > 10
    assert len(np.unique(panel.reshape(-1, 3), axis=0)) > 50


def test_histogram_tile_for_gray_and_color(gray_image: np.ndarray, color_image: np.ndarray) -> None:
    for image in (gray_image, color_image):
        tile = histogram_tile(image, (300, 200), "Histogram")
        assert tile.shape == (200, 300, 3)
        assert tile.std() > 0


def test_histogram_tile_with_extreme_spike_does_not_crash() -> None:
    spike = np.zeros((100, 100), dtype=np.uint8)
    spike[:5, :5] = 255
    assert histogram_tile(spike, (300, 200), "spike").shape == (200, 300, 3)


def test_curve_tile_variants() -> None:
    assert curve_tile(negative_lut(), (400, 300)).shape == (300, 400, 3)
    assert curve_tile(None, (400, 300)).shape == (300, 400, 3)


def test_stats_tile_shape(gray_image: np.ndarray) -> None:
    stats = ImageStats.from_image(gray_image)
    assert stats_tile(stats, stats, (400, 300), "catatan").shape == (300, 400, 3)


def test_fit_image_keeps_aspect_and_size() -> None:
    wide = np.full((50, 200, 3), 10, dtype=np.uint8)
    fitted = fit_image(wide, (100, 100))
    assert fitted.shape == (100, 100, 3)
    rows = np.where((fitted != 255).any(axis=(1, 2)))[0]
    assert rows.max() - rows.min() + 1 == 25  # 200x50 -> 100x25
    assert fit_image(np.zeros((2, 2), dtype=np.uint8), (100, 100)).shape == (100, 100, 3)


def test_image_tile_shape(color_image: np.ndarray) -> None:
    assert image_tile(color_image, (320, 240), "Asli").shape == (240, 320, 3)


def test_contact_sheet_grid(color_image: np.ndarray, gray_image: np.ndarray) -> None:
    items = [
        ("a", color_image),
        ("b", gray_image),
        ("c", color_image),
        ("d", gray_image),
        ("e", color_image),
    ]
    sheet = contact_sheet(items, columns=3, tile=(200, 150), title="Judul")
    assert sheet.shape == (52 + 2 * 16 + 2 * 150 + 10, 2 * 16 + 3 * 200 + 2 * 10, 3)
    one = contact_sheet(items[:1], columns=4, tile=(200, 150))
    assert one.shape[1] == 2 * 16 + 200  # kolom dipangkas sesuai jumlah gambar


def test_contact_sheet_rejects_empty() -> None:
    with pytest.raises(ValueError, match="minimal satu"):
        contact_sheet([])


def test_side_by_side_equalizes_heights(color_image: np.ndarray, gray_image: np.ndarray) -> None:
    combined = side_by_side(color_image, gray_image)
    assert combined.shape[0] == color_image.shape[0] and combined.shape[2] == 3
    assert combined.shape[1] > color_image.shape[1]


def test_live_overlay_with_and_without_histogram(color_image: np.ndarray) -> None:
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    with_hist = draw_live_overlay(frame, "Judul", "detail", frame, "petunjuk")
    plain = draw_live_overlay(frame, "Judul", "detail")
    assert with_hist.shape == plain.shape == frame.shape
    assert with_hist.sum() > plain.sum() > 0
    tiny = draw_live_overlay(
        color_image[:20, :20], "J", "d", color_image
    )  # terlalu kecil untuk histogram
    assert tiny.shape == (20, 20, 3)


def test_safe_strips_non_ascii() -> None:
    assert _safe("Citra – asli ≥ 128") == "Citra  asli  128"
    assert _safe("≥") == "?"
    assert _safe("biasa") == "biasa"
