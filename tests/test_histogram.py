from __future__ import annotations

import cv2
import numpy as np
import pytest

from imgenhance import (
    ParameterError,
    channel_histograms,
    equalization_map,
    equalize,
    equalize_lut,
    histogram,
)
from imgenhance.histogram import cumulative_distribution, probabilities

#: Contoh pada materi hlm. 17-21: citra 64 x 64, L = 8 tingkat keabuan.
SLIDE_COUNTS = np.array([790, 1023, 850, 656, 329, 245, 122, 81], dtype=np.int64)


def test_materi_peluang_dan_distribusi_kumulatif() -> None:
    assert SLIDE_COUNTS.sum() == 64 * 64
    np.testing.assert_allclose(
        probabilities(SLIDE_COUNTS), [0.19, 0.25, 0.21, 0.16, 0.08, 0.06, 0.03, 0.02], atol=0.005
    )
    np.testing.assert_allclose(
        cumulative_distribution(SLIDE_COUNTS),
        [0.19, 0.44, 0.65, 0.81, 0.89, 0.95, 0.98, 1.0],
        atol=0.005,
    )


def test_materi_peta_equalization() -> None:
    """Tabel r_k -> s_k pada hlm. 20: 0.14, 0.43, 0.71, 0.86, 0.86, 1.00, 1.00, 1.00."""
    mapping = equalization_map(SLIDE_COUNTS)
    assert mapping.tolist() == [1, 3, 5, 6, 6, 7, 7, 7]
    np.testing.assert_allclose(mapping / 7, [0.14, 0.43, 0.71, 0.86, 0.86, 1, 1, 1], atol=0.005)


def test_materi_jumlah_piksel_hasil_equalization() -> None:
    """Hlm. 21: hanya lima intensitas terpakai dengan 790, 1023, 850, 985, dan 448 piksel."""
    mapping = equalization_map(SLIDE_COUNTS)
    counts = {int(level): int(SLIDE_COUNTS[mapping == level].sum()) for level in np.unique(mapping)}
    assert counts == {1: 790, 3: 1023, 5: 850, 6: 985, 7: 448}


def test_histogram_counts_pixels(gray_image: np.ndarray) -> None:
    hist = histogram(gray_image)
    reference, _ = np.histogram(gray_image, bins=256, range=(0, 256))
    assert hist.shape == (256,) and hist.dtype == np.int64
    np.testing.assert_array_equal(hist, reference)
    assert hist.sum() == gray_image.size


def test_histogram_validation(color_image: np.ndarray, gray_image: np.ndarray) -> None:
    with pytest.raises(ParameterError):
        histogram(color_image)
    with pytest.raises(ParameterError):
        histogram(gray_image, levels=1)
    with pytest.raises(ParameterError):
        histogram(gray_image, levels=300)


def test_probabilities_reject_empty_histogram() -> None:
    with pytest.raises(ParameterError):
        probabilities(np.zeros(256, dtype=np.int64))


def test_channel_histograms_keys(color_image: np.ndarray, gray_image: np.ndarray) -> None:
    assert list(channel_histograms(gray_image)) == ["gray"]
    colored = channel_histograms(color_image)
    assert list(colored) == ["blue", "green", "red"]
    np.testing.assert_array_equal(colored["red"], histogram(color_image[:, :, 2].copy()))
    one_channel = channel_histograms(gray_image[:, :, None])
    assert list(one_channel) == ["gray"]


def test_equalize_lut_requires_256_levels() -> None:
    with pytest.raises(ParameterError):
        equalize_lut(SLIDE_COUNTS)


def test_equalization_flattens_the_cumulative_distribution() -> None:
    rng = np.random.default_rng(1)
    dark = np.clip(rng.normal(50, 20, size=(200, 200)), 0, 255).astype(np.uint8)
    out = equalize(dark)
    cdf = np.cumsum(histogram(out)) / out.size
    ideal = (np.arange(256) + 1) / 256
    assert np.abs(cdf - ideal).max() < 0.1  # hampir lurus: histogram merata
    assert out.mean() > dark.mean() + 40
    assert out.std() > dark.std()


def test_equalize_is_monotonic_mapping(gray_image: np.ndarray) -> None:
    lut = equalize_lut(histogram(gray_image))
    assert np.all(np.diff(lut.astype(int)) >= 0)


def test_equalize_constant_image_is_unchanged() -> None:
    flat = np.full((10, 12), 90, dtype=np.uint8)
    np.testing.assert_array_equal(equalize(flat), flat)


def test_equalize_color_default_keeps_hue(color_image: np.ndarray) -> None:
    pale = (128 + 0.3 * (color_image.astype(float) - 128)).astype(np.uint8)
    out = equalize(pale)
    before = cv2.cvtColor(pale, cv2.COLOR_BGR2YCrCb).astype(int)
    after = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb).astype(int)
    assert out.shape == pale.shape and out.dtype == np.uint8
    assert after[:, :, 0].std() > before[:, :, 0].std()


def test_equalize_per_channel_mode_equalizes_each_channel() -> None:
    rng = np.random.default_rng(3)
    skewed = np.dstack(
        [
            np.clip(rng.normal(40, 15, (80, 80)), 0, 255),
            np.clip(rng.normal(120, 15, (80, 80)), 0, 255),
            np.clip(rng.normal(210, 15, (80, 80)), 0, 255),
        ]
    ).astype(np.uint8)
    out = equalize(skewed, luminance_only=False)
    means = [out[:, :, i].mean() for i in range(3)]
    assert max(means) - min(means) < 30  # tiap kanal diratakan, jadi rata-ratanya mendekat
    assert not np.array_equal(out, equalize(skewed))


def test_equalize_gray_matches_opencv_closely(gray_image: np.ndarray) -> None:
    """Rumus materi (CDF langsung) hampir sama dengan cv2.equalizeHist (yang mengurangi cdf_min)."""
    ours = equalize(gray_image).astype(int)
    reference = cv2.equalizeHist(gray_image).astype(int)
    assert np.abs(ours - reference).max() <= 5
