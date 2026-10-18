from __future__ import annotations

import numpy as np
import pytest

from imgenhance import ParameterError, median_filter, sharpen, smooth


def _noisy(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(128, 40, (60, 60)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("method", ["mean", "gaussian"])
def test_smooth_reduces_noise(method: str) -> None:
    noisy = _noisy()
    out = smooth(noisy, 5, method)
    assert out.shape == noisy.shape and out.dtype == np.uint8
    assert out.std() < noisy.std() / 2


def test_smooth_handles_color(color_image: np.ndarray) -> None:
    assert smooth(color_image, 3).shape == color_image.shape


def test_median_removes_salt_and_pepper_completely() -> None:
    flat = np.full((30, 30), 100, dtype=np.uint8)
    corrupted = flat.copy()
    corrupted[5, 5], corrupted[10, 20], corrupted[22, 8] = 255, 0, 255
    np.testing.assert_array_equal(median_filter(corrupted, 3), flat)


def test_sharpen_increases_edge_contrast() -> None:
    edge = np.full((20, 20), 80, dtype=np.uint8)
    edge[:, 10:] = 170
    soft = smooth(edge, 5)
    sharp = sharpen(soft, 1.5, 5)
    assert (
        np.abs(np.diff(sharp.astype(int), axis=1)).max()
        > np.abs(np.diff(soft.astype(int), axis=1)).max()
    )


def test_sharpen_amount_zero_is_identity(color_image: np.ndarray) -> None:
    np.testing.assert_array_equal(sharpen(color_image, 0.0), color_image)


@pytest.mark.parametrize("ksize", [1, 2, 4, 0, -3])
def test_kernel_must_be_odd_and_at_least_three(ksize: int) -> None:
    image = _noisy()
    for func in (smooth, median_filter, sharpen):
        with pytest.raises(ParameterError):
            func(image, ksize=ksize)  # type: ignore[operator]


def test_unknown_smooth_method_and_negative_amount() -> None:
    with pytest.raises(ParameterError):
        smooth(_noisy(), 3, "median")
    with pytest.raises(ParameterError):
        sharpen(_noisy(), -1)
