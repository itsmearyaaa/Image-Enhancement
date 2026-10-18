from __future__ import annotations

import cv2
import numpy as np
import pytest

from imgenhance import (
    ParameterError,
    apply_lut,
    bitplane_lut,
    brightness_lut,
    contrast_lut,
    gamma_lut,
    log_lut,
    negative_lut,
    quantize_lut,
    slice_lut,
    stretch_lut,
    threshold_lut,
    to_grayscale,
)
from imgenhance.point import as_bgr, identity_lut, luminance

R = np.arange(256)


def _is_lut(lut: np.ndarray) -> bool:
    return lut.shape == (256,) and lut.dtype == np.uint8


# --------------------------------------------------------------------------- #
# Contoh dan soal pada materi P03
# --------------------------------------------------------------------------- #
def test_materi_thresholding_soal() -> None:
    """Materi hlm. 29: nilai < 128 -> 0, nilai >= 128 -> 255."""
    image = np.array(
        [
            [40, 160, 69, 170, 123],
            [20, 250, 140, 80, 90],
            [70, 30, 128, 115, 85],
            [40, 234, 70, 211, 125],
            [20, 34, 80, 221, 30],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [0, 255, 0, 255, 0],
            [0, 255, 255, 0, 0],
            [0, 0, 255, 0, 0],
            [0, 255, 0, 255, 0],
            [0, 0, 0, 255, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(apply_lut(image, threshold_lut(128)), expected)


def test_materi_kontras_gain_2_pusat_10() -> None:
    """Materi hlm. 26: matriks 5x5 dengan G = 2 dan P = 10."""
    original = np.array(
        [
            [15, 30, 40, 50, 45],
            [25, 25, 25, 40, 30],
            [40, 50, 35, 45, 35],
            [50, 40, 15, 25, 40],
            [30, 40, 20, 25, 20],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [20, 50, 70, 90, 80],
            [40, 40, 40, 70, 50],
            [70, 90, 60, 80, 60],
            [90, 70, 20, 40, 70],
            [50, 70, 30, 40, 30],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(apply_lut(original, contrast_lut(2, 10)), expected)


def test_materi_mbit_contoh_x_100() -> None:
    """Materi hlm. 12: X = 100 dengan langkah 64 menjadi 64 (langkah 64 = 4 tingkat = 2 bit)."""
    assert quantize_lut(2)[100] == 64


def test_materi_negasi() -> None:
    """Materi hlm. 22: G_baru = 255 - G_lama."""
    np.testing.assert_array_equal(negative_lut(), 255 - R)


# --------------------------------------------------------------------------- #
# Bentuk dan sifat LUT
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lut",
    [
        identity_lut(),
        negative_lut(),
        brightness_lut(40),
        contrast_lut(1.7, 100),
        gamma_lut(0.45),
        log_lut(),
        stretch_lut(50, 10, 200, 240),
        threshold_lut(100),
        quantize_lut(3),
        quantize_lut(3, rescale=True),
        slice_lut(50, 120),
        bitplane_lut(5),
    ],
)
def test_every_lut_is_256_uint8(lut: np.ndarray) -> None:
    assert _is_lut(lut)


@pytest.mark.parametrize(
    "lut",
    [
        brightness_lut(-60),
        brightness_lut(60),
        contrast_lut(0.6),
        contrast_lut(2.5),
        gamma_lut(0.3),
        gamma_lut(3.0),
        log_lut(),
        stretch_lut(40, 10, 210, 245),
        threshold_lut(77),
        quantize_lut(4),
    ],
)
def test_monotonic_operations_never_decrease(lut: np.ndarray) -> None:
    assert np.all(np.diff(lut.astype(int)) >= 0)


def test_identity_leaves_image_unchanged(ramp: np.ndarray) -> None:
    np.testing.assert_array_equal(apply_lut(ramp, identity_lut()), ramp)


def test_negative_twice_is_identity(color_image: np.ndarray) -> None:
    once = apply_lut(color_image, negative_lut())
    np.testing.assert_array_equal(apply_lut(once, negative_lut()), color_image)


# --------------------------------------------------------------------------- #
# Brightness, kontras, gamma, log
# --------------------------------------------------------------------------- #
def test_brightness_adds_offset_and_clips() -> None:
    lut = brightness_lut(75)
    assert lut[0] == 75 and lut[100] == 175 and lut[200] == 255 and lut[255] == 255
    lut = brightness_lut(-75)
    assert lut[0] == 0 and lut[100] == 25 and lut[255] == 180


@pytest.mark.parametrize("offset", [-256, 256, 1000])
def test_brightness_rejects_out_of_range(offset: int) -> None:
    with pytest.raises(ParameterError):
        brightness_lut(offset)


def test_contrast_fixed_point_and_direction() -> None:
    higher, lower = contrast_lut(2.0, 128), contrast_lut(0.5, 128)
    assert higher[128] == 128 == lower[128]
    assert higher[100] < 100 < higher[200] + 0  # gelap makin gelap
    assert lower[100] > 100 and lower[200] < 200  # menyempit ke tengah


def test_contrast_validation() -> None:
    with pytest.raises(ParameterError):
        contrast_lut(-1)
    with pytest.raises(ParameterError):
        contrast_lut(1, 300)


def test_gamma_endpoints_and_direction() -> None:
    for gamma in (0.2, 0.5, 1.0, 2.2):
        lut = gamma_lut(gamma)
        assert lut[0] == 0 and lut[255] == 255
    assert gamma_lut(1.0).tolist() == R.tolist()
    assert gamma_lut(0.5)[64] > 64  # < 1 mencerahkan
    assert gamma_lut(2.0)[64] < 64  # > 1 menggelapkan


@pytest.mark.parametrize("gamma", [0, -1])
def test_gamma_rejects_non_positive(gamma: float) -> None:
    with pytest.raises(ParameterError):
        gamma_lut(gamma)


def test_log_endpoints_and_brightens_shadows() -> None:
    lut = log_lut()
    assert lut[0] == 0 and lut[255] == 255
    assert lut[10] > 10 * 4  # area gelap dipekarkan kuat


# --------------------------------------------------------------------------- #
# Contrast stretching
# --------------------------------------------------------------------------- #
def test_stretch_linear_special_case_is_identity() -> None:
    np.testing.assert_array_equal(stretch_lut(80, 80, 170, 170), R)


def test_stretch_min_max() -> None:
    lut = stretch_lut(60, 0, 180, 255)
    assert lut[60] == 0 and lut[180] == 255 and lut[0] == 0 and lut[255] == 255
    assert lut[120] == pytest.approx(127.5, abs=1)


def test_stretch_with_r1_equal_r2_is_threshold() -> None:
    np.testing.assert_array_equal(stretch_lut(100, 0, 100, 255), threshold_lut(100))


def test_stretch_edges_of_range() -> None:
    assert _is_lut(stretch_lut(0, 0, 255, 255))
    assert _is_lut(stretch_lut(0, 20, 0, 20))


@pytest.mark.parametrize(
    "points",
    [(100, 0, 50, 255), (10, 200, 100, 100), (-1, 0, 5, 5), (0, 0, 256, 255)],
)
def test_stretch_rejects_invalid_points(points: tuple[int, int, int, int]) -> None:
    with pytest.raises(ParameterError):
        stretch_lut(*points)


# --------------------------------------------------------------------------- #
# Threshold, m-bit, slicing, bit-plane
# --------------------------------------------------------------------------- #
def test_threshold_is_inclusive_at_t() -> None:
    lut = threshold_lut(128)
    assert lut[127] == 0 and lut[128] == 255
    assert set(np.unique(lut)) == {0, 255}


def test_threshold_rejects_out_of_range() -> None:
    with pytest.raises(ParameterError):
        threshold_lut(256)


@pytest.mark.parametrize("bits", range(1, 9))
def test_quantize_level_count(bits: int) -> None:
    assert len(np.unique(quantize_lut(bits))) == 2**bits


def test_quantize_8_bits_is_identity_and_rescale_reaches_white() -> None:
    np.testing.assert_array_equal(quantize_lut(8), R)
    rescaled = quantize_lut(2, rescale=True)
    assert rescaled.max() == 255 and sorted(np.unique(rescaled)) == [0, 85, 170, 255]
    assert quantize_lut(2).max() == 192  # tanpa rescale: putih jadi 192


@pytest.mark.parametrize("bits", [0, 9, -1, 2.5])
def test_quantize_rejects_bad_bits(bits: object) -> None:
    with pytest.raises(ParameterError):
        quantize_lut(bits)  # type: ignore[arg-type]


def test_slice_highlights_range_and_optionally_keeps_background() -> None:
    plain = slice_lut(100, 150)
    assert plain[99] == 0 and plain[100] == 255 and plain[150] == 255 and plain[151] == 0
    kept = slice_lut(100, 150, 200, preserve=True)
    assert kept[50] == 50 and kept[120] == 200 and kept[250] == 250


def test_slice_rejects_inverted_range() -> None:
    with pytest.raises(ParameterError):
        slice_lut(150, 100)


def test_bitplane_matches_binary_digits() -> None:
    assert np.array_equal(bitplane_lut(7) == 255, R >= 128)
    assert np.array_equal(bitplane_lut(0) == 255, R % 2 == 1)
    assert np.array_equal(bitplane_lut(3) == 255, (R >> 3) & 1 == 1)


@pytest.mark.parametrize("bit", [-1, 8, 1.5])
def test_bitplane_rejects_bad_bit(bit: object) -> None:
    with pytest.raises(ParameterError):
        bitplane_lut(bit)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Konversi grayscale dan penerapan LUT
# --------------------------------------------------------------------------- #
def test_grayscale_average_follows_slide_formula() -> None:
    pixel = np.array([[[30, 60, 90]]], dtype=np.uint8)  # B, G, R
    assert to_grayscale(pixel, "average")[0, 0] == round((90 + 60 + 30) / 3)


def test_grayscale_luminance_matches_opencv(color_image: np.ndarray) -> None:
    ours = to_grayscale(color_image, "luminance").astype(int)
    reference = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY).astype(int)
    assert np.abs(ours - reference).max() <= 1


def test_grayscale_custom_weights_and_validation() -> None:
    red_only = np.array([[[10, 20, 200]]], dtype=np.uint8)
    assert to_grayscale(red_only, weights=(1.0, 0.0, 0.0))[0, 0] == 200
    with pytest.raises(ParameterError):
        to_grayscale(red_only, weights=(0.5, 0.5, 0.5))
    with pytest.raises(ParameterError):
        to_grayscale(red_only, weights=(1.5, -0.5, 0.0))
    with pytest.raises(ParameterError):
        to_grayscale(red_only, "tidak-ada")


def test_grayscale_of_gray_is_a_copy(gray_image: np.ndarray) -> None:
    out = to_grayscale(gray_image)
    np.testing.assert_array_equal(out, gray_image)
    out[0, 0] ^= 0xFF
    assert out[0, 0] != gray_image[0, 0]


def test_apply_lut_color_uses_same_table_for_each_channel(color_image: np.ndarray) -> None:
    out = apply_lut(color_image, negative_lut())
    np.testing.assert_array_equal(out, 255 - color_image)


def test_apply_lut_luminance_only_keeps_chroma(color_image: np.ndarray) -> None:
    # warna sedang (tidak jenuh ekstrem) agar hasil tidak terpotong di batas gamut RGB
    muted = (60 + 0.4 * color_image.astype(float)).astype(np.uint8)
    lut = gamma_lut(0.6)
    on_luma = apply_lut(muted, lut, luminance_only=True)
    on_channels = apply_lut(muted, lut)

    def chroma_shift(out: np.ndarray) -> float:
        before = cv2.cvtColor(muted, cv2.COLOR_BGR2YCrCb).astype(int)
        after = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb).astype(int)
        return float(np.abs(after[:, :, 1:] - before[:, :, 1:]).mean())

    assert chroma_shift(on_luma) < 3  # Cr, Cb hampir sama
    assert chroma_shift(on_luma) < chroma_shift(on_channels)  # lebih setia daripada per kanal
    assert luminance(on_luma).mean() > luminance(muted).mean()  # tetapi lebih terang


def test_apply_lut_validates_inputs(gray_image: np.ndarray) -> None:
    with pytest.raises(ParameterError):
        apply_lut(gray_image, np.zeros(10, dtype=np.uint8))
    with pytest.raises(ParameterError):
        apply_lut(gray_image.astype(np.float32), identity_lut())
    with pytest.raises(ParameterError):
        apply_lut(np.zeros((4, 4, 5), dtype=np.uint8), identity_lut())


def test_luminance_and_as_bgr_helpers(color_image: np.ndarray, gray_image: np.ndarray) -> None:
    assert luminance(color_image).shape == color_image.shape[:2]
    assert luminance(gray_image) is gray_image
    bgr = as_bgr(gray_image)
    assert bgr.shape == (*gray_image.shape, 3)
    assert np.array_equal(bgr[:, :, 0], gray_image)
    assert as_bgr(color_image) is color_image
