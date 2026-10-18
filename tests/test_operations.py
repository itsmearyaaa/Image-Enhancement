from __future__ import annotations

import json

import numpy as np
import pytest

from imgenhance import (
    OPERATIONS,
    OperationError,
    ParameterError,
    apply_operation,
    get_operation,
    list_operations,
)
from imgenhance.operations import Param

GRAY_ONLY = {"grayscale", "threshold", "mbit", "slice", "bitplane"}


def test_registry_is_consistent() -> None:
    operations = list_operations()
    assert len(operations) == len(OPERATIONS) == 15
    assert len({op.name for op in operations}) == len(operations)
    for op in operations:
        assert op.title and op.summary and op.formula and op.ref and op.group
        assert len({p.name for p in op.params}) == len(op.params)
        for param in op.params:  # nilai bawaan harus lolos validasinya sendiri
            assert param.coerce(param.default) == param.default


def test_every_operation_serializes_to_json() -> None:
    payload = json.dumps([op.to_dict() for op in list_operations()])
    assert "equalize" in payload


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_every_operation_runs_on_color_and_gray(
    name: str, color_image: np.ndarray, gray_image: np.ndarray
) -> None:
    for image in (color_image, gray_image):
        result = apply_operation(image, name)
        assert result.image.dtype == np.uint8
        assert result.image.shape[:2] == image.shape[:2]
        if result.lut is not None:
            assert result.lut.shape == (256,) and result.lut.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(GRAY_ONLY))
def test_gray_only_operations_return_2d(name: str, color_image: np.ndarray) -> None:
    assert apply_operation(color_image, name).image.ndim == 2


@pytest.mark.parametrize("name", ["negative", "brightness", "contrast", "gamma", "log", "equalize"])
def test_tone_operations_keep_color(name: str, color_image: np.ndarray) -> None:
    assert apply_operation(color_image, name).image.shape == color_image.shape


@pytest.mark.parametrize("name", ["smooth", "median", "sharpen"])
def test_local_operations_have_no_curve(name: str, color_image: np.ndarray) -> None:
    assert apply_operation(color_image, name).lut is None


def test_operations_do_not_modify_their_input(color_image: np.ndarray) -> None:
    snapshot = color_image.copy()
    for name in OPERATIONS:
        apply_operation(color_image, name)
    np.testing.assert_array_equal(color_image, snapshot)


def test_operations_work_on_tiny_images() -> None:
    tiny = np.arange(9, dtype=np.uint8).reshape(3, 3)
    for name in OPERATIONS:
        assert apply_operation(
            tiny, name, {"ksize": 3} if "ksize" in _names(name) else {}
        ).image.shape[:2] == (3, 3)


def _names(operation: str) -> set[str]:
    return {p.name for p in get_operation(operation).params}


# --------------------------------------------------------------------------- #
# Parameter: nilai bawaan, konversi tipe, dan validasi
# --------------------------------------------------------------------------- #
def test_resolve_fills_defaults_and_coerces_strings() -> None:
    op = get_operation("brightness")
    assert op.resolve() == {"value": 50}
    assert op.resolve({"value": "-30"}) == {"value": -30}
    assert op.resolve({"value": 12.0}) == {"value": 12}
    contrast = get_operation("contrast").resolve({"gain": "2.5"})
    assert contrast == {"gain": 2.5, "center": 128}


@pytest.mark.parametrize(
    ("operation", "params", "message"),
    [
        ("brightness", {"nilai": 3}, "tidak punya parameter nilai"),
        ("brightness", {"value": 300}, "melebihi batas maksimum"),
        ("brightness", {"value": -300}, "di bawah batas minimum"),
        ("brightness", {"value": 1.5}, "bilangan bulat"),
        ("brightness", {"value": "abc"}, "bukan angka"),
        ("brightness", {"value": True}, "bukan boolean"),
        ("brightness", {"value": float("nan")}, "berhingga"),
        ("negative", {"x": 1}, "tidak punya parameter x"),
        ("grayscale", {"method": "rata"}, "tidak valid"),
        ("threshold", {"otsu": "mungkin"}, "boolean"),
        ("gamma", {"gamma": 0}, "di bawah batas minimum"),
        ("mbit", {"bits": 9}, "melebihi batas maksimum"),
    ],
)
def test_resolve_rejects_bad_params(
    operation: str, params: dict[str, object], message: str
) -> None:
    with pytest.raises(ParameterError, match=message):
        get_operation(operation).resolve(params)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        ("ya", True),
        ("TRUE", True),
        ("1", True),
        (False, False),
        ("tidak", False),
        ("0", False),
    ],
)
def test_bool_coercion(raw: object, expected: bool) -> None:
    param = Param("flag", "bool", False, "Flag", "")
    assert param.coerce(raw) is expected


def test_get_operation_unknown_lists_choices() -> None:
    with pytest.raises(OperationError, match="equalize"):
        get_operation("tidak-ada")


def test_primary_parameter() -> None:
    assert get_operation("gamma").primary is not None
    assert get_operation("gamma").primary.name == "gamma"  # type: ignore[union-attr]
    assert get_operation("negative").primary is None
    assert get_operation("grayscale").primary is None  # hanya pilihan, bukan angka


def test_conditional_params_declare_their_trigger() -> None:
    stretch = get_operation("stretch")
    names = {p.name for p in stretch.params}
    for param in stretch.params:
        if param.when:
            assert param.when[0] in names


# --------------------------------------------------------------------------- #
# Perilaku operasi tertentu
# --------------------------------------------------------------------------- #
def test_threshold_slide_problem_through_registry() -> None:
    image = np.array([[40, 160, 128], [127, 250, 20]], dtype=np.uint8)
    out = apply_operation(image, "threshold", {"threshold": 128}).image
    np.testing.assert_array_equal(out, [[0, 255, 255], [0, 255, 0]])


def test_threshold_otsu_separates_two_modes() -> None:
    rng = np.random.default_rng(5)
    dark = rng.normal(60, 8, 2000)
    bright = rng.normal(190, 8, 2000)
    image = np.clip(np.concatenate([dark, bright]), 0, 255).astype(np.uint8).reshape(40, 100)
    result = apply_operation(image, "threshold", {"otsu": True})
    assert "Otsu" in result.note
    t = int(result.note.rsplit("=", 1)[1])
    assert 70 < t < 180
    assert set(np.unique(result.image)) == {0, 255}
    assert (result.image == 255).mean() == pytest.approx(0.5, abs=0.02)


def test_mbit_limits_number_of_levels(color_image: np.ndarray) -> None:
    for bits in (1, 2, 3, 5):
        out = apply_operation(color_image, "mbit", {"bits": bits}).image
        assert len(np.unique(out)) <= 2**bits


def test_grayscale_methods_differ_on_color(color_image: np.ndarray) -> None:
    average = apply_operation(color_image, "grayscale", {"method": "average"}).image
    luminance = apply_operation(color_image, "grayscale", {"method": "luminance"}).image
    assert not np.array_equal(average, luminance)


def test_stretch_percentile_widens_a_narrow_histogram() -> None:
    rng = np.random.default_rng(2)
    narrow = np.clip(rng.normal(120, 6, (60, 60)), 90, 150).astype(np.uint8)
    result = apply_operation(narrow, "stretch", {"mode": "percentile", "percent": 1})
    assert result.image.std() > 3 * narrow.std()
    assert result.image.min() < 10 and result.image.max() > 245
    assert result.lut is not None and "r1=" in result.note


def test_stretch_minmax_maps_extremes_to_full_range() -> None:
    image = np.array([[50, 100], [150, 200]], dtype=np.uint8)
    out = apply_operation(image, "stretch", {"mode": "minmax"}).image
    assert out.min() == 0 and out.max() == 255


def test_stretch_flat_image_is_returned_as_is() -> None:
    flat = np.full((8, 8), 77, dtype=np.uint8)
    result = apply_operation(flat, "stretch", {"mode": "minmax"})
    np.testing.assert_array_equal(result.image, flat)
    assert result.lut is None and "datar" in result.note


def test_stretch_custom_matches_stretch_lut(ramp: np.ndarray) -> None:
    params = {"mode": "custom", "r1": 70, "s1": 30, "r2": 180, "s2": 225}
    result = apply_operation(ramp, "stretch", params)
    assert result.lut is not None
    np.testing.assert_array_equal(result.image.ravel(), result.lut)


def test_stretch_custom_rejects_inverted_points(ramp: np.ndarray) -> None:
    with pytest.raises(ParameterError):
        apply_operation(ramp, "stretch", {"mode": "custom", "r1": 200, "r2": 100})


def test_channel_modes_return_no_single_curve(color_image: np.ndarray) -> None:
    for name in ("stretch", "equalize"):
        result = apply_operation(color_image, name, {"color_mode": "channels"})
        assert result.lut is None and result.image.shape == color_image.shape
    gray_result = apply_operation(color_image[:, :, 0], "equalize", {"color_mode": "channels"})
    assert gray_result.lut is not None  # grayscale selalu punya satu kurva


def test_equalize_result_curve_matches_applied_mapping(ramp: np.ndarray) -> None:
    result = apply_operation(ramp, "equalize")
    assert result.lut is not None
    np.testing.assert_array_equal(result.image.ravel(), result.lut)


def test_slice_preserve_flag(ramp: np.ndarray) -> None:
    plain = apply_operation(ramp, "slice", {"low": 100, "high": 120}).image.ravel()
    kept = apply_operation(ramp, "slice", {"low": 100, "high": 120, "preserve": True}).image.ravel()
    assert plain[50] == 0 and kept[50] == 50 and plain[110] == kept[110] == 255


def test_bitplane_top_bit_is_a_threshold_at_128(gray_image: np.ndarray) -> None:
    top = apply_operation(gray_image, "bitplane", {"bit": 7}).image
    thr = apply_operation(gray_image, "threshold", {"threshold": 128}).image
    np.testing.assert_array_equal(top, thr)


def test_brightness_roundtrip_where_not_clipped() -> None:
    mid = np.full((4, 4), 100, dtype=np.uint8)
    up = apply_operation(mid, "brightness", {"value": 75}).image
    down = apply_operation(up, "brightness", {"value": -75}).image
    assert up[0, 0] == 175 and down[0, 0] == 100


def test_invalid_kernel_is_a_parameter_error(color_image: np.ndarray) -> None:
    with pytest.raises(ParameterError):
        apply_operation(color_image, "median", {"ksize": 4})
