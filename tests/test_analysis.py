from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imgenhance import (
    ImageStats,
    Pipeline,
    auto_pipeline,
    diagnose,
    diagnose_image,
    read_image,
)


def test_stats_of_a_uniform_ramp() -> None:
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    stats = ImageStats.from_image(ramp)
    assert stats.mean == pytest.approx(127.5)
    assert stats.minimum == 0 and stats.maximum == 255
    assert stats.entropy == pytest.approx(8.0)  # semua 256 tingkat muncul sama banyak
    assert stats.spread > 240


def test_stats_of_a_constant_image() -> None:
    stats = ImageStats.from_image(np.full((5, 5), 90, dtype=np.uint8))
    assert stats.mean == 90 and stats.std == 0 and stats.entropy == 0 and stats.spread == 0


def test_stats_use_luminance_for_color_images(color_image: np.ndarray) -> None:
    stats = ImageStats.from_image(color_image)
    assert 0 < stats.mean < 255
    assert set(stats.to_dict()) == {"mean", "std", "min", "max", "spread", "entropy"}


def _wide(mean: float, std: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, std, (100, 100)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    ("image", "brightness", "contrast", "first_step"),
    [
        (_wide(40, 12), "gelap", "rendah", "equalize"),
        (_wide(215, 10), "terang", "rendah", "equalize"),
        (_wide(128, 8), "normal", "rendah", "stretch"),
        (_wide(128, 60), "normal", "baik", None),
        (_wide(55, 45), "gelap", "baik", "gamma"),
        (_wide(200, 40), "terang", "baik", "gamma"),
    ],
)
def test_diagnosis_rules(
    image: np.ndarray, brightness: str, contrast: str, first_step: str | None
) -> None:
    diagnosis = diagnose(ImageStats.from_image(image))
    assert (diagnosis.brightness, diagnosis.contrast) == (brightness, contrast)
    assert (diagnosis.steps[0].name if diagnosis.steps else None) == first_step
    assert diagnosis.needs_enhancement == (first_step is not None)
    assert diagnosis.summary.startswith("Gambar ")


def test_gamma_suggestion_direction() -> None:
    dark = diagnose(ImageStats.from_image(_wide(55, 45)))
    bright = diagnose(ImageStats.from_image(_wide(200, 40)))
    assert dark.steps[0].params["gamma"] < 1 < bright.steps[0].params["gamma"]


def test_diagnosis_to_dict_is_json_friendly() -> None:
    data = diagnose(ImageStats.from_image(_wide(40, 12))).to_dict()
    assert data["brightness"] == "gelap" and data["steps"] == ["equalize"]


def test_auto_pipeline_none_when_image_is_fine() -> None:
    assert auto_pipeline(_wide(128, 60)) is None


@pytest.mark.parametrize(
    ("sample", "expected_step"),
    [
        ("sample_astronaut_gelap.png", "gamma"),
        ("sample_chelsea_pudar.png", "stretch"),
        ("sample_coffee_terang.png", "gamma"),
    ],
)
def test_sample_images_get_the_intended_diagnosis(
    samples_dir: Path, sample: str, expected_step: str
) -> None:
    image = read_image(samples_dir / sample)
    assert diagnose_image(image).steps[0].name == expected_step


def test_sample_images_that_are_fine_need_nothing(samples_dir: Path) -> None:
    for sample in ("sample_camera.png", "sample_coins.png"):
        assert auto_pipeline(read_image(samples_dir / sample)) is None


@pytest.mark.parametrize(
    "sample", ["sample_astronaut_gelap.png", "sample_chelsea_pudar.png", "sample_coffee_terang.png"]
)
def test_auto_enhancement_moves_the_image_toward_the_normal_range(
    samples_dir: Path, sample: str
) -> None:
    image = read_image(samples_dir / sample)
    pipeline = auto_pipeline(image)
    assert isinstance(pipeline, Pipeline)
    before = ImageStats.from_image(image)
    after = ImageStats.from_image(pipeline.run(image).image)
    assert abs(after.mean - 128) < abs(before.mean - 128) or after.spread > before.spread
    # sesudah diperbaiki, gambar tidak lagi dianggap bermasalah
    assert diagnose(after).brightness == "normal" and diagnose(after).contrast == "baik"
