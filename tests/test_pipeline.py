from __future__ import annotations

import numpy as np
import pytest

from imgenhance import OperationError, ParameterError, Pipeline, Step, apply_operation


def test_step_parse_name_only_and_with_params() -> None:
    assert Step.parse("equalize") == Step("equalize", {})
    step = Step.parse(" gamma : gamma=0.6 , color_mode=luminance ")
    assert step.name == "gamma"
    assert dict(step.params) == {"gamma": "0.6", "color_mode": "luminance"}


@pytest.mark.parametrize("spec", ["", "  ", ":gamma=1", "gamma:0.5", "gamma:=2"])
def test_step_parse_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(ParameterError):
        Step.parse(spec)


def test_step_label() -> None:
    assert Step("equalize").label() == "equalize"
    assert Step("gamma", {"gamma": 0.5}).label() == "gamma(gamma=0.5)"


def test_pipeline_validates_at_construction() -> None:
    with pytest.raises(OperationError):
        Pipeline.parse(["tidak-ada"])
    with pytest.raises(ParameterError):
        Pipeline.parse(["gamma:gamma=99"])
    with pytest.raises(ParameterError):
        Pipeline.parse(["gamma:warna=1"])
    with pytest.raises(ParameterError):
        Pipeline([])


def test_pipeline_runs_in_order(color_image: np.ndarray) -> None:
    result = Pipeline.parse(["grayscale", "negative"]).run(color_image)
    expected = 255 - apply_operation(color_image, "grayscale").image
    np.testing.assert_array_equal(result.image, expected)
    assert [s.step.name for s in result.stages] == ["grayscale", "negative"]


def test_two_negatives_cancel(color_image: np.ndarray) -> None:
    result = Pipeline.parse(["negative", "negative"]).run(color_image)
    np.testing.assert_array_equal(result.image, color_image)


def test_lut_only_available_for_single_step(color_image: np.ndarray) -> None:
    assert Pipeline.parse(["negative"]).run(color_image).lut is not None
    assert Pipeline.parse(["negative", "gamma"]).run(color_image).lut is None


def test_pipeline_collects_notes_and_resolved_params(gray_image: np.ndarray) -> None:
    result = Pipeline.parse(["stretch:mode=minmax", "mbit:bits=3"]).run(gray_image)
    assert any("r1=" in n for n in result.notes) and any("8 tingkat" in n for n in result.notes)
    assert result.stages[1].params["bits"] == 3
    assert result.stages[1].params["rescale"] is False


def test_pipeline_label() -> None:
    assert Pipeline.parse(["grayscale", "equalize"]).label() == "grayscale > equalize"
