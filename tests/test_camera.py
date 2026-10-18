from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from imgenhance import CameraError, OperationError
from imgenhance import camera as camera_module
from imgenhance.camera import LiveState, process_frame, run_webcam
from imgenhance.operations import list_operations


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 120, size=(240, 320, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# LiveState
# --------------------------------------------------------------------------- #
def test_state_starts_on_requested_operation_with_defaults() -> None:
    state = LiveState.for_operation("gamma")
    assert state.operation.name == "gamma"
    assert state.params == state.operation.defaults()


def test_state_unknown_operation() -> None:
    with pytest.raises(OperationError):
        LiveState.for_operation("tidak-ada")


def test_cycle_wraps_around_and_resets_params() -> None:
    state = LiveState.for_operation("gamma")
    state.adjust(+1)
    names = [op.name for op in list_operations()]
    state.cycle(+1)
    assert state.operation.name == names[(names.index("gamma") + 1) % len(names)]
    assert state.params == state.operation.defaults()
    last = LiveState(index=len(names) - 1)
    last.cycle(+1)
    assert last.index == 0
    first = LiveState()
    first.cycle(-1)
    assert first.index == len(names) - 1


def test_adjust_steps_and_clamps_to_range() -> None:
    state = LiveState.for_operation("brightness")
    assert state.params["value"] == 50
    state.adjust(+1)
    assert state.params["value"] == 55
    for _ in range(200):
        state.adjust(+1)
    assert state.params["value"] == 255
    for _ in range(200):
        state.adjust(-1)
    assert state.params["value"] == -255


def test_adjust_float_and_odd_kernel_sizes() -> None:
    state = LiveState.for_operation("gamma")
    state.adjust(+1)
    assert state.params["gamma"] == pytest.approx(0.55)
    blur = LiveState.for_operation("median")
    blur.adjust(+1)
    assert blur.params["ksize"] == 7 and blur.params["ksize"] % 2 == 1


def test_adjust_does_nothing_without_a_numeric_parameter() -> None:
    state = LiveState.for_operation("negative")
    state.adjust(+1)
    assert state.params == {}
    gray = LiveState.for_operation("grayscale")
    gray.adjust(+1)
    assert gray.params == gray.operation.defaults()


def test_reset_restores_defaults() -> None:
    state = LiveState.for_operation("contrast")
    state.adjust(+1)
    state.reset()
    assert state.params == state.operation.defaults()


def test_detail_hides_irrelevant_conditional_params() -> None:
    state = LiveState.for_operation("stretch")
    assert "percent=" in state.detail() and "r1=" not in state.detail()
    state.params["mode"] = "custom"
    assert "r1=" in state.detail() and "percent=" not in state.detail()
    assert LiveState.for_operation("negative").detail() == "(tanpa parameter)"


# --------------------------------------------------------------------------- #
# process_frame
# --------------------------------------------------------------------------- #
def test_process_frame_returns_annotated_display_and_result() -> None:
    frame = _frame()
    state = LiveState.for_operation("brightness")
    display, result = process_frame(frame, state)
    assert display.shape == frame.shape and display.dtype == np.uint8
    assert result.image.mean() > frame.mean()


def test_process_frame_compare_mode_is_twice_as_wide() -> None:
    frame = _frame()
    state = LiveState.for_operation("negative")
    state.compare = True
    display, _ = process_frame(frame, state)
    assert display.shape[1] == 2 * frame.shape[1]


def test_process_frame_works_for_every_operation() -> None:
    frame = _frame()
    for index in range(len(list_operations())):
        state = LiveState(index=index)
        display, _ = process_frame(frame, state)
        assert display.shape == frame.shape  # operasi grayscale pun ditampilkan sebagai BGR


# --------------------------------------------------------------------------- #
# run_webcam dengan kamera palsu
# --------------------------------------------------------------------------- #
class _FakeCapture:
    def __init__(self, frames: int = 20, opened: bool = True) -> None:
        self._frames, self._opened, self.released = frames, opened, False

    def isOpened(self) -> bool:  # noqa: N802 - mengikuti API OpenCV
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._frames <= 0:
            return False, None
        self._frames -= 1
        return True, _frame()

    def release(self) -> None:
        self.released = True


def _patch_cv2(monkeypatch: pytest.MonkeyPatch, capture: _FakeCapture, keys: list[int]) -> None:
    iterator = iter(keys)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: capture)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *_: next(iterator))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)


def test_run_webcam_raises_when_camera_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _FakeCapture(opened=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: capture)
    with pytest.raises(CameraError, match="tidak bisa dibuka"):
        run_webcam("equalize", camera_index=7)
    assert capture.released


def test_run_webcam_validates_operation_before_opening_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[int] = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: opened.append(1))
    with pytest.raises(OperationError):
        run_webcam("tidak-ada")
    assert opened == []


def test_run_webcam_handles_every_key_and_saves_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture = _FakeCapture()
    keys = [ord(k) for k in "n+-rhcps"] + [27]  # ... lalu ESC
    _patch_cv2(monkeypatch, capture, keys)
    run_webcam("equalize", output_dir=tmp_path)
    assert capture.released
    assert len(list(tmp_path.glob("webcam_*.png"))) == 1


def test_run_webcam_q_quits_immediately(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    capture = _FakeCapture()
    _patch_cv2(monkeypatch, capture, [ord("q")])
    run_webcam(output_dir=tmp_path)
    assert capture.released and not list(tmp_path.iterdir())


def test_run_webcam_raises_when_frames_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _FakeCapture(frames=0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: capture)
    with pytest.raises(CameraError, match="frame"):
        run_webcam()
    assert capture.released


def test_hint_mentions_the_main_keys() -> None:
    for key in ("n/p", "+/-", "q"):
        assert key in camera_module.HINT


def test_run_webcam_original_error_is_not_masked_by_headless_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_gui() -> None:
        raise cv2.error("The function is not implemented. Rebuild the library with GTK")

    capture = _FakeCapture(frames=0)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: capture)
    monkeypatch.setattr(cv2, "destroyAllWindows", no_gui)
    with pytest.raises(CameraError, match="frame"):  # bukan cv2.error dari pembersihan
        run_webcam()
    assert capture.released
