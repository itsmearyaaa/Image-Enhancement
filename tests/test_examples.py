from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from conftest import ROOT

EXAMPLES = ROOT / "examples"


def test_soal_materi_prints_the_slide_answers(capsys: pytest.CaptureFixture[str]) -> None:
    namespace = runpy.run_path(str(EXAMPLES / "soal_materi.py"), run_name="not_main")
    assert namespace["main"]() == 0
    out = capsys.readouterr().out
    assert "X baru = 64" in out
    assert "0.14" in out and "0.43" in out and "0.71" in out and "0.86" in out
    assert "985 piksel" in out and "448 piksel" in out and "1023 piksel" in out
    # soal thresholding: baris pertama hasil adalah 0 255 0 255 0
    assert "   0  255    0  255    0" in out
    # contoh kontras: baris pertama hasil adalah 20 50 70 90 80
    assert "  20   50   70   90   80" in out


def test_usage_example_runs_and_writes_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["usage.py"])
    namespace = runpy.run_path(str(EXAMPLES / "usage.py"), run_name="not_main")
    assert namespace["main"]() == 0
    assert "gelap" in capsys.readouterr().out
    assert (tmp_path / "output" / "sample_astronaut_gelap_contoh.png").is_file()
    assert (tmp_path / "output" / "sample_astronaut_gelap_contoh_compare.png").is_file()


def test_usage_example_reports_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["usage.py", str(tmp_path / "hilang.png")])
    namespace = runpy.run_path(str(EXAMPLES / "usage.py"), run_name="not_main")
    assert namespace["main"]() == 1
    assert "Galat" in capsys.readouterr().err


def test_generate_samples_script_is_importable_without_skimage() -> None:
    namespace = runpy.run_path(str(EXAMPLES / "generate_samples.py"), run_name="not_main")
    import numpy as np

    mid = np.full((4, 4, 3), 128.0)
    assert namespace["too_dark"](mid).mean() < 60
    assert namespace["too_bright"](mid).mean() > 200
    ramp = np.tile(np.linspace(0, 255, 16), (4, 1))[:, :, None].repeat(3, axis=2)
    assert namespace["low_contrast"](ramp).std() < 0.3 * ramp.std()
