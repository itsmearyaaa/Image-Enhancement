from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from imgenhance import ImageStats, ImageWriteError, diagnose
from imgenhance.report import (
    CSV_FIELDS,
    csv_row,
    format_diagnosis,
    format_stats_table,
    format_text,
    to_json,
    write_csv,
)


@pytest.fixture
def stats() -> tuple[ImageStats, ImageStats]:
    rng = np.random.default_rng(0)
    dark = np.clip(rng.normal(50, 10, (40, 40)), 0, 255).astype(np.uint8)
    return ImageStats.from_image(dark), ImageStats.from_image(255 - dark)


def test_format_text_contains_name_steps_notes_and_numbers(
    stats: tuple[ImageStats, ImageStats],
) -> None:
    before, after = stats
    text = format_text("foto.png", "negative", before, after, ["catatan uji"])
    assert "=== foto.png ===" in text and "Operasi : negative" in text and "catatan uji" in text
    assert f"{before.mean:.1f}" in text and f"{after.mean:.1f}" in text
    assert "Asli" in format_stats_table(before, after)


def test_format_diagnosis_with_and_without_suggestion(stats: tuple[ImageStats, ImageStats]) -> None:
    dark, _ = stats
    assert "Saran" in format_diagnosis("a.png", dark, diagnose(dark))
    fine = ImageStats(128, 60, 0, 255, 200, 7.5)
    assert "Saran" not in format_diagnosis("b.png", fine, diagnose(fine))


def test_to_json_keeps_indonesian_text() -> None:
    parsed = json.loads(to_json({"kecerahan": "gelap", "angka": [1, 2]}))
    assert parsed == {"kecerahan": "gelap", "angka": [1, 2]}
    assert "ö" in to_json({"x": "ö"})


def test_write_csv_has_bom_header_and_rows(
    tmp_path: Path, stats: tuple[ImageStats, ImageStats]
) -> None:
    before, after = stats
    path = write_csv(
        tmp_path / "out" / "ringkasan.csv",
        [csv_row("a.png", "negative", before, after, "a_negative.png")],
    )
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM UTF-8 agar Excel membaca dengan benar
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(CSV_FIELDS)
    assert rows[0]["file"] == "a.png" and float(rows[0]["rata2_asli"]) == pytest.approx(
        before.mean, abs=0.01
    )


def test_write_csv_wraps_os_errors(tmp_path: Path) -> None:
    blocker = tmp_path / "berkas"
    blocker.write_text("x")
    with pytest.raises(ImageWriteError):
        write_csv(blocker / "x.csv", [])
