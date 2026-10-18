"""Format laporan: teks untuk terminal, JSON, dan CSV ringkasan batch."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .analysis import Diagnosis, ImageStats
from .errors import ImageWriteError

CSV_FIELDS = (
    "file",
    "langkah",
    "rata2_asli",
    "std_asli",
    "sebaran_asli",
    "entropi_asli",
    "rata2_hasil",
    "std_hasil",
    "sebaran_hasil",
    "entropi_hasil",
    "keluaran",
)


def format_stats_table(before: ImageStats, after: ImageStats) -> str:
    """Tabel statistik dua kolom (asli / hasil)."""
    rows = (
        ("Rata-rata", f"{before.mean:.1f}", f"{after.mean:.1f}"),
        ("Simpangan baku", f"{before.std:.1f}", f"{after.std:.1f}"),
        ("Min - Maks", f"{before.minimum}-{before.maximum}", f"{after.minimum}-{after.maximum}"),
        ("Sebaran p2-p98", str(before.spread), str(after.spread)),
        ("Entropi (bit)", f"{before.entropy:.2f}", f"{after.entropy:.2f}"),
    )
    lines = [f"  {'':<16}{'Asli':>10}{'Hasil':>10}"]
    lines += [f"  {label:<16}{a:>10}{b:>10}" for label, a, b in rows]
    return "\n".join(lines)


def format_text(
    name: str,
    steps: str,
    before: ImageStats,
    after: ImageStats,
    notes: Sequence[str] = (),
) -> str:
    """Laporan teks satu gambar: operasi yang dipakai + perbandingan statistik."""
    lines = [f"=== {name} ===", f"Operasi : {steps}"]
    lines += [f"Catatan : {note}" for note in notes]
    lines.append(format_stats_table(before, after))
    return "\n".join(lines)


def format_diagnosis(name: str, stats: ImageStats, diagnosis: Diagnosis) -> str:
    """Laporan teks hasil ``analyze``: statistik, diagnosis, dan saran."""
    lines = [
        f"=== {name} ===",
        f"Rata-rata keabuan : {stats.mean:.1f} / 255",
        f"Simpangan baku    : {stats.std:.1f}",
        f"Min - Maks        : {stats.minimum} - {stats.maximum}",
        f"Sebaran p2-p98    : {stats.spread}",
        f"Entropi           : {stats.entropy:.2f} bit",
        f"Kecerahan         : {diagnosis.brightness}",
        f"Kontras           : {diagnosis.contrast}",
        f"Diagnosis         : {diagnosis.summary}",
    ]
    if diagnosis.steps:
        suggestion = " > ".join(step.label() for step in diagnosis.steps)
        lines.append(f"Saran             : imgenhance auto <gambar>   (= {suggestion})")
    return "\n".join(lines)


def to_json(data: Any) -> str:
    """JSON berindentasi; karakter non-ASCII dibiarkan apa adanya."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def csv_row(
    name: str, steps: str, before: ImageStats, after: ImageStats, output: str
) -> dict[str, Any]:
    return {
        "file": name,
        "langkah": steps,
        "rata2_asli": round(before.mean, 2),
        "std_asli": round(before.std, 2),
        "sebaran_asli": before.spread,
        "entropi_asli": round(before.entropy, 3),
        "rata2_hasil": round(after.mean, 2),
        "std_hasil": round(after.std, 2),
        "sebaran_hasil": after.spread,
        "entropi_hasil": round(after.entropy, 3),
        "keluaran": output,
    }


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Tulis ringkasan batch ke CSV (UTF-8 dengan BOM agar terbuka benar di Excel)."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise ImageWriteError(f"Gagal menulis {path}: {exc}") from exc
    return path
