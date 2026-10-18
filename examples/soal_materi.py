"""Mengerjakan contoh dan soal materi P03 (Peningkatan Kualitas Citra) dengan library imgenhance.

    python examples/soal_materi.py

Setiap bagian mencetak masukan, hasil, dan nomor halaman materinya sehingga bisa dicocokkan langsung
dengan slide. Hasil yang sama juga dijaga oleh uji otomatis (tests/test_point.py dan
tests/test_histogram.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imgenhance import (  # noqa: E402
    apply_lut,
    contrast_lut,
    equalization_map,
    quantize_lut,
    threshold_lut,
)
from imgenhance.histogram import cumulative_distribution, probabilities  # noqa: E402


def _matrix(title: str, values: np.ndarray) -> None:
    print(title)
    for row in values:
        print("   " + " ".join(f"{int(v):4d}" for v in row))


def soal_thresholding() -> None:
    print("=== Soal thresholding (materi hlm. 29) ===")
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
    _matrix("Citra:", image)
    _matrix("Hasil (< 128 menjadi 0, >= 128 menjadi 255):", apply_lut(image, threshold_lut(128)))


def contoh_kontras() -> None:
    print("\n=== Contoh pengaturan kontras G = 2, P = 10 (materi hlm. 26) ===")
    image = np.array(
        [
            [15, 30, 40, 50, 45],
            [25, 25, 25, 40, 30],
            [40, 50, 35, 45, 35],
            [50, 40, 15, 25, 40],
            [30, 40, 20, 25, 20],
        ],
        dtype=np.uint8,
    )
    _matrix("Matriks asal:", image)
    _matrix("Matriks hasil, s = G*(r - P) + P:", apply_lut(image, contrast_lut(gain=2, center=10)))


def contoh_mbit() -> None:
    print("\n=== Contoh grayscale ke m-bit (materi hlm. 12) ===")
    print(
        "X = 100 dengan langkah kuantisasi 64 (4 tingkat = 2 bit): X baru =", quantize_lut(2)[100]
    )
    print("Catatan: materi menulis '4 bit (0-64)'; langkah 64 sesuai 2 bit karena 256 / 2^2 = 64.")
    print("Dengan 4 bit sungguhan (16 tingkat, langkah 16): X baru =", quantize_lut(4)[100])


def contoh_equalization() -> None:
    print("\n=== Contoh histogram equalization L = 8, n = 64 x 64 (materi hlm. 17-21) ===")
    counts = np.array([790, 1023, 850, 656, 329, 245, 122, 81], dtype=np.int64)
    probs, cdf, mapping = (
        probabilities(counts),
        cumulative_distribution(counts),
        equalization_map(counts),
    )
    print(" k   r_k    n_k   P(r_k)   s_k (kumulatif)   dibulatkan ke")
    for k, n in enumerate(counts):
        print(
            f" {k}   {k / 7:.2f}  {n:5d}   {probs[k]:.2f}      {cdf[k]:.2f}"
            f"              {mapping[k] / 7:.2f}"
        )
    print("\nJumlah piksel hasil transformasi:")
    for level in np.unique(mapping):
        total = int(counts[mapping == level].sum())
        print(f"   s = {level / 7:.2f}: {total} piksel (P = {total / counts.sum():.2f})")
    print(
        "Catatan: tabel materi hlm. 20 menulis '254 + 122 + 81 = 448' dan P = 0.23 untuk s = 0.86; "
        "hitungan yang benar adalah 245 + 122 + 81 = 448 dan 985 / 4096 = 0.24."
    )


def main() -> int:
    soal_thresholding()
    contoh_kontras()
    contoh_mbit()
    contoh_equalization()
    return 0


if __name__ == "__main__":
    sys.exit(main())
