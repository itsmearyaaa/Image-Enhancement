"""Buat ulang gambar contoh di ``examples/images/``.

Gambar contoh dibuat dari foto bawaan scikit-image yang berlisensi bebas (lihat
``examples/images/README.md``) lalu, untuk tiga di antaranya, sengaja dirusak supaya masalah yang
dibahas pada materi (terlalu gelap, kontras rendah, terlalu terang) terlihat jelas.

Skrip ini hanya dibutuhkan bila gambar contoh ingin dibuat ulang; file hasilnya sudah disertakan.

    pip install scikit-image
    python examples/generate_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imgenhance import write_image  # noqa: E402

OUT_DIR = ROOT / "examples" / "images"


def _to_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def too_dark(rgb: np.ndarray) -> np.ndarray:
    """Gelap: nilai ditekan ke bawah (gamma besar + redup) seperti foto kurang cahaya."""
    return _to_u8(255 * 0.75 * (rgb / 255.0) ** 2.6)


def low_contrast(rgb: np.ndarray) -> np.ndarray:
    """Kontras rendah / pudar: seluruh intensitas dipadatkan ke sekitar abu-abu tengah."""
    return _to_u8(128 + 0.28 * (rgb.astype(np.float64) - 128))


def too_bright(rgb: np.ndarray) -> np.ndarray:
    """Terlalu terang: gamma kecil + penguatan sehingga banyak piksel terpotong di 255."""
    return _to_u8(255 * np.clip(1.25 * (rgb / 255.0) ** 0.45, 0, 1))


def main() -> int:
    try:
        from skimage import data
    except ImportError:
        print("Butuh scikit-image: pip install scikit-image", file=sys.stderr)
        return 1

    def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(rgb[:, :, ::-1])

    samples = {
        "sample_astronaut_gelap.png": rgb_to_bgr(too_dark(data.astronaut().astype(np.float64))),
        "sample_chelsea_pudar.png": rgb_to_bgr(low_contrast(data.chelsea().astype(np.float64))),
        "sample_coffee_terang.png": rgb_to_bgr(too_bright(data.coffee().astype(np.float64))),
        "sample_camera.png": np.ascontiguousarray(data.camera()),
        "sample_coins.png": np.ascontiguousarray(data.coins()),
    }
    for name, image in samples.items():
        path = write_image(OUT_DIR / name, image)
        print(
            f"{path.relative_to(ROOT)}  {image.shape[1]}x{image.shape[0]}"
            f"  rata-rata {image.mean():.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
