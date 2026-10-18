from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "examples" / "images"


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    assert SAMPLES_DIR.is_dir(), "jalankan: python examples/generate_samples.py"
    return SAMPLES_DIR


@pytest.fixture
def ramp() -> np.ndarray:
    """Gambar grayscale 16x16 yang berisi 0..255 persis sekali: hasil operasi = isi LUT-nya."""
    return np.arange(256, dtype=np.uint8).reshape(16, 16)


@pytest.fixture
def color_image() -> np.ndarray:
    """Gambar BGR acak yang tetap (seed) agar uji deterministik."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


@pytest.fixture
def gray_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(40, 56), dtype=np.uint8)
