"""Statistik gambar dan diagnosis histogram (terang / gelap / kontras rendah).

Materi P03 hlm. 14-15: *dari histogram dapat dilihat apakah sebuah citra lebih banyak warna gelap
atau terang*. Modul ini mengotomatiskan pembacaan itu dan menyarankan operasi perbaikannya.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._types import Gray, Image
from .histogram import histogram, probabilities
from .pipeline import Pipeline, Step
from .point import luminance

#: Rata-rata keabuan di bawah / di atas batas ini dianggap gelap / terang (sepertiga rentang).
DARK_MEAN = 85.0
BRIGHT_MEAN = 170.0
#: Selisih persentil ke-2 dan ke-98 di bawah nilai ini dianggap kontras rendah
#: (kurang dari separuh rentang).
LOW_CONTRAST_SPREAD = 128
_SPREAD_PERCENTILES = (2.0, 98.0)


@dataclass(frozen=True)
class ImageStats:
    """Ringkasan histogram luminans sebuah gambar."""

    mean: float
    std: float
    minimum: int
    maximum: int
    spread: int  #: selisih persentil ke-98 dan ke-2 (tahan terhadap piksel pencilan)
    entropy: float  #: entropi Shannon (bit); makin merata histogram, makin besar (maks 8)

    @classmethod
    def from_gray(cls, gray: Gray) -> ImageStats:
        probs = probabilities(histogram(gray))
        nonzero = probs[probs > 0]
        low, high = np.percentile(gray, _SPREAD_PERCENTILES)
        return cls(
            mean=float(gray.mean()),
            std=float(gray.std()),
            minimum=int(gray.min()),
            maximum=int(gray.max()),
            spread=round(high - low),
            entropy=float(-(nonzero * np.log2(nonzero)).sum()),
        )

    @classmethod
    def from_image(cls, image: Image) -> ImageStats:
        return cls.from_gray(luminance(image))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "min": self.minimum,
            "max": self.maximum,
            "spread": self.spread,
            "entropy": round(self.entropy, 3),
        }


@dataclass(frozen=True)
class Diagnosis:
    """Hasil pembacaan histogram beserta saran perbaikannya."""

    brightness: str  #: "gelap" | "normal" | "terang"
    contrast: str  #: "rendah" | "baik"
    summary: str
    steps: tuple[Step, ...]  #: langkah yang disarankan (kosong bila tidak perlu perbaikan)

    @property
    def needs_enhancement(self) -> bool:
        return bool(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "summary": self.summary,
            "steps": [s.label() for s in self.steps],
        }


def diagnose(stats: ImageStats) -> Diagnosis:
    """Baca kecerahan dan kontras dari ``stats`` lalu sarankan langkah perbaikan.

    Aturan (ambang ada di konstanta modul ini):

    ===================  ==========================  ============================================
    Kecerahan            Kontras                     Saran
    ===================  ==========================  ============================================
    gelap / terang       rendah                      ``equalize`` (meratakan sekaligus keduanya)
    gelap                baik                        ``gamma`` 0.5 (cerahkan tanpa memotong putih)
    terang               baik                        ``gamma`` 2.0 (gelapkan tanpa memotong hitam)
    normal               rendah                      ``stretch`` persentil (regangkan histogram)
    normal               baik                        tidak perlu perbaikan
    ===================  ==========================  ============================================
    """
    if stats.mean < DARK_MEAN:
        brightness = "gelap"
    elif stats.mean > BRIGHT_MEAN:
        brightness = "terang"
    else:
        brightness = "normal"
    contrast = "rendah" if stats.spread < LOW_CONTRAST_SPREAD else "baik"

    steps: tuple[Step, ...]
    if brightness != "normal" and contrast == "rendah":
        steps = (Step("equalize"),)
        advice = "histogram menumpuk di satu sisi dan sempit; histogram equalization meratakannya"
    elif brightness == "gelap":
        steps = (Step("gamma", {"gamma": 0.5}),)
        advice = "gambar gelap tetapi rentangnya cukup lebar; gamma 0.5 mencerahkan area gelap"
    elif brightness == "terang":
        steps = (Step("gamma", {"gamma": 2.0}),)
        advice = "gambar terang tetapi rentangnya cukup lebar; gamma 2.0 menggelapkan area terang"
    elif contrast == "rendah":
        steps = (Step("stretch", {"mode": "percentile", "percent": 1.0}),)
        advice = "kecerahan wajar tetapi histogram sempit; contrast stretching melebarkannya"
    else:
        steps = ()
        advice = "kecerahan dan kontras sudah baik; tidak perlu perbaikan"

    summary = (
        f"Gambar {brightness} (rata-rata {stats.mean:.0f}/255), kontras {contrast} "
        f"(sebaran {stats.spread}/255): {advice}."
    )
    return Diagnosis(brightness, contrast, summary, steps)


def diagnose_image(image: Image) -> Diagnosis:
    """Pintasan: statistik + diagnosis untuk ``image``."""
    return diagnose(ImageStats.from_image(image))


def auto_pipeline(image: Image) -> Pipeline | None:
    """Pipeline hasil saran diagnosis, atau ``None`` bila gambar sudah baik."""
    diagnosis = diagnose_image(image)
    return Pipeline(diagnosis.steps) if diagnosis.steps else None
