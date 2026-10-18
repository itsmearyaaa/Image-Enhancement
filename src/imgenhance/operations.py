"""Daftar operasi peningkatan citra (*registry*) beserta parameternya.

Setiap operasi dideskripsikan oleh data (:class:`Operation` + :class:`Param`): nama, rumus, rujukan
materi, dan parameter bertipe lengkap dengan rentang yang valid. Satu deskripsi dipakai bersama oleh
semua lapisan, jadi menambah operasi baru cukup menambah satu entri di sini:

* CLI membuat sub-perintah dan opsinya dari daftar ini (``imgenhance apply <operasi> --help``);
* demo web membuat kontrol (slider / pilihan) dari ``GET /api/operations``;
* webcam memakai parameter pertama sebagai nilai yang diatur dengan tombol ``+`` / ``-``;
* validasi parameter (tipe, rentang, nama tidak dikenal) ada di satu tempat:
  :meth:`Operation.resolve`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from ._types import Image, Lut
from .errors import OperationError, ParameterError
from .histogram import equalize, equalize_lut, histogram
from .point import (
    GRAY_WEIGHTS,
    MAX_LEVEL,
    apply_lut,
    bitplane_lut,
    brightness_lut,
    contrast_lut,
    gamma_lut,
    is_gray,
    log_lut,
    luminance,
    negative_lut,
    quantize_lut,
    slice_lut,
    stretch_lut,
    threshold_lut,
    to_grayscale,
)
from .spatial import SMOOTH_METHODS, median_filter, sharpen, smooth

ParamKind = Literal["int", "float", "choice", "bool"]

COLOR_MODES = ("luminance", "channels")
_TRUE = frozenset({"true", "1", "ya", "yes", "y", "on"})
_FALSE = frozenset({"false", "0", "tidak", "no", "n", "off"})


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Param:
    """Satu parameter operasi: tipe, nilai bawaan, rentang, dan teks bantuan."""

    name: str
    kind: ParamKind
    default: Any
    label: str
    help: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    #: ``(nama_param, nilai)``: parameter hanya relevan bila parameter lain bernilai tertentu.
    when: tuple[str, str] | None = None

    def coerce(self, value: Any) -> Any:
        """Ubah ``value`` (mis. string dari CLI) ke tipe parameter dan periksa rentangnya."""
        if self.kind == "bool":
            return self._coerce_bool(value)
        if self.kind == "choice":
            if value not in self.choices:
                raise ParameterError(
                    f"{self.name}: '{value}' tidak valid; pilih salah satu dari "
                    f"{', '.join(self.choices)}"
                )
            return str(value)
        number = self._coerce_number(value)
        if self.minimum is not None and number < self.minimum:
            raise ParameterError(f"{self.name}: {number} di bawah batas minimum {self.minimum:g}")
        if self.maximum is not None and number > self.maximum:
            raise ParameterError(f"{self.name}: {number} melebihi batas maksimum {self.maximum:g}")
        return number

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ParameterError(f"{self.name}: '{value}' bukan nilai boolean (ya/tidak)")

    def _coerce_number(self, value: Any) -> int | float:
        if isinstance(value, bool):
            raise ParameterError(f"{self.name}: harus berupa angka, bukan boolean")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ParameterError(f"{self.name}: '{value}' bukan angka") from None
        if not math.isfinite(number):
            raise ParameterError(f"{self.name}: nilai harus berhingga, bukan {value}")
        if self.kind == "int":
            if number != int(number):
                raise ParameterError(f"{self.name}: harus bilangan bulat, bukan {value}")
            return int(number)
        return number

    def to_dict(self) -> dict[str, Any]:
        """Representasi JSON-aman (dipakai demo web untuk membuat kontrol)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "label": self.label,
            "help": self.help,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "choices": list(self.choices),
            "when": list(self.when) if self.when else None,
        }


@dataclass(frozen=True)
class EnhanceResult:
    """Hasil satu operasi.

    ``image`` bisa grayscale (2 dimensi) untuk operasi yang memang bekerja pada grayscale.
    ``lut`` adalah kurva transformasi ``s = T(r)`` untuk operasi titik berbasis LUT (``None`` untuk
    filter aras lokal, atau bila tiap kanal warna memakai LUT berbeda).
    """

    image: Image
    lut: Lut | None = None
    note: str = ""


Handler = Callable[[Image, dict[str, Any]], EnhanceResult]


@dataclass(frozen=True)
class Operation:
    """Deskripsi satu operasi peningkatan citra."""

    name: str
    title: str
    group: str
    summary: str
    formula: str
    ref: str
    params: tuple[Param, ...]
    handler: Handler

    @property
    def primary(self) -> Param | None:
        """Parameter numerik pertama (diatur dengan ``+`` / ``-`` pada webcam)."""
        return next((p for p in self.params if p.kind in ("int", "float")), None)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def resolve(self, given: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Gabungkan ``given`` dengan nilai bawaan; validasi nama, tipe, dan rentang."""
        given = dict(given or {})
        known = {p.name: p for p in self.params}
        unknown = sorted(set(given) - set(known))
        if unknown:
            valid = ", ".join(known) or "(tidak ada parameter)"
            raise ParameterError(
                f"Operasi '{self.name}' tidak punya parameter {', '.join(unknown)}. "
                f"Parameter yang tersedia: {valid}"
            )
        return {name: param.coerce(given.get(name, param.default)) for name, param in known.items()}

    def apply(self, image: Image, params: Mapping[str, Any] | None = None) -> EnhanceResult:
        """Validasi ``params`` lalu terapkan operasi pada ``image``."""
        return self.handler(image, self.resolve(params))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "group": self.group,
            "summary": self.summary,
            "formula": self.formula,
            "ref": self.ref,
            "params": [p.to_dict() for p in self.params],
        }


# --------------------------------------------------------------------------- #
# Handler: operasi titik
# --------------------------------------------------------------------------- #
def _grayscale(image: Image, p: dict[str, Any]) -> EnhanceResult:
    return EnhanceResult(to_grayscale(image, p["method"]))


def _negative(image: Image, p: dict[str, Any]) -> EnhanceResult:
    lut = negative_lut()
    return EnhanceResult(apply_lut(image, lut), lut)


def _threshold(image: Image, p: dict[str, Any]) -> EnhanceResult:
    gray = to_grayscale(image)
    note = ""
    value = p["threshold"]
    if p["otsu"]:
        # cv2 memberi piksel > t nilai 255; materi memakai r >= T, jadi T = t + 1.
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        value = min(MAX_LEVEL, int(otsu) + 1)
        note = f"T otomatis (Otsu) = {value}"
    lut = threshold_lut(value)
    return EnhanceResult(apply_lut(gray, lut), lut, note)


def _mbit(image: Image, p: dict[str, Any]) -> EnhanceResult:
    lut = quantize_lut(p["bits"], rescale=p["rescale"])
    return EnhanceResult(
        apply_lut(to_grayscale(image), lut), lut, f"{2 ** p['bits']} tingkat keabuan"
    )


def _brightness(image: Image, p: dict[str, Any]) -> EnhanceResult:
    lut = brightness_lut(p["value"])
    return EnhanceResult(apply_lut(image, lut), lut)


def _contrast(image: Image, p: dict[str, Any]) -> EnhanceResult:
    lut = contrast_lut(p["gain"], p["center"])
    return EnhanceResult(apply_lut(image, lut), lut)


def _gamma(image: Image, p: dict[str, Any]) -> EnhanceResult:
    lut = gamma_lut(p["gamma"])
    return EnhanceResult(apply_lut(image, lut), lut)


def _log(image: Image, p: dict[str, Any]) -> EnhanceResult:
    lut = log_lut()
    return EnhanceResult(apply_lut(image, lut), lut)


def _auto_stretch_lut(gray: Image, mode: str, percent: float) -> tuple[Lut | None, str]:
    """LUT peregangan Min-Max atau persentil dari histogram ``gray``; ``None`` bila gambar datar."""
    if mode == "minmax":
        low, high = int(gray.min()), int(gray.max())
    else:  # percentile
        lo_f, hi_f = np.percentile(gray, [percent, 100.0 - percent])
        low, high = math.floor(lo_f), math.ceil(hi_f)
    if low >= high:
        return None, "gambar datar: tidak ada yang diregangkan"
    return stretch_lut(low, 0, high, MAX_LEVEL), f"r1={low}, r2={high}"


def _stretch(image: Image, p: dict[str, Any]) -> EnhanceResult:
    if p["mode"] == "custom":
        custom = stretch_lut(p["r1"], p["s1"], p["r2"], p["s2"])
        shifted = apply_lut(image, custom, luminance_only=p["color_mode"] == "luminance")
        return EnhanceResult(shifted, custom)

    if is_gray(image) or p["color_mode"] == "luminance":
        auto, note = _auto_stretch_lut(luminance(image), p["mode"], p["percent"])
        if auto is None:
            return EnhanceResult(image.copy(), None, note)
        return EnhanceResult(apply_lut(image, auto, luminance_only=True), auto, note)

    channels = []
    for i in range(image.shape[2]):
        plane = np.ascontiguousarray(image[:, :, i])
        per_channel, _ = _auto_stretch_lut(plane, p["mode"], p["percent"])
        channels.append(plane if per_channel is None else apply_lut(plane, per_channel))
    return EnhanceResult(np.dstack(channels), None, "tiap kanal B, G, R diregangkan sendiri")


def _equalize(image: Image, p: dict[str, Any]) -> EnhanceResult:
    if is_gray(image) or p["color_mode"] == "luminance":
        lut = equalize_lut(histogram(luminance(image)))
        return EnhanceResult(apply_lut(image, lut, luminance_only=True), lut)
    return EnhanceResult(
        equalize(image, luminance_only=False), None, "tiap kanal B, G, R diratakan"
    )


def _slice(image: Image, p: dict[str, Any]) -> EnhanceResult:
    lut = slice_lut(p["low"], p["high"], p["level"], preserve=p["preserve"])
    return EnhanceResult(apply_lut(to_grayscale(image), lut), lut)


def _bitplane(image: Image, p: dict[str, Any]) -> EnhanceResult:
    lut = bitplane_lut(p["bit"])
    return EnhanceResult(apply_lut(to_grayscale(image), lut), lut)


# --------------------------------------------------------------------------- #
# Handler: aras lokal
# --------------------------------------------------------------------------- #
def _smooth(image: Image, p: dict[str, Any]) -> EnhanceResult:
    return EnhanceResult(smooth(image, p["ksize"], p["method"]))


def _median(image: Image, p: dict[str, Any]) -> EnhanceResult:
    return EnhanceResult(median_filter(image, p["ksize"]))


def _sharpen(image: Image, p: dict[str, Any]) -> EnhanceResult:
    return EnhanceResult(sharpen(image, p["amount"], p["ksize"]))


# --------------------------------------------------------------------------- #
# Daftar operasi
# --------------------------------------------------------------------------- #
_COLOR_MODE = Param(
    "color_mode",
    "choice",
    "luminance",
    "Mode warna",
    "luminance: hanya kecerahan (Y) yang diubah sehingga warna terjaga; "
    "channels: tiap kanal B, G, R diproses sendiri (warna bisa bergeser)",
    choices=COLOR_MODES,
)

_KSIZE = Param("ksize", "int", 5, "Ukuran kernel", "sisi kernel (bilangan ganjil >= 3)", 3, 31, 2)

_OPERATIONS: tuple[Operation, ...] = (
    # ----- Konversi citra -------------------------------------------------- #
    Operation(
        "grayscale",
        "RGB ke Grayscale",
        "Konversi citra",
        "Ubah citra berwarna menjadi derajat keabuan 0-255.",
        "x = a_r*r + a_g*g + a_b*b,  a_r + a_g + a_b = 1",
        "P03 hlm. 10",
        (
            Param(
                "method",
                "choice",
                "luminance",
                "Metode",
                "average: (r+g+b)/3 seperti pada materi; luminance: bobot 0.299/0.587/0.114 "
                "yang mengikuti kepekaan mata",
                choices=tuple(GRAY_WEIGHTS),
            ),
        ),
        _grayscale,
    ),
    Operation(
        "negative",
        "Negasi Citra",
        "Konversi citra",
        "Balik terang-gelap sehingga hasilnya seperti klise foto.",
        "s = 255 - r",
        "P03 hlm. 22",
        (),
        _negative,
    ),
    Operation(
        "threshold",
        "Thresholding (Grayscale ke Biner)",
        "Konversi citra",
        "Ubah citra menjadi hitam-putih: r >= T menjadi putih (255), selain itu hitam (0).",
        "s = 255 jika r >= T, selain itu 0",
        "P03 hlm. 11 dan 27-29",
        (
            Param("threshold", "int", 128, "Ambang (T)", "nilai ambang 0-255", 0, 255, 1),
            Param(
                "otsu",
                "bool",
                False,
                "Ambang otomatis (Otsu)",
                "pilih T otomatis dari histogram; mengabaikan nilai T di atas",
            ),
        ),
        _threshold,
    ),
    Operation(
        "mbit",
        "Grayscale ke m-bit",
        "Konversi citra",
        "Kurangi jumlah tingkat keabuan menjadi 2^m (kuantisasi).",
        "s = langkah * floor(r / langkah),  langkah = 256 / 2^m",
        "P03 hlm. 12",
        (
            Param("bits", "int", 4, "Jumlah bit (m)", "1 bit = 2 tingkat ... 8 bit = 256", 1, 8, 1),
            Param(
                "rescale",
                "bool",
                False,
                "Sebar ke 0-255",
                "sebar tingkat keabuan merata sampai 255 agar putih tetap putih",
            ),
        ),
        _mbit,
    ),
    # ----- Kecerahan dan kontras ------------------------------------------- #
    Operation(
        "brightness",
        "Pengaturan Brightness",
        "Kecerahan & kontras",
        "Tambahkan nilai tetap pada setiap piksel (positif = lebih terang).",
        "s = r + t",
        "P03 hlm. 13",
        (Param("value", "int", 50, "Perubahan (t)", "-255 sampai 255", -255, 255, 5),),
        _brightness,
    ),
    Operation(
        "contrast",
        "Pengaturan Kontras",
        "Kecerahan & kontras",
        "Perbesar / perkecil selisih tiap piksel terhadap pusat pengontrasan.",
        "s = G * (r - P) + P",
        "P03 hlm. 25",
        (
            Param(
                "gain",
                "float",
                1.5,
                "Penguatan (G)",
                "G > 1 menaikkan, G < 1 menurunkan",
                0,
                5,
                0.05,
            ),
            Param("center", "int", 128, "Pusat (P)", "intensitas yang tidak berubah", 0, 255, 1),
        ),
        _contrast,
    ),
    Operation(
        "stretch",
        "Contrast Stretching",
        "Kecerahan & kontras",
        "Regangkan intensitas agar mengisi seluruh rentang 0-255 (fungsi linear per bagian).",
        "garis (0,0)-(r1,s1)-(r2,s2)-(255,255)",
        "P03 hlm. 23-26",
        (
            Param(
                "mode",
                "choice",
                "percentile",
                "Mode",
                "minmax: rmin-rmax; percentile: buang persentase piksel terekstrem; "
                "custom: titik (r1,s1), (r2,s2) sendiri",
                choices=("minmax", "percentile", "custom"),
            ),
            Param(
                "percent",
                "float",
                1.0,
                "Persentil (%)",
                "persentase piksel yang dibuang di tiap ujung histogram",
                0,
                49,
                0.5,
                when=("mode", "percentile"),
            ),
            Param(
                "r1",
                "int",
                70,
                "r1",
                "titik belok pertama (masukan)",
                0,
                255,
                1,
                when=("mode", "custom"),
            ),
            Param(
                "s1",
                "int",
                30,
                "s1",
                "titik belok pertama (keluaran)",
                0,
                255,
                1,
                when=("mode", "custom"),
            ),
            Param(
                "r2",
                "int",
                180,
                "r2",
                "titik belok kedua (masukan)",
                0,
                255,
                1,
                when=("mode", "custom"),
            ),
            Param(
                "s2",
                "int",
                225,
                "s2",
                "titik belok kedua (keluaran)",
                0,
                255,
                1,
                when=("mode", "custom"),
            ),
            _COLOR_MODE,
        ),
        _stretch,
    ),
    Operation(
        "gamma",
        "Koreksi Gamma (Power-Law)",
        "Kecerahan & kontras",
        "Cerahkan (gamma < 1) atau gelapkan (gamma > 1) tanpa membuat putih / hitam terpotong.",
        "s = 255 * (r / 255) ^ gamma",
        "P03 hlm. 9",
        (Param("gamma", "float", 0.5, "Gamma", "< 1 mencerahkan, > 1 menggelapkan", 0.1, 5, 0.05),),
        _gamma,
    ),
    Operation(
        "log",
        "Transformasi Logaritma",
        "Kecerahan & kontras",
        "Pekarkan area gelap dan mampatkan area terang.",
        "s = c * log(1 + r),  c = 255 / log(256)",
        "P03 hlm. 9",
        (),
        _log,
    ),
    # ----- Histogram ------------------------------------------------------- #
    Operation(
        "equalize",
        "Histogram Equalization",
        "Histogram",
        "Ratakan distribusi keabuan memakai distribusi kumulatif agar citra lebih jelas.",
        "s_k = round( (L-1) * sum_{j<=k} n_j / n )",
        "P03 hlm. 14-21",
        (_COLOR_MODE,),
        _equalize,
    ),
    # ----- Slicing --------------------------------------------------------- #
    Operation(
        "slice",
        "Gray-Level Slicing",
        "Slicing",
        "Sorot rentang intensitas tertentu [A, B].",
        "s = level jika A <= r <= B, selain itu 0 (atau r bila 'pertahankan latar')",
        "P03 hlm. 9",
        (
            Param("low", "int", 100, "Batas bawah (A)", "awal rentang yang disorot", 0, 255, 1),
            Param("high", "int", 180, "Batas atas (B)", "akhir rentang yang disorot", 0, 255, 1),
            Param(
                "level", "int", 255, "Level sorotan", "nilai untuk piksel dalam rentang", 0, 255, 1
            ),
            Param(
                "preserve",
                "bool",
                False,
                "Pertahankan latar",
                "piksel di luar rentang tidak diubah (bukan dihitamkan)",
            ),
        ),
        _slice,
    ),
    Operation(
        "bitplane",
        "Bit-Plane Slicing",
        "Slicing",
        "Tampilkan satu bidang bit (0 = paling tidak berarti ... 7 = paling berarti).",
        "s = 255 jika bit ke-k dari r bernilai 1, selain itu 0",
        "P03 hlm. 9",
        (Param("bit", "int", 7, "Bidang bit (k)", "0 sampai 7", 0, 7, 1),),
        _bitplane,
    ),
    # ----- Aras lokal (lanjutan) ------------------------------------------- #
    Operation(
        "smooth",
        "Smoothing (Penghalusan)",
        "Aras lokal (lanjutan)",
        "Kurangi noise dengan merata-ratakan piksel dan tetangganya.",
        "s(x,y) = rata-rata / Gaussian dari jendela k x k",
        "P03 hlm. 7 (Mask Operation)",
        (
            Param(
                "method",
                "choice",
                "gaussian",
                "Metode",
                "mean: rata-rata biasa; gaussian: tetangga dekat berbobot lebih besar",
                choices=SMOOTH_METHODS,
            ),
            _KSIZE,
        ),
        _smooth,
    ),
    Operation(
        "median",
        "Median Filtering",
        "Aras lokal (lanjutan)",
        "Ganti tiap piksel dengan median tetangganya; efektif untuk noise garam-lada.",
        "s(x,y) = median dari jendela k x k",
        "P03 hlm. 7 (Mask Operation)",
        (
            Param(
                "ksize", "int", 5, "Ukuran kernel", "sisi kernel (bilangan ganjil >= 3)", 3, 31, 2
            ),
        ),
        _median,
    ),
    Operation(
        "sharpen",
        "Sharpening (Penajaman)",
        "Aras lokal (lanjutan)",
        "Pertegas tepi dengan unsharp mask untuk mengurangi blur.",
        "s = r + amount * (r - blur(r))",
        "P03 hlm. 7 (Mask Operation)",
        (
            Param("amount", "float", 1.0, "Besar penajaman", "0 = tidak berubah", 0, 5, 0.1),
            _KSIZE,
        ),
        _sharpen,
    ),
)

OPERATIONS: dict[str, Operation] = {op.name: op for op in _OPERATIONS}


def get_operation(name: str) -> Operation:
    """Cari operasi berdasarkan nama; ``OperationError`` bila tidak ada."""
    try:
        return OPERATIONS[name]
    except KeyError:
        raise OperationError(
            f"Operasi '{name}' tidak dikenal. Pilihan: {', '.join(OPERATIONS)}"
        ) from None


def list_operations() -> list[Operation]:
    """Semua operasi, berurutan menurut kelompoknya."""
    return list(OPERATIONS.values())


def apply_operation(
    image: Image, name: str, params: Mapping[str, Any] | None = None
) -> EnhanceResult:
    """Terapkan operasi ``name`` pada ``image``; pintasan untuk ``get_operation(name).apply``."""
    return get_operation(name).apply(image, params)
