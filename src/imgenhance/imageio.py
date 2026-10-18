"""Baca / tulis / enkode gambar. Aman untuk path berisi spasi dan karakter non-ASCII."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from ._types import Image
from .errors import ImageLoadError, ImageWriteError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def collapse_gray(image: Image) -> Image:
    """Ubah gambar 3 kanal yang kanal B, G, R-nya identik menjadi grayscale 2 dimensi.

    File grayscale dibaca OpenCV sebagai 3 kanal sama; dengan ini histogram dan operasinya
    diperlakukan sebagai grayscale sungguhan. Gambar berwarna dikembalikan apa adanya.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if np.array_equal(blue, green) and np.array_equal(green, red):
            return cast(Image, np.ascontiguousarray(blue))
    return image


def decode_image(data: bytes, source: str = "<bytes>") -> Image:
    """Dekode byte gambar (JPEG/PNG/...) menjadi array BGR (grayscale bila semua kanalnya sama)."""
    if not data:
        raise ImageLoadError(f"Data gambar kosong: {source}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageLoadError(f"Bukan file gambar yang valid: {source}")
    return collapse_gray(cast(Image, image))


def read_image(path: str | Path) -> Image:
    """Baca file gambar sebagai BGR (atau grayscale 2 dimensi bila semua kanalnya identik).

    ``cv2.imread`` gagal pada path non-ASCII di Windows, sehingga file dibaca lewat NumPy lalu
    didekode dengan ``cv2.imdecode``.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"File tidak ditemukan: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageLoadError(f"File tidak bisa dibaca ({path}): {exc}") from exc
    logger.debug("Membaca %s (%d byte)", path, len(data))
    return decode_image(data, str(path))


def encode_image(image: Image, extension: str = ".png") -> bytes:
    """Enkode gambar (BGR atau grayscale) ke byte (default PNG)."""
    try:
        ok, buffer = cv2.imencode(extension, image)
    except cv2.error as exc:  # ekstensi tidak dikenal OpenCV
        raise ImageWriteError(f"Format {extension} tidak didukung untuk menyimpan gambar") from exc
    if not ok:
        raise ImageWriteError(f"Gagal mengenkode gambar ke format {extension}")
    return bytes(buffer.tobytes())


def write_image(path: str | Path, image: Image) -> Path:
    """Tulis gambar ke disk; folder tujuan dibuat bila belum ada."""
    path = Path(path)
    data = encode_image(image, path.suffix or ".png")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as exc:
        raise ImageWriteError(f"Gagal menulis {path}: {exc}") from exc
    logger.debug("Menulis %s (%d byte)", path, len(data))
    return path


def list_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    """Daftar file gambar (terurut) di dalam ``folder``."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ImageLoadError(f"Folder tidak ditemukan: {folder}")
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def resize_max_side(image: Image, max_side: int) -> Image:
    """Perkecil gambar bila sisi terpanjangnya melebihi ``max_side`` (tidak pernah memperbesar)."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cast(Image, cv2.resize(image, size, interpolation=cv2.INTER_AREA))
