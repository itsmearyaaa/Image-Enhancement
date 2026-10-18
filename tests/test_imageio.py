from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imgenhance import (
    ImageLoadError,
    ImageWriteError,
    decode_image,
    encode_image,
    read_image,
    write_image,
)
from imgenhance.imageio import collapse_gray, list_images, resize_max_side


def test_write_then_read_roundtrip_with_unicode_path(
    tmp_path: Path, color_image: np.ndarray
) -> None:
    target = write_image(tmp_path / "folder uji" / "fotö ünïcode.png", color_image)
    assert target.is_file()
    np.testing.assert_array_equal(read_image(target), color_image)


def test_grayscale_file_is_read_as_2d(tmp_path: Path, gray_image: np.ndarray) -> None:
    path = write_image(tmp_path / "gray.png", gray_image)
    loaded = read_image(path)
    assert loaded.ndim == 2
    np.testing.assert_array_equal(loaded, gray_image)


def test_color_file_stays_3_channels(tmp_path: Path, color_image: np.ndarray) -> None:
    assert read_image(write_image(tmp_path / "c.png", color_image)).shape == color_image.shape


def test_collapse_gray_only_when_channels_are_identical(color_image: np.ndarray) -> None:
    gray = np.dstack([color_image[:, :, 0]] * 3).copy()
    assert collapse_gray(gray).ndim == 2
    assert collapse_gray(color_image).ndim == 3
    two_d = color_image[:, :, 0]
    assert collapse_gray(two_d) is two_d


def test_decode_and_encode_roundtrip(color_image: np.ndarray) -> None:
    data = encode_image(color_image, ".png")
    np.testing.assert_array_equal(decode_image(data), color_image)
    assert encode_image(color_image, ".jpg")[:2] == b"\xff\xd8"  # tanda JPEG


def test_decode_rejects_empty_and_garbage() -> None:
    with pytest.raises(ImageLoadError, match="kosong"):
        decode_image(b"")
    with pytest.raises(ImageLoadError, match="valid"):
        decode_image(b"ini bukan gambar", "x.png")


def test_encode_rejects_unknown_extension(color_image: np.ndarray) -> None:
    with pytest.raises(ImageWriteError, match="tidak didukung"):
        encode_image(color_image, ".tidakada")


def test_read_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError, match="tidak ditemukan"):
        read_image(tmp_path / "hilang.png")


def test_write_failure_is_wrapped(tmp_path: Path, color_image: np.ndarray) -> None:
    blocker = tmp_path / "berkas"
    blocker.write_text("bukan folder")
    with pytest.raises(ImageWriteError):
        write_image(blocker / "x.png", color_image)


def test_list_images_filters_sorts_and_recurses(tmp_path: Path, color_image: np.ndarray) -> None:
    write_image(tmp_path / "b.png", color_image)
    write_image(tmp_path / "a.JPG", color_image)
    (tmp_path / "catatan.txt").write_text("abaikan")
    write_image(tmp_path / "sub" / "c.png", color_image)
    assert [p.name for p in list_images(tmp_path)] == ["a.JPG", "b.png"]
    assert [p.name for p in list_images(tmp_path, recursive=True)] == ["a.JPG", "b.png", "c.png"]
    with pytest.raises(ImageLoadError):
        list_images(tmp_path / "tidak-ada")


def test_resize_max_side() -> None:
    big = np.zeros((200, 400, 3), dtype=np.uint8)
    assert resize_max_side(big, 100).shape[:2] == (50, 100)
    assert resize_max_side(big, 0) is big  # 0 = tanpa perkecilan
    assert resize_max_side(big, 1000) is big  # tidak pernah memperbesar
    assert resize_max_side(np.zeros((300, 10), dtype=np.uint8), 30).shape == (30, 1)
