"""Hierarki exception khusus aplikasi.

Semua galat yang *diharapkan* (input salah, file rusak, parameter di luar rentang, kamera tidak
ada, dll.) diturunkan dari :class:`EnhanceError` sehingga lapisan atas (CLI / web) cukup menangkap
satu tipe dasar.
"""

from __future__ import annotations


class EnhanceError(Exception):
    """Kelas dasar semua galat aplikasi."""


class ImageLoadError(EnhanceError):
    """File gambar tidak ditemukan, tidak bisa dibaca, atau formatnya tidak valid."""


class ImageWriteError(EnhanceError):
    """Gambar gagal ditulis ke disk."""


class OperationError(EnhanceError):
    """Nama operasi peningkatan citra tidak dikenal."""


class ParameterError(EnhanceError, ValueError):
    """Parameter operasi tidak valid (tipe salah, di luar rentang, atau tidak dikenal).

    Turunan ``ValueError`` juga, sehingga fungsi tingkat rendah (mis. ``brightness_lut``) tetap
    berperilaku seperti fungsi Python biasa bila dipakai langsung sebagai library.
    """


class CameraError(EnhanceError):
    """Kamera tidak dapat dibuka atau gagal membaca frame."""
