# Changelog

Format mengikuti [Keep a Changelog](https://keepachangelog.com/id-ID/1.1.0/) dan
[Semantic Versioning](https://semver.org/lang/id/).

## [1.0.0] - 2026-10-13

### Ditambahkan
- Library `imgenhance` (src layout, bertipe penuh / `py.typed`).
- Operasi titik berbasis look-up table: negasi, RGB ke grayscale, grayscale ke biner / thresholding
  (termasuk Otsu), grayscale ke m-bit, brightness, kontras, contrast stretching (min-max, persentil,
  kustom), koreksi gamma, transformasi log, gray-level slicing, dan bit-plane slicing.
- Histogram, distribusi kumulatif, dan histogram equalization sesuai rumus materi P03; gambar
  berwarna diproses pada kanal luminans agar warna tidak bergeser.
- Operasi aras lokal sebagai pelengkap: smoothing (mean / Gaussian), median filter, sharpening.
- Daftar operasi berbasis data (`Operation`, `Param`) dengan validasi tipe dan rentang; CLI, demo web,
  dan webcam dibangun dari daftar yang sama.
- `Pipeline` untuk merangkai operasi, serta diagnosis histogram (gelap / terang / kontras rendah)
  yang menyarankan perbaikan (`auto`).
- CLI `imgenhance` dengan sub-perintah `list`, `apply`, `pipeline`, `auto`, `analyze`, `all`, `batch`,
  `webcam`, `serve`; keluaran teks atau `--json`, kode keluar yang jelas, dan logging (`-v` / `-q`).
- Gambar perbandingan (asli, hasil, statistik, histogram, kurva transformasi) dan lembar kontak.
- Demo web FastAPI opsional (`pip install "imgenhance[web]"`).
- Uji otomatis (pytest) termasuk soal contoh pada materi, lint (ruff), type check (mypy strict), dan
  CI GitHub Actions.
