.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install install-dev samples demo test cov lint format typecheck check build clean

help:  ## Tampilkan daftar perintah
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Pasang paket (mode editable)
	$(PYTHON) -m pip install -e .

install-dev:  ## Pasang paket + semua dependensi pengembangan
	$(PYTHON) -m pip install -e ".[dev]"

samples:  ## Buat ulang gambar contoh (butuh scikit-image)
	$(PYTHON) examples/generate_samples.py

demo:  ## Terapkan semua operasi pada gambar contoh (hasil di folder output/)
	$(PYTHON) -m imgenhance all examples/images/sample_astronaut_gelap.png -o output --no-show
	$(PYTHON) -m imgenhance batch examples/images --auto -o output

test:  ## Jalankan uji unit
	$(PYTHON) -m pytest

cov:  ## Jalankan uji dengan laporan cakupan
	$(PYTHON) -m pytest --cov --cov-report=term-missing

lint:  ## Cek gaya kode (ruff)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Rapikan kode otomatis
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

typecheck:  ## Cek tipe statis (mypy strict)
	$(PYTHON) -m mypy

check: lint typecheck test  ## Jalankan semua pemeriksaan kualitas

build:  ## Bangun paket sdist + wheel ke dist/
	$(PYTHON) -m pip install build
	$(PYTHON) -m build

clean:  ## Hapus artefak build / cache
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
