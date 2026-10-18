"""Contoh pemakaian imgenhance sebagai library.

python examples/usage.py [path/gambar.jpg]
"""

from __future__ import annotations

import sys
from pathlib import Path

from imgenhance import (
    EnhanceError,
    Pipeline,
    apply_operation,
    diagnose_image,
    read_image,
    write_image,
)
from imgenhance.render import comparison

DEFAULT_IMAGE = Path(__file__).resolve().parent / "images" / "sample_astronaut_gelap.png"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    try:
        image = read_image(path)

        # 1. Baca histogram: gelap / terang / kontras rendah, beserta sarannya
        diagnosis = diagnose_image(image)
        print(diagnosis.summary)

        # 2. Satu operasi (parameter yang tidak diisi memakai nilai bawaan)
        result = apply_operation(image, "gamma", {"gamma": 0.5})
        print(
            "Gamma 0.5 -> ukuran",
            result.image.shape,
            "| kurva T(r) tersedia:",
            result.lut is not None,
        )

        # 3. Rangkaian operasi
        pipeline = Pipeline.parse(["gamma:gamma=0.6", "stretch:mode=percentile,percent=1"])
        final = pipeline.run(image)
        print("Pipeline:", pipeline.label())

        # 4. Simpan hasil dan gambar perbandingan (asli, hasil, statistik, histogram, kurva)
        out = Path("output")
        write_image(out / f"{path.stem}_contoh.png", final.image)
        panel = comparison(image, final.image, "Contoh pipeline", subtitle=pipeline.label())
        write_image(out / f"{path.stem}_contoh_compare.png", panel)
        print("Disimpan di folder:", out.resolve())
    except EnhanceError as exc:
        print(f"Galat: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
