"""Antarmuka command line: ``imgenhance list | apply | pipeline | auto | analyze | all | batch |
webcam | serve``."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from . import __version__
from ._types import Image
from .analysis import ImageStats, auto_pipeline, diagnose
from .camera import run_webcam
from .errors import (
    CameraError,
    EnhanceError,
    ImageLoadError,
    OperationError,
    ParameterError,
)
from .imageio import list_images, read_image, resize_max_side, write_image
from .operations import Operation, Param, get_operation, list_operations
from .pipeline import Pipeline, Step
from .render import comparison, contact_sheet, histogram_tile
from .report import (
    csv_row,
    format_diagnosis,
    format_text,
    to_json,
    write_csv,
)

logger = logging.getLogger("imgenhance")

DEFAULT_OUTPUT = Path("output")
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


# --------------------------------------------------------------------------- #
# Utilitas
# --------------------------------------------------------------------------- #
def _has_display() -> bool:
    """False pada Linux tanpa display (server/headless); Windows & macOS dianggap punya layar."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _show(title: str, image: Image, requested: bool | None) -> None:
    """Tampilkan jendela OpenCV bila diminta/mungkin. ``requested=None`` berarti otomatis."""
    wanted = _has_display() if requested is None else requested
    if not wanted:
        return
    if not _has_display():
        logger.warning("Tidak ada layar/display; jendela tidak ditampilkan.")
        return
    try:
        cv2.imshow(title, image)
        logger.info("Tekan tombol apa saja pada jendela gambar untuk menutup.")
        cv2.waitKey(0)
    except cv2.error as exc:  # OpenCV tanpa dukungan GUI (build headless)
        logger.warning("Jendela gambar tidak bisa dibuka: %s", str(exc).splitlines()[0])
        return
    finally:
        with contextlib.suppress(cv2.error):
            cv2.destroyAllWindows()


def _silence_stdout() -> None:
    """Cegah pesan galat tambahan saat stdout ditutup lebih dulu oleh pipa (mis. ``| head``)."""
    # stdout tanpa file descriptor (mis. diganti saat pengujian) cukup diabaikan
    with contextlib.suppress(OSError, ValueError):
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("tidak boleh negatif")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("harus bilangan bulat positif")
    return value


def _setup_logging(verbose: int, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        level=level, format="%(levelname)-7s %(message)s", stream=sys.stderr, force=True
    )


def _load(path: str | Path, max_side: int) -> Image:
    return resize_max_side(read_image(path), max_side)


def _tag(pipeline: Pipeline) -> str:
    """Bagian nama file keluaran: nama operasi, atau gabungan nama langkah."""
    name = "_".join(step.name for step in pipeline.steps)
    return name if len(name) <= 40 else f"pipeline{len(pipeline.steps)}"


# --------------------------------------------------------------------------- #
# Inti pemrosesan satu gambar
# --------------------------------------------------------------------------- #
@dataclass
class Outcome:
    """Hasil memproses satu gambar lewat satu pipeline."""

    name: str
    steps: str
    before: ImageStats
    after: ImageStats
    notes: list[str]
    output: Path
    compare: Path | None
    image: Image
    panel: Image | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.name,
            "steps": self.steps,
            "notes": self.notes,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "output": str(self.output),
            "compare": str(self.compare) if self.compare else None,
        }


def _process(
    image: Image,
    name: str,
    stem: str,
    pipeline: Pipeline,
    output_dir: Path,
    compare: bool,
) -> Outcome:
    """Jalankan ``pipeline`` pada ``image``; simpan hasil (dan gambar perbandingan bila diminta)."""
    result = pipeline.run(image)
    before, after = ImageStats.from_image(image), ImageStats.from_image(result.image)
    tag = _tag(pipeline)
    output = write_image(output_dir / f"{stem}_{tag}.png", result.image)

    panel: Image | None = None
    compare_path: Path | None = None
    if compare:
        single = len(pipeline.steps) == 1
        title = get_operation(pipeline.steps[0].name).title if single else "Pipeline"
        panel = comparison(
            image,
            result.image,
            title,
            lut=result.lut,
            subtitle=f"{name}  |  {pipeline.label()}",
            note="; ".join(result.notes),
        )
        compare_path = write_image(output_dir / f"{stem}_{tag}_compare.png", panel)
    return Outcome(
        name,
        pipeline.label(),
        before,
        after,
        result.notes,
        output,
        compare_path,
        result.image,
        panel,
    )


def _report(outcome: Outcome, as_json: bool) -> None:
    if as_json:
        print(to_json(outcome.to_dict()))
    else:
        print(
            format_text(outcome.name, outcome.steps, outcome.before, outcome.after, outcome.notes)
        )
    logger.info("Hasil disimpan: %s", outcome.output)
    if outcome.compare:
        logger.info("Perbandingan : %s", outcome.compare)


def _run_single(args: argparse.Namespace, pipeline: Pipeline) -> int:
    path = Path(args.path)
    image = _load(path, args.max_side)
    outcome = _process(
        image, path.name, path.stem, pipeline, Path(args.output), not args.no_compare
    )
    _report(outcome, args.json)
    _show(
        "Image Enhancement",
        outcome.panel if outcome.panel is not None else outcome.image,
        args.show,
    )
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Sub-perintah
# --------------------------------------------------------------------------- #
def cmd_list(args: argparse.Namespace) -> int:
    operations = list_operations()
    if args.json:
        print(to_json([op.to_dict() for op in operations]))
        return EXIT_OK

    print("Operasi peningkatan citra. Pakai: imgenhance apply <operasi> GAMBAR [opsi]\n")
    group = ""
    for op in operations:
        if op.group != group:
            group = op.group
            print(f"{group}")
        print(f"  {op.name:<11} {op.title}  [{op.ref}]")
        print(f"  {'':<11} rumus: {op.formula}")
        for param in op.params:
            print(f"  {'':<11} --{param.name.replace('_', '-')}: {_describe_param(param)}")
        print()
    return EXIT_OK


def cmd_apply(args: argparse.Namespace) -> int:
    operation = get_operation(args.operation)
    given = {
        p.name: value
        for p in operation.params
        if (value := getattr(args, f"param_{p.name}", None)) is not None
    }
    return _run_single(args, Pipeline([Step(operation.name, given)]))


def cmd_pipeline(args: argparse.Namespace) -> int:
    return _run_single(args, Pipeline.parse(args.steps))


def cmd_auto(args: argparse.Namespace) -> int:
    path = Path(args.path)
    image = _load(path, args.max_side)
    pipeline = auto_pipeline(image)
    if pipeline is None:
        diagnosis = diagnose(ImageStats.from_image(image))
        print(f"=== {path.name} ===\n{diagnosis.summary}")
        logger.info("Tidak ada file yang dibuat karena gambar sudah baik.")
        return EXIT_OK
    return _run_single(args, pipeline)


def cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.path)
    image = _load(path, args.max_side)
    stats = ImageStats.from_image(image)
    diagnosis = diagnose(stats)
    figure = histogram_tile(image, (720, 360), f"Histogram - {path.name}")
    figure_path = write_image(Path(args.output) / f"{path.stem}_histogram.png", figure)

    if args.json:
        print(to_json({"file": path.name, **stats.to_dict(), "diagnosis": diagnosis.to_dict()}))
    else:
        print(format_diagnosis(path.name, stats, diagnosis))
    logger.info("Histogram disimpan: %s", figure_path)
    _show("Histogram", figure, args.show)
    return EXIT_OK


def cmd_all(args: argparse.Namespace) -> int:
    path = Path(args.path)
    image = _load(path, args.max_side)
    output = Path(args.output)
    items: list[tuple[str, Image]] = [("Asli", image)]
    for operation in list_operations():
        result = operation.apply(image)
        write_image(output / f"{path.stem}_{operation.name}.png", result.image)
        items.append((operation.title, result.image))
        logger.debug("%s selesai", operation.name)

    sheet = contact_sheet(items, columns=4, title=f"Semua operasi (parameter bawaan) - {path.name}")
    sheet_path = write_image(output / f"{path.stem}_semua.png", sheet)
    logger.info("%d operasi diterapkan pada %s", len(items) - 1, path.name)
    logger.info("Lembar kontak: %s", sheet_path)
    if args.json:
        print(
            to_json(
                {
                    "file": path.name,
                    "operations": [op.name for op in list_operations()],
                    "contact_sheet": str(sheet_path),
                }
            )
        )
    _show("Semua Operasi", sheet, args.show)
    return EXIT_OK


def cmd_batch(args: argparse.Namespace) -> int:
    folder = Path(args.path)
    files = list_images(folder, recursive=args.recursive)
    if not files:
        raise ImageLoadError(f"Tidak ada file gambar di folder: {folder}")
    if args.steps:
        fixed: Pipeline | None = Pipeline.parse(args.steps)
    elif args.auto:
        fixed = None
    else:
        raise ParameterError("Beri minimal satu langkah (mis. equalize) atau gunakan --auto")

    output = Path(args.output)
    rows: list[dict[str, Any]] = []
    outcomes: list[Outcome] = []
    failures = skipped = 0
    for file in files:
        try:
            image = _load(file, args.max_side)
            pipeline = fixed if fixed is not None else auto_pipeline(image)
            if pipeline is None:
                skipped += 1
                logger.info("Lewati %s: gambar sudah baik", file.name)
                continue
            flat = "__".join(file.relative_to(folder).with_suffix("").parts)
            outcome = _process(
                image, str(file.relative_to(folder)), flat, pipeline, output, not args.no_compare
            )
        except EnhanceError as exc:
            failures += 1
            logger.warning("Lewati %s: %s", file.name, exc)
            continue
        outcomes.append(outcome)
        rows.append(
            csv_row(outcome.name, outcome.steps, outcome.before, outcome.after, outcome.output.name)
        )
        if not args.json:
            print(
                format_text(
                    outcome.name, outcome.steps, outcome.before, outcome.after, outcome.notes
                ),
                end="\n\n",
            )

    csv_path = write_csv(output / "summary.csv", rows)
    if args.json:
        print(to_json([o.to_dict() for o in outcomes]))
    logger.info(
        "%d gambar diproses, %d sudah baik, %d gagal. Ringkasan: %s",
        len(outcomes),
        skipped,
        failures,
        csv_path,
    )
    return EXIT_ERROR if failures else EXIT_OK


def cmd_webcam(args: argparse.Namespace) -> int:
    if not _has_display():
        raise CameraError("Mode webcam membutuhkan layar/display untuk menampilkan jendela.")
    run_webcam(args.operation, args.camera, Path(args.output))
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .web.app import create_app
    except ImportError as exc:
        raise EnhanceError(
            'Dependensi web belum terpasang. Jalankan: pip install "imgenhance[web]"'
        ) from exc
    logger.info("Demo web berjalan di http://%s:%d (Ctrl+C untuk berhenti)", args.host, args.port)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _describe_param(param: Param) -> str:
    """Teks bantuan satu parameter: penjelasan, rentang / pilihan, dan nilai bawaan."""
    if param.kind == "choice":
        domain = "{" + ",".join(param.choices) + "}"
    elif param.kind == "bool":
        domain = "ya/tidak"
    else:
        domain = f"{param.minimum:g}..{param.maximum:g}"
    text = f"{param.help} [{domain}, bawaan: {param.default}]"
    if param.when:
        text += f" (dipakai bila {param.when[0]}={param.when[1]})"
    return text


def _add_param_argument(parser: argparse.ArgumentParser, param: Param) -> None:
    flag = "--" + param.name.replace("_", "-")
    dest = f"param_{param.name}"
    help_text = _describe_param(param).replace("%", "%%")
    if param.kind == "bool":
        parser.add_argument(
            flag, dest=dest, action=argparse.BooleanOptionalAction, default=None, help=help_text
        )
    elif param.kind == "choice":
        parser.add_argument(flag, dest=dest, choices=param.choices, default=None, help=help_text)
    else:  # angka: konversi dan pemeriksaan rentang dilakukan Param.coerce
        parser.add_argument(flag, dest=dest, default=None, metavar="NILAI", help=help_text)


def _common_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-v", "--verbose", action="count", default=0, help="tampilkan log debug")
    parent.add_argument(
        "-q", "--quiet", action="store_true", help="hanya tampilkan peringatan dan galat"
    )
    return parent


def _output_parent(*, window: bool, compare: bool = True) -> argparse.ArgumentParser:
    """Opsi hasil yang dipakai bersama; ``window`` menambah ``--show``."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="DIR",
        help="folder hasil (default: output)",
    )
    parent.add_argument(
        "--max-side",
        type=_non_negative_int,
        default=0,
        metavar="PX",
        help="perkecil gambar ke sisi maksimum ini sebelum diproses (default: 0 = ukuran asli)",
    )
    parent.add_argument(
        "--json", action="store_true", help="keluarkan hasil sebagai JSON ke stdout"
    )
    if compare:
        parent.add_argument(
            "--no-compare",
            action="store_true",
            help="jangan buat gambar perbandingan (*_compare.png)",
        )
    if window:
        parent.add_argument(
            "--show",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="buka jendela hasil (default: otomatis bila ada layar)",
        )
    return parent


def _operation_parser(
    sub: Any, operation: Operation, parents: list[argparse.ArgumentParser]
) -> None:
    parser = sub.add_parser(
        operation.name,
        parents=parents,
        help=operation.title,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"{operation.title}\n\n{operation.summary}\n\n"
            f"Rumus   : {operation.formula}\nRujukan : {operation.ref}"
        ),
    )
    parser.add_argument("path", help="path file gambar")
    for param in operation.params:
        _add_param_argument(parser, param)
    parser.set_defaults(func=cmd_apply, operation=operation.name)


def build_parser() -> argparse.ArgumentParser:
    common = _common_parent()
    single = _output_parent(window=True)
    batch_out = _output_parent(window=False)
    parser = argparse.ArgumentParser(
        prog="imgenhance",
        description="Peningkatan kualitas citra (image enhancement) dengan Python dan OpenCV.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="PERINTAH")

    p = sub.add_parser("list", parents=[common], help="tampilkan semua operasi dan parameternya")
    p.add_argument("--json", action="store_true", help="keluarkan sebagai JSON")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("apply", help="terapkan satu operasi pada satu gambar (lihat: list)")
    op_sub = p.add_subparsers(dest="operation", required=True, metavar="OPERASI")
    for operation in list_operations():
        _operation_parser(op_sub, operation, [common, single])

    p = sub.add_parser(
        "pipeline",
        parents=[common, single],
        help="jalankan beberapa operasi berurutan pada satu gambar",
        epilog="Langkah: NAMA atau NAMA:kunci=nilai,kunci=nilai. "
        "Contoh: imgenhance pipeline foto.jpg grayscale gamma:gamma=0.6 equalize",
    )
    p.add_argument("path", help="path file gambar")
    p.add_argument("steps", nargs="+", metavar="LANGKAH", help="satu atau lebih langkah")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser(
        "auto",
        parents=[common, single],
        help="baca histogram, pilih perbaikan yang cocok, lalu terapkan",
    )
    p.add_argument("path", help="path file gambar")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser(
        "analyze",
        parents=[common, _output_parent(window=True, compare=False)],
        help="analisis histogram: gelap / terang / kontras rendah beserta saran",
    )
    p.add_argument("path", help="path file gambar")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser(
        "all",
        parents=[common, _output_parent(window=True, compare=False)],
        help="terapkan semua operasi (parameter bawaan) dan buat lembar kontak",
    )
    p.add_argument("path", help="path file gambar")
    p.set_defaults(func=cmd_all)

    p = sub.add_parser(
        "batch",
        parents=[common, batch_out],
        help="proses semua gambar dalam satu folder (+ ringkasan CSV)",
    )
    p.add_argument("path", help="path folder berisi gambar")
    p.add_argument("steps", nargs="*", metavar="LANGKAH", help="langkah pipeline (lihat: pipeline)")
    p.add_argument("--auto", action="store_true", help="pilih perbaikan otomatis per gambar")
    p.add_argument("-r", "--recursive", action="store_true", help="sertakan sub-folder")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("webcam", parents=[common], help="peningkatan citra real-time dari webcam")
    p.add_argument(
        "operation", nargs="?", default="equalize", help="operasi awal (default: equalize)"
    )
    p.add_argument(
        "--camera",
        type=_non_negative_int,
        default=0,
        metavar="INDEKS",
        help="indeks kamera (default: 0)",
    )
    p.add_argument(
        "-o", "--output", default=str(DEFAULT_OUTPUT), metavar="DIR", help="folder snapshot"
    )
    p.set_defaults(func=cmd_webcam)

    p = sub.add_parser(
        "serve", parents=[common], help="jalankan demo web (butuh: pip install 'imgenhance[web]')"
    )
    p.add_argument("--host", default="127.0.0.1", help="alamat bind (default: 127.0.0.1)")
    p.add_argument("--port", type=_positive_int, default=8000, help="port (default: 8000)")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose, args.quiet)
    try:
        return int(args.func(args))
    except (ParameterError, OperationError) as exc:
        logger.error("%s", exc)
        return EXIT_USAGE
    except EnhanceError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        logger.info("Dihentikan oleh pengguna.")
        return EXIT_INTERRUPTED
    except BrokenPipeError:  # mis. `imgenhance list | head`
        _silence_stdout()
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
