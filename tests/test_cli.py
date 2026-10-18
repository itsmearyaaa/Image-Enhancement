from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from imgenhance import __version__, read_image, write_image
from imgenhance import cli as cli_module
from imgenhance.cli import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, build_parser, main
from imgenhance.operations import OPERATIONS


@pytest.fixture
def dark(samples_dir: Path) -> str:
    return str(samples_dir / "sample_astronaut_gelap.png")


@pytest.fixture
def good(samples_dir: Path) -> str:
    return str(samples_dir / "sample_camera.png")


def run(*argv: str) -> int:
    return main(
        [*argv, "--no-show"]
        if argv[0] in {"apply", "pipeline", "auto", "analyze", "all"}
        else list(argv)
    )


# --------------------------------------------------------------------------- #
# list / version / bantuan
# --------------------------------------------------------------------------- #
def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_list_text_mentions_every_operation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == EXIT_OK
    out = capsys.readouterr().out
    for name in OPERATIONS:
        assert name in out
    assert "--threshold" in out and "rumus" in out


def test_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert [op["name"] for op in data] == list(OPERATIONS)


def test_every_operation_has_a_working_subcommand_help(capsys: pytest.CaptureFixture[str]) -> None:
    for name in OPERATIONS:
        with pytest.raises(SystemExit) as exc:
            main(["apply", name, "--help"])
        assert exc.value.code == 0
        assert "Rumus" in capsys.readouterr().out


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def test_apply_writes_result_and_comparison(
    dark: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("apply", "equalize", dark, "-o", str(tmp_path)) == EXIT_OK
    out = capsys.readouterr().out
    assert "sample_astronaut_gelap.png" in out and "Rata-rata" in out
    result = read_image(tmp_path / "sample_astronaut_gelap_equalize.png")
    original = read_image(dark)
    assert result.mean() > original.mean() + 40
    assert (tmp_path / "sample_astronaut_gelap_equalize_compare.png").is_file()


def test_apply_with_parameters(dark: str, tmp_path: Path) -> None:
    assert (
        run("apply", "brightness", dark, "--value", "-40", "-o", str(tmp_path), "--no-compare")
        == EXIT_OK
    )
    out = read_image(tmp_path / "sample_astronaut_gelap_brightness.png")
    assert out.mean() < read_image(dark).mean()
    assert not list(tmp_path.glob("*_compare.png"))


def test_apply_boolean_and_choice_flags(
    good: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("apply", "threshold", good, "--otsu", "-o", str(tmp_path), "--json") == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert "Otsu" in data["notes"][0]
    assert set(np.unique(read_image(tmp_path / "sample_camera_threshold.png"))) == {0, 255}
    assert (
        run("apply", "grayscale", good, "--method", "average", "-o", str(tmp_path), "-q") == EXIT_OK
    )
    assert (
        run("apply", "equalize", good, "--color-mode", "channels", "-o", str(tmp_path), "-q")
        == EXIT_OK
    )


def test_apply_json_output_is_valid(
    dark: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run("apply", "gamma", dark, "--gamma", "0.4", "-o", str(tmp_path), "--json", "-q")
        == EXIT_OK
    )
    data = json.loads(capsys.readouterr().out)
    assert data["file"] == "sample_astronaut_gelap.png" and data["steps"] == "gamma(gamma=0.4)"
    assert data["after"]["mean"] > data["before"]["mean"]
    assert Path(data["output"]).is_file() and Path(data["compare"]).is_file()


@pytest.mark.parametrize(
    "args",
    [
        ["--value", "999"],  # di luar rentang
        ["--value", "abc"],  # bukan angka
        ["--value", "1.5"],  # bukan bilangan bulat
    ],
)
def test_apply_bad_parameter_exits_with_usage_code(
    dark: str, tmp_path: Path, args: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    assert run("apply", "brightness", dark, *args, "-o", str(tmp_path)) == EXIT_USAGE
    assert not list(tmp_path.iterdir())


def test_apply_unknown_operation_and_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["apply", "tidak-ada", "x.png"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["apply", "negative", "x.png", "--tidak-ada", "1"])
    assert exc.value.code == 2


def test_apply_invalid_choice_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["apply", "grayscale", "x.png", "--method", "rata"])
    assert exc.value.code == 2


def test_apply_missing_file_exits_with_error(tmp_path: Path) -> None:
    assert run("apply", "negative", str(tmp_path / "hilang.png"), "-o", str(tmp_path)) == EXIT_ERROR


def test_apply_corrupt_file_exits_with_error(tmp_path: Path) -> None:
    bad = tmp_path / "rusak.png"
    bad.write_bytes(b"bukan gambar")
    assert run("apply", "negative", str(bad), "-o", str(tmp_path / "out")) == EXIT_ERROR


def test_apply_max_side_shrinks_before_processing(dark: str, tmp_path: Path) -> None:
    assert run("apply", "negative", dark, "--max-side", "128", "-o", str(tmp_path), "-q") == EXIT_OK
    assert max(read_image(tmp_path / "sample_astronaut_gelap_negative.png").shape[:2]) == 128


def test_apply_to_grayscale_file_keeps_it_grayscale(good: str, tmp_path: Path) -> None:
    assert run("apply", "equalize", good, "-o", str(tmp_path), "-q") == EXIT_OK
    assert read_image(tmp_path / "sample_camera_equalize.png").ndim == 2


# --------------------------------------------------------------------------- #
# pipeline / auto
# --------------------------------------------------------------------------- #
def test_pipeline_chains_steps(dark: str, tmp_path: Path) -> None:
    code = run(
        "pipeline", dark, "grayscale", "gamma:gamma=0.5", "equalize", "-o", str(tmp_path), "-q"
    )
    assert code == EXIT_OK
    out = read_image(tmp_path / "sample_astronaut_gelap_grayscale_gamma_equalize.png")
    assert out.ndim == 2
    assert (tmp_path / "sample_astronaut_gelap_grayscale_gamma_equalize_compare.png").is_file()


def test_pipeline_rejects_unknown_step_and_bad_params(dark: str, tmp_path: Path) -> None:
    assert run("pipeline", dark, "tidak-ada", "-o", str(tmp_path)) == EXIT_USAGE
    assert run("pipeline", dark, "gamma:gamma=99", "-o", str(tmp_path)) == EXIT_USAGE
    assert run("pipeline", dark, "gamma:gamma", "-o", str(tmp_path)) == EXIT_USAGE


def test_auto_enhances_a_dark_image(
    dark: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("auto", dark, "-o", str(tmp_path), "-q") == EXIT_OK
    assert "gamma" in capsys.readouterr().out
    assert (
        read_image(tmp_path / "sample_astronaut_gelap_gamma.png").mean() > read_image(dark).mean()
    )


def test_auto_leaves_a_good_image_alone(
    good: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("auto", good, "-o", str(tmp_path)) == EXIT_OK
    assert "tidak perlu perbaikan" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# analyze / all
# --------------------------------------------------------------------------- #
def test_analyze_text_and_histogram_figure(
    dark: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("analyze", dark, "-o", str(tmp_path), "-q") == EXIT_OK
    out = capsys.readouterr().out
    assert "gelap" in out and "Saran" in out
    assert (tmp_path / "sample_astronaut_gelap_histogram.png").is_file()


def test_analyze_json(dark: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("analyze", dark, "-o", str(tmp_path), "--json", "-q") == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["diagnosis"]["brightness"] == "gelap" and data["mean"] < 85


def test_all_writes_every_operation_and_a_contact_sheet(dark: str, tmp_path: Path) -> None:
    assert run("all", dark, "-o", str(tmp_path), "-q") == EXIT_OK
    produced = {p.name for p in tmp_path.iterdir()}
    for name in OPERATIONS:
        assert f"sample_astronaut_gelap_{name}.png" in produced
    assert "sample_astronaut_gelap_semua.png" in produced


def test_all_json(dark: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("all", dark, "-o", str(tmp_path), "--json", "-q") == EXIT_OK
    assert json.loads(capsys.readouterr().out)["operations"] == list(OPERATIONS)


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #
def _folder(tmp_path: Path, samples_dir: Path) -> Path:
    folder = tmp_path / "foto"
    for name in ("sample_astronaut_gelap.png", "sample_chelsea_pudar.png", "sample_camera.png"):
        write_image(folder / name, read_image(samples_dir / name))
    return folder


def test_batch_with_steps_writes_results_and_summary(
    tmp_path: Path, samples_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder, out = _folder(tmp_path, samples_dir), tmp_path / "hasil"
    assert main(["batch", str(folder), "equalize", "-o", str(out), "--no-compare", "-q"]) == EXIT_OK
    assert len(list(out.glob("*_equalize.png"))) == 3
    with (out / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 and {r["langkah"] for r in rows} == {"equalize"}
    assert capsys.readouterr().out.count("Operasi : equalize") == 3


def test_batch_auto_skips_good_images(tmp_path: Path, samples_dir: Path) -> None:
    folder, out = _folder(tmp_path, samples_dir), tmp_path / "hasil"
    assert main(["batch", str(folder), "--auto", "-o", str(out), "--no-compare", "-q"]) == EXIT_OK
    names = {p.name for p in out.glob("*.png")}
    assert names == {"sample_astronaut_gelap_gamma.png", "sample_chelsea_pudar_stretch.png"}


def test_batch_json(tmp_path: Path, samples_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = _folder(tmp_path, samples_dir)
    assert (
        main(["batch", str(folder), "negative", "-o", str(tmp_path / "o"), "--json", "-q"])
        == EXIT_OK
    )
    assert len(json.loads(capsys.readouterr().out)) == 3


def test_batch_continues_after_a_broken_file(tmp_path: Path, samples_dir: Path) -> None:
    folder, out = _folder(tmp_path, samples_dir), tmp_path / "hasil"
    (folder / "rusak.png").write_bytes(b"bukan gambar")
    assert (
        main(["batch", str(folder), "negative", "-o", str(out), "-q"]) == EXIT_ERROR
    )  # ada yang gagal
    assert len(list(out.glob("*_negative.png"))) == 3  # sisanya tetap diproses


def test_batch_recursive(tmp_path: Path, samples_dir: Path) -> None:
    folder = _folder(tmp_path, samples_dir)
    write_image(folder / "dalam" / "x.png", read_image(samples_dir / "sample_camera.png"))
    out = tmp_path / "hasil"
    assert (
        main(["batch", str(folder), "negative", "-r", "-o", str(out), "--no-compare", "-q"])
        == EXIT_OK
    )
    assert (out / "dalam__x_negative.png").is_file()


def test_batch_needs_steps_or_auto_and_a_real_folder(tmp_path: Path, samples_dir: Path) -> None:
    folder = _folder(tmp_path, samples_dir)
    assert main(["batch", str(folder), "-o", str(tmp_path / "o"), "-q"]) == EXIT_USAGE
    assert main(["batch", str(tmp_path / "tidak-ada"), "negative", "-q"]) == EXIT_ERROR
    empty = tmp_path / "kosong"
    empty.mkdir()
    assert main(["batch", str(empty), "negative", "-q"]) == EXIT_ERROR


# --------------------------------------------------------------------------- #
# webcam / serve / sinyal
# --------------------------------------------------------------------------- #
def test_webcam_requires_a_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "_has_display", lambda: False)
    assert main(["webcam", "-q"]) == EXIT_ERROR


def test_webcam_passes_arguments_to_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, Path]] = []
    monkeypatch.setattr(cli_module, "_has_display", lambda: True)
    monkeypatch.setattr(cli_module, "run_webcam", lambda op, cam, out: calls.append((op, cam, out)))
    assert main(["webcam", "gamma", "--camera", "2", "-o", "snap", "-q"]) == EXIT_OK
    assert calls == [("gamma", 2, Path("snap"))]


def test_webcam_unknown_operation_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "_has_display", lambda: True)
    assert main(["webcam", "tidak-ada", "-q"]) == EXIT_USAGE


def test_serve_starts_uvicorn_with_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    import uvicorn

    started: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(app=app, **kw))
    assert main(["serve", "--port", "9123", "-q"]) == EXIT_OK
    assert started["port"] == 9123 and started["host"] == "127.0.0.1"


def test_keyboard_interrupt_gives_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "cmd_list", boom)
    assert main(["list", "-q"]) == EXIT_INTERRUPTED


def test_broken_pipe_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_args: object) -> int:
        raise BrokenPipeError

    monkeypatch.setattr(cli_module, "cmd_list", broken)
    monkeypatch.setattr(cli_module, "_silence_stdout", lambda: None)
    assert main(["list", "-q"]) == EXIT_OK


def test_show_does_nothing_without_a_display(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    shown: list[str] = []
    monkeypatch.setattr(cli_module, "_has_display", lambda: False)
    monkeypatch.setattr(cli_module.cv2, "imshow", lambda *a, **k: shown.append("imshow"))
    img = np.zeros((4, 4), dtype=np.uint8)
    cli_module._show("t", img, None)  # otomatis: tidak ada layar -> diam
    with caplog.at_level("WARNING", logger="imgenhance"):
        cli_module._show("t", img, True)  # diminta tetapi tidak mungkin -> peringatan
    assert shown == [] and "Tidak ada layar" in caplog.text


def test_show_opens_a_window_when_possible(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli_module, "_has_display", lambda: True)
    monkeypatch.setattr(cli_module.cv2, "imshow", lambda *a, **k: events.append("imshow"))
    monkeypatch.setattr(cli_module.cv2, "waitKey", lambda *_: events.append("waitKey") or 0)
    monkeypatch.setattr(cli_module.cv2, "destroyAllWindows", lambda: events.append("destroy"))
    cli_module._show("t", np.zeros((4, 4), dtype=np.uint8), None)
    assert events == ["imshow", "waitKey", "destroy"]
    events.clear()
    cli_module._show("t", np.zeros((4, 4), dtype=np.uint8), False)  # diminta tidak tampil
    assert events == []


def test_has_display_by_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert cli_module._has_display() is False
    monkeypatch.setenv("DISPLAY", ":0")
    assert cli_module._has_display() is True
    monkeypatch.setattr(cli_module.sys, "platform", "win32")
    monkeypatch.delenv("DISPLAY")
    assert cli_module._has_display() is True


def test_argument_types() -> None:
    import argparse

    assert cli_module._non_negative_int("0") == 0
    assert cli_module._positive_int("5") == 5
    with pytest.raises(argparse.ArgumentTypeError):
        cli_module._non_negative_int("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        cli_module._positive_int("0")


def test_show_survives_opencv_without_gui(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import cv2

    def no_gui(*_a: object, **_k: object) -> None:
        raise cv2.error("The function is not implemented. Rebuild the library with GTK")

    monkeypatch.setattr(cli_module, "_has_display", lambda: True)
    monkeypatch.setattr(cli_module.cv2, "imshow", no_gui)
    monkeypatch.setattr(cli_module.cv2, "destroyAllWindows", no_gui)
    with caplog.at_level("WARNING", logger="imgenhance"):
        cli_module._show("t", np.zeros((4, 4), dtype=np.uint8), True)  # tidak boleh melempar
    assert "tidak bisa dibuka" in caplog.text
