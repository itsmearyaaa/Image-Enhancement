from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from imgenhance import decode_image
from imgenhance.operations import OPERATIONS
from imgenhance.web import app as web_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(web_app.create_app())


@pytest.fixture
def dark_bytes(samples_dir: Path) -> bytes:
    return (samples_dir / "sample_astronaut_gelap.png").read_bytes()


def _post(client: TestClient, data: bytes, **form: str) -> object:
    return client.post("/api/enhance", files={"file": ("foto.png", data, "image/png")}, data=form)


def test_health_and_index(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "ImgEnhance" in page.text


def test_operations_endpoint_matches_registry(client: TestClient) -> None:
    body = client.get("/api/operations").json()
    assert [op["name"] for op in body["operations"]] == list(OPERATIONS)
    gamma = next(op for op in body["operations"] if op["name"] == "gamma")
    assert gamma["params"][0]["kind"] == "float" and gamma["formula"]


def test_enhance_returns_image_stats_histograms_and_curve(
    client: TestClient, dark_bytes: bytes
) -> None:
    response = _post(client, dark_bytes, operation="gamma", params=json.dumps({"gamma": 0.4}))
    assert response.status_code == 200  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert body["operation"]["name"] == "gamma" and body["params"]["gamma"] == 0.4
    assert body["image_png"].startswith("data:image/png;base64,")
    result = decode_image(base64.b64decode(body["image_png"].split(",", 1)[1]))
    assert result.shape == (512, 512, 3) and result.mean() > 48 + 20
    assert body["after"]["stats"]["mean"] > body["before"]["stats"]["mean"]
    assert set(body["before"]["histogram"]) == {"blue", "green", "red"}
    assert len(body["curve"]) == 256 and body["curve"][0] == 0 and body["curve"][255] == 255
    assert body["diagnosis"]["brightness"] == "gelap"
    assert body["diagnosis"]["suggestion"][0]["name"] == "gamma"


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_every_operation_works_through_the_api(
    client: TestClient, dark_bytes: bytes, name: str
) -> None:
    response = _post(client, dark_bytes, operation=name, max_side="128")
    assert response.status_code == 200, response.text  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert max(body["size"]) == 128
    assert (body["curve"] is None) == (name in {"grayscale", "smooth", "median", "sharpen"})


def test_grayscale_upload_and_grayscale_output_histograms(
    client: TestClient, samples_dir: Path
) -> None:
    data = (samples_dir / "sample_camera.png").read_bytes()
    body = _post(client, data, operation="equalize").json()  # type: ignore[attr-defined]
    assert list(body["before"]["histogram"]) == ["gray"]
    assert sum(body["after"]["histogram"]["gray"]) == 512 * 512


def test_default_operation_is_equalize(client: TestClient, dark_bytes: bytes) -> None:
    response = client.post("/api/enhance", files={"file": ("a.png", dark_bytes, "image/png")})
    assert response.status_code == 200 and response.json()["operation"]["name"] == "equalize"


@pytest.mark.parametrize(
    ("form", "status"),
    [
        ({"operation": "tidak-ada"}, 422),
        ({"operation": "gamma", "params": json.dumps({"gamma": 99})}, 422),
        ({"operation": "gamma", "params": json.dumps({"warna": 1})}, 422),
        ({"operation": "gamma", "params": "{bukan json"}, 422),
        ({"operation": "gamma", "params": "[1, 2]"}, 422),
        ({"operation": "gamma", "max_side": "99999"}, 422),
        ({"operation": "gamma", "max_side": "-1"}, 422),
    ],
)
def test_bad_requests_are_rejected_with_422(
    client: TestClient, dark_bytes: bytes, form: dict[str, str], status: int
) -> None:
    response = _post(client, dark_bytes, **form)
    assert response.status_code == status  # type: ignore[attr-defined]
    assert response.json()["detail"]  # type: ignore[attr-defined]


def test_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post("/api/enhance", files={"file": ("a.txt", b"halo", "text/plain")})
    assert response.status_code == 415


def test_rejects_oversized_upload(
    client: TestClient, dark_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_app, "MAX_UPLOAD_BYTES", 1024)
    assert _post(client, dark_bytes).status_code == 413  # type: ignore[attr-defined]


def test_corrupt_image_returns_422(client: TestClient) -> None:
    response = _post(client, b"bukan gambar")
    assert response.status_code == 422  # type: ignore[attr-defined]
    assert "valid" in response.json()["detail"]  # type: ignore[attr-defined]


def test_results_are_deterministic(client: TestClient, dark_bytes: bytes) -> None:
    first = _post(client, dark_bytes, operation="equalize").json()  # type: ignore[attr-defined]
    second = _post(client, dark_bytes, operation="equalize").json()  # type: ignore[attr-defined]
    assert first == second
    assert np.isfinite(first["after"]["stats"]["entropy"])
