"""Demo web (FastAPI): unggah foto, pilih operasi, atur parameter, lihat hasil + histogram + kurva.

Jalankan dengan ``imgenhance serve`` atau ``uvicorn --factory imgenhance.web.app:create_app``.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .. import __version__
from ..analysis import ImageStats, diagnose
from ..errors import EnhanceError
from ..histogram import channel_histograms
from ..imageio import decode_image, encode_image, resize_max_side
from ..operations import get_operation, list_operations

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_SIDE_LIMIT = 4096
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

INDEX_HTML = Path(__file__).with_name("index.html").read_text(encoding="utf-8")


def _histograms(image: Any) -> dict[str, list[int]]:
    return {name: [int(v) for v in hist] for name, hist in channel_histograms(image).items()}


def _parse_params(raw: str) -> dict[str, Any]:
    try:
        params = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"params bukan JSON yang valid: {exc}") from exc
    if not isinstance(params, dict):
        raise HTTPException(status_code=422, detail="params harus berupa objek JSON")
    return params


def create_app() -> FastAPI:
    """Pabrik aplikasi FastAPI (memudahkan pengujian dan konfigurasi)."""
    app = FastAPI(title="ImgEnhance", version=__version__, docs_url="/api/docs", redoc_url=None)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/operations")
    def operations() -> dict[str, Any]:
        return {"operations": [op.to_dict() for op in list_operations()]}

    @app.post("/api/enhance")
    def enhance(
        file: UploadFile = File(...),
        operation: str = Form("equalize"),
        params: str = Form("{}"),
        max_side: int = Form(1280),
    ) -> dict[str, Any]:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=415, detail="Format tidak didukung; gunakan JPG, PNG, WebP, atau BMP."
            )
        if not 0 <= max_side <= MAX_SIDE_LIMIT:
            raise HTTPException(
                status_code=422, detail=f"max_side harus 0..{MAX_SIDE_LIMIT} (0 = ukuran asli)"
            )
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Ukuran file melebihi 8 MB.")

        given = _parse_params(params)
        try:
            image = resize_max_side(decode_image(data, file.filename or "upload"), max_side)
            op = get_operation(operation)
            resolved = op.resolve(given)
            result = op.handler(image, resolved)
            png = encode_image(result.image, ".png")
        except EnhanceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        before = ImageStats.from_image(image)
        after = ImageStats.from_image(result.image)
        diagnosis = diagnose(before)
        logger.info("Enhance %s: %s", file.filename, op.name)
        height, width = result.image.shape[:2]
        return {
            "operation": {"name": op.name, "title": op.title, "formula": op.formula},
            "params": resolved,
            "note": result.note,
            "image_png": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
            "size": [width, height],
            "before": {"stats": before.to_dict(), "histogram": _histograms(image)},
            "after": {"stats": after.to_dict(), "histogram": _histograms(result.image)},
            "curve": [int(v) for v in result.lut] if result.lut is not None else None,
            "diagnosis": {
                **diagnosis.to_dict(),
                "suggestion": [{"name": s.name, "params": dict(s.params)} for s in diagnosis.steps],
            },
        }

    return app
