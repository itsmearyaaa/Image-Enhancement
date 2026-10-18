"""Rangkaian operasi (*pipeline*): beberapa operasi dijalankan berurutan.

Satu langkah ditulis sebagai teks ``nama`` atau ``nama:kunci=nilai,kunci=nilai``, misalnya::

    Pipeline.parse(["grayscale", "gamma:gamma=0.6", "equalize"])
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._types import Image
from .errors import ParameterError
from .operations import EnhanceResult, get_operation


@dataclass(frozen=True)
class Step:
    """Satu langkah: nama operasi + parameternya (yang tidak diisi memakai nilai bawaan)."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, spec: str) -> Step:
        """Urai ``"nama"`` atau ``"nama:kunci=nilai,kunci=nilai"``."""
        name, _, rest = spec.strip().partition(":")
        name = name.strip()
        if not name:
            raise ParameterError(f"Langkah kosong: '{spec}'")
        params: dict[str, Any] = {}
        for part in filter(None, (chunk.strip() for chunk in rest.split(","))):
            key, sep, value = part.partition("=")
            if not sep or not key.strip():
                raise ParameterError(
                    f"Parameter '{part}' pada langkah '{name}' harus berbentuk kunci=nilai"
                )
            params[key.strip()] = value.strip()
        return cls(name, params)

    def label(self) -> str:
        """Teks ringkas, mis. ``gamma(gamma=0.6)``."""
        if not self.params:
            return self.name
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({inner})"


@dataclass(frozen=True)
class StageResult:
    """Hasil satu langkah beserta parameter lengkap yang dipakai."""

    step: Step
    params: dict[str, Any]
    result: EnhanceResult


@dataclass(frozen=True)
class PipelineResult:
    """Hasil seluruh rangkaian."""

    image: Image
    stages: tuple[StageResult, ...]

    @property
    def lut(self) -> Any:
        """Kurva transformasi, hanya bila rangkaian terdiri dari satu langkah berbasis LUT."""
        return self.stages[0].result.lut if len(self.stages) == 1 else None

    @property
    def notes(self) -> list[str]:
        return [s.result.note for s in self.stages if s.result.note]


class Pipeline:
    """Urutan :class:`Step`; langkah divalidasi sejak pembuatan (gagal cepat)."""

    def __init__(self, steps: Sequence[Step]) -> None:
        if not steps:
            raise ParameterError("Pipeline minimal berisi satu langkah")
        self.steps: tuple[Step, ...] = tuple(steps)
        for step in self.steps:  # validasi dini: nama operasi dan parameter
            get_operation(step.name).resolve(step.params)

    @classmethod
    def parse(cls, specs: Sequence[str]) -> Pipeline:
        return cls([Step.parse(spec) for spec in specs])

    def label(self) -> str:
        return " > ".join(step.label() for step in self.steps)

    def run(self, image: Image) -> PipelineResult:
        """Jalankan semua langkah; keluaran satu langkah menjadi masukan langkah berikutnya."""
        stages: list[StageResult] = []
        current = image
        for step in self.steps:
            operation = get_operation(step.name)
            params = operation.resolve(step.params)
            result = operation.handler(current, params)
            stages.append(StageResult(step, params, result))
            current = result.image
        return PipelineResult(current, tuple(stages))
