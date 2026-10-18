"""imgenhance - peningkatan kualitas citra (*image enhancement*) dengan Python dan OpenCV.

Penggunaan sebagai library::

    from imgenhance import apply_operation, read_image

    result = apply_operation(read_image("foto.jpg"), "equalize")
    enhanced = result.image          # array NumPy (BGR atau grayscale)
"""

from __future__ import annotations

__version__ = "1.0.0"

from .analysis import Diagnosis, ImageStats, auto_pipeline, diagnose, diagnose_image
from .errors import (
    CameraError,
    EnhanceError,
    ImageLoadError,
    ImageWriteError,
    OperationError,
    ParameterError,
)
from .histogram import channel_histograms, equalization_map, equalize, equalize_lut, histogram
from .imageio import decode_image, encode_image, read_image, write_image
from .operations import (
    OPERATIONS,
    EnhanceResult,
    Operation,
    Param,
    apply_operation,
    get_operation,
    list_operations,
)
from .pipeline import Pipeline, PipelineResult, Step
from .point import (
    apply_lut,
    bitplane_lut,
    brightness_lut,
    contrast_lut,
    gamma_lut,
    log_lut,
    negative_lut,
    quantize_lut,
    slice_lut,
    stretch_lut,
    threshold_lut,
    to_grayscale,
)
from .spatial import median_filter, sharpen, smooth

__all__ = [
    "OPERATIONS",
    "CameraError",
    "Diagnosis",
    "EnhanceError",
    "EnhanceResult",
    "ImageLoadError",
    "ImageStats",
    "ImageWriteError",
    "Operation",
    "OperationError",
    "Param",
    "ParameterError",
    "Pipeline",
    "PipelineResult",
    "Step",
    "__version__",
    "apply_lut",
    "apply_operation",
    "auto_pipeline",
    "bitplane_lut",
    "brightness_lut",
    "channel_histograms",
    "contrast_lut",
    "decode_image",
    "diagnose",
    "diagnose_image",
    "encode_image",
    "equalization_map",
    "equalize",
    "equalize_lut",
    "gamma_lut",
    "get_operation",
    "histogram",
    "list_operations",
    "log_lut",
    "median_filter",
    "negative_lut",
    "quantize_lut",
    "read_image",
    "sharpen",
    "slice_lut",
    "smooth",
    "stretch_lut",
    "threshold_lut",
    "to_grayscale",
    "write_image",
]
