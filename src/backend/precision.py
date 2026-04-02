"""Precision interface definitions for QUBO-related compiler paths."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def mlir_dtype(self) -> str:
        if self is Precision.FP32:
            return "f32"
        if self is Precision.FP16:
            return "f16"
        if self is Precision.INT8:
            return "i8"
        if self is Precision.INT4:
            return "i4"
        raise ValueError(f"Unsupported precision: {self}")

    @property
    def runtime_dtype(self) -> str:
        if self is Precision.FP32:
            return "float32"
        if self is Precision.FP16:
            return "float16"
        if self in {Precision.INT8, Precision.INT4}:
            return "int8"
        raise ValueError(f"Unsupported precision: {self}")


def normalize_precision(value: Optional[Union[str, Precision]]) -> Precision:
    if value is None:
        return Precision.FP32
    if isinstance(value, Precision):
        return value

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "fp32": Precision.FP32,
        "float32": Precision.FP32,
        "f32": Precision.FP32,
        "fp16": Precision.FP16,
        "float16": Precision.FP16,
        "f16": Precision.FP16,
        "int8": Precision.INT8,
        "i8": Precision.INT8,
        "int4": Precision.INT4,
        "i4": Precision.INT4,
    }
    if normalized in aliases:
        return aliases[normalized]

    try:
        return Precision(normalized)
    except ValueError as exc:
        valid = ", ".join(item.value for item in Precision)
        raise ValueError(f"Unsupported precision: {value!r}. Expected one of: {valid}") from exc