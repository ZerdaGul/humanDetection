import os
from typing import Any, Dict

import numpy as np


try:  # Prefer PyYAML when available for flexible parsing
    import yaml  # type: ignore
except ImportError:  # Fallback to a tiny parser implemented below
    yaml = None  # type: ignore


def _parse_scalar(value: str) -> Any:
    """Parse a scalar YAML-like value without external dependencies."""
    value = value.strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    # List support, e.g. [1, 2, 3]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [item.strip() for item in inner.split(",") if item.strip()]
        return [_parse_scalar(part) for part in parts]

    # Attempt integer / float parsing
    try:
        if value.startswith("0") and len(value) > 1 and value[1].isdigit():
            # Leave strings like file paths intact
            raise ValueError
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _minimal_yaml_parser(text: str) -> Dict[str, Any]:
    """Parse a very small subset of YAML (dicts + scalars + lists)."""
    root: Dict[str, Any] = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in raw_line:
            continue
        key, remainder = raw_line.strip().split(":", 1)
        value = remainder.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if not value:
            new_dict: Dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            parent[key] = _parse_scalar(value)

    return root


def load_settings(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file with an optional manual fallback."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Settings file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = _minimal_yaml_parser(text)

    if not isinstance(data, dict):
        raise ValueError("Settings file must define a dictionary at the top level")

    return data


def ensure_odd(value: int) -> int:
    """Force convolution kernel sizes to be odd numbers."""
    return value if value % 2 == 1 else value + 1


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbor resize implemented manually with NumPy."""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")

    src_h, src_w = frame.shape[:2]
    if src_h == height and src_w == width:
        return frame

    y_indices = np.linspace(0, src_h - 1, height).astype(np.int32)
    x_indices = np.linspace(0, src_w - 1, width).astype(np.int32)
    resized = frame[y_indices][:, x_indices]
    return resized
