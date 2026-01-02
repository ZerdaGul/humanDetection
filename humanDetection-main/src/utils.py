"""
Utility functions for the human detection pipeline.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Union

import numpy as np

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore


# ============================================================================
# YAML Parsing
# ============================================================================

def _parse_scalar(value: str) -> Any:
    """Parse a scalar YAML-like value without external dependencies."""
    value = value.strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None

    # List support: [1, 2, 3]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = _split_list_items(inner)
        return [_parse_scalar(part) for part in parts]

    # String with quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    # Numeric parsing
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_list_items(text: str) -> List[str]:
    """Split list items handling nested brackets."""
    items = []
    current = []
    depth = 0
    
    for char in text:
        if char == "[":
            depth += 1
            current.append(char)
        elif char == "]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    
    if current:
        items.append("".join(current).strip())
    
    return [item for item in items if item]


def _minimal_yaml_parser(text: str) -> Dict[str, Any]:
    """Parse a subset of YAML (nested dicts, scalars, lists)."""
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        
        if ":" not in raw_line:
            continue
            
        colon_idx = raw_line.index(":")
        key = raw_line[:colon_idx].strip()
        remainder = raw_line[colon_idx + 1:].strip()

        # Pop stack until we find parent
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        
        parent = stack[-1][1]

        if not remainder:
            new_dict: Dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            parent[key] = _parse_scalar(remainder)

    return root


def load_settings(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
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


# ============================================================================
# Image Processing Utilities
# ============================================================================

def ensure_odd(value: int) -> int:
    """Force kernel sizes to be odd numbers."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Bilinear interpolation resize - better quality than nearest neighbor.
    
    Args:
        frame: Input image (grayscale or color)
        width: Target width
        height: Target height
    
    Returns:
        Resized image
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")

    src_h, src_w = frame.shape[:2]
    if src_h == height and src_w == width:
        return frame

    # Create coordinate grids
    y_ratio = (src_h - 1) / max(1, height - 1)
    x_ratio = (src_w - 1) / max(1, width - 1)
    
    y_coords = np.arange(height) * y_ratio
    x_coords = np.arange(width) * x_ratio
    
    # Integer and fractional parts
    y_floor = np.floor(y_coords).astype(np.int32)
    x_floor = np.floor(x_coords).astype(np.int32)
    y_ceil = np.minimum(y_floor + 1, src_h - 1)
    x_ceil = np.minimum(x_floor + 1, src_w - 1)
    
    y_frac = y_coords - y_floor
    x_frac = x_coords - x_floor
    
    # Bilinear interpolation
    if frame.ndim == 2:
        # Grayscale
        top_left = frame[y_floor][:, x_floor]
        top_right = frame[y_floor][:, x_ceil]
        bottom_left = frame[y_ceil][:, x_floor]
        bottom_right = frame[y_ceil][:, x_ceil]
        
        top = top_left + (top_right - top_left) * x_frac
        bottom = bottom_left + (bottom_right - bottom_left) * x_frac
        result = top + (bottom - top) * y_frac[:, np.newaxis]
    else:
        # Color image
        top_left = frame[y_floor][:, x_floor]
        top_right = frame[y_floor][:, x_ceil]
        bottom_left = frame[y_ceil][:, x_floor]
        bottom_right = frame[y_ceil][:, x_ceil]
        
        top = top_left + (top_right - top_left) * x_frac[:, np.newaxis]
        bottom = bottom_left + (bottom_right - bottom_left) * x_frac[:, np.newaxis]
        result = top + (bottom - top) * y_frac[:, np.newaxis, np.newaxis]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def resize_frame_fast(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Fast nearest-neighbor resize for performance-critical paths.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")

    src_h, src_w = frame.shape[:2]
    if src_h == height and src_w == width:
        return frame

    y_indices = (np.arange(height) * src_h // height).astype(np.int32)
    x_indices = (np.arange(width) * src_w // width).astype(np.int32)
    
    return frame[y_indices][:, x_indices]


def clamp(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Union[int, float]:
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))