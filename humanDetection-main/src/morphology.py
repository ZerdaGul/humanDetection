"""
Morphological operations for binary image processing.

Supports multiple structuring element shapes and sizes.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


StructuringElementShape = Literal["rect", "cross", "ellipse"]


def get_structuring_element(
    shape: StructuringElementShape = "rect",
    size: int = 3
) -> np.ndarray:
    """
    Create a structuring element for morphological operations.
    
    Args:
        shape: Element shape - "rect", "cross", or "ellipse"
        size: Element size (will be forced to odd)
    
    Returns:
        Binary structuring element as uint8 array
    """
    size = max(3, size if size % 2 == 1 else size + 1)
    center = size // 2
    
    if shape == "rect":
        return np.ones((size, size), dtype=np.uint8)
    
    elif shape == "cross":
        element = np.zeros((size, size), dtype=np.uint8)
        element[center, :] = 1
        element[:, center] = 1
        return element
    
    elif shape == "ellipse":
        y, x = np.ogrid[:size, :size]
        mask = ((x - center) ** 2 + (y - center) ** 2) <= center ** 2
        return mask.astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown structuring element shape: {shape}")


def _apply_morphology(
    image: np.ndarray,
    kernel: np.ndarray,
    operation: Literal["erode", "dilate"]
) -> np.ndarray:
    """
    Apply a single morphological operation.
    
    Uses min/max operations which are faster than boolean comparisons.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad with 0 for erode (shrinks white), 255 for dilate (grows white)
    pad_value = 0 if operation == "erode" else 0
    padded = np.pad(
        image.astype(np.uint8),
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=pad_value
    )
    
    windows = sliding_window_view(padded, kernel.shape)
    
    # Create mask for kernel positions
    kernel_mask = kernel == 1
    
    if operation == "erode":
        # All kernel positions must be 255 → use min
        # Flatten windows at kernel positions and take min
        masked = np.where(kernel_mask, windows, 255)
        result = np.min(masked, axis=(-2, -1))
    else:  # dilate
        # Any kernel position being 255 → use max
        masked = np.where(kernel_mask, windows, 0)
        result = np.max(masked, axis=(-2, -1))
    
    return result.astype(np.uint8)


def erode(
    image: np.ndarray,
    iterations: int = 1,
    kernel_size: int = 3,
    kernel_shape: StructuringElementShape = "rect",
    kernel: np.ndarray | None = None
) -> np.ndarray:
    """
    Erode a binary image - shrinks white regions.
    
    Args:
        image: Binary input image (0 and 255)
        iterations: Number of times to apply erosion
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        kernel: Custom kernel (overrides size and shape)
    
    Returns:
        Eroded binary image
    """
    if kernel is None:
        kernel = get_structuring_element(kernel_shape, kernel_size)
    
    result = image.copy()
    for _ in range(max(1, iterations)):
        result = _apply_morphology(result, kernel, "erode")
    
    return result


def dilate(
    image: np.ndarray,
    iterations: int = 1,
    kernel_size: int = 3,
    kernel_shape: StructuringElementShape = "rect",
    kernel: np.ndarray | None = None
) -> np.ndarray:
    """
    Dilate a binary image - grows white regions.
    
    Args:
        image: Binary input image (0 and 255)
        iterations: Number of times to apply dilation
        kernel_size: Size of structuring element
        kernel_shape: Shape of structuring element
        kernel: Custom kernel (overrides size and shape)
    
    Returns:
        Dilated binary image
    """
    if kernel is None:
        kernel = get_structuring_element(kernel_shape, kernel_size)
    
    result = image.copy()
    for _ in range(max(1, iterations)):
        result = _apply_morphology(result, kernel, "dilate")
    
    return result


def opening(
    image: np.ndarray,
    iterations: int = 1,
    kernel_size: int = 3,
    kernel_shape: StructuringElementShape = "rect",
    kernel: np.ndarray | None = None
) -> np.ndarray:
    """
    Morphological opening = Erode → Dilate.
    
    Removes small white noise while preserving shape of larger objects.
    """
    if kernel is None:
        kernel = get_structuring_element(kernel_shape, kernel_size)
    
    result = erode(image, iterations, kernel=kernel)
    result = dilate(result, iterations, kernel=kernel)
    
    return result


def closing(
    image: np.ndarray,
    iterations: int = 1,
    kernel_size: int = 3,
    kernel_shape: StructuringElementShape = "rect",
    kernel: np.ndarray | None = None
) -> np.ndarray:
    """
    Morphological closing = Dilate → Erode.
    
    Fills small holes in white regions while preserving shape.
    """
    if kernel is None:
        kernel = get_structuring_element(kernel_shape, kernel_size)
    
    result = dilate(image, iterations, kernel=kernel)
    result = erode(result, iterations, kernel=kernel)
    
    return result


def gradient(
    image: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: StructuringElementShape = "rect",
    kernel: np.ndarray | None = None
) -> np.ndarray:
    """
    Morphological gradient = Dilate - Erode.
    
    Extracts edges/outlines of objects.
    """
    if kernel is None:
        kernel = get_structuring_element(kernel_shape, kernel_size)
    
    dilated = dilate(image, 1, kernel=kernel)
    eroded = erode(image, 1, kernel=kernel)
    
    return (dilated.astype(np.int16) - eroded.astype(np.int16)).clip(0, 255).astype(np.uint8)


def tophat(
    image: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: StructuringElementShape = "rect",
    kernel: np.ndarray | None = None
) -> np.ndarray:
    """
    Top-hat transform = Original - Opening.
    
    Extracts bright details smaller than the structuring element.
    """
    if kernel is None:
        kernel = get_structuring_element(kernel_shape, kernel_size)
    
    opened = opening(image, 1, kernel=kernel)
    
    return (image.astype(np.int16) - opened.astype(np.int16)).clip(0, 255).astype(np.uint8)


def blackhat(
    image: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: StructuringElementShape = "rect",
    kernel: np.ndarray | None = None
) -> np.ndarray:
    """
    Black-hat transform = Closing - Original.
    
    Extracts dark details smaller than the structuring element.
    """
    if kernel is None:
        kernel = get_structuring_element(kernel_shape, kernel_size)
    
    closed = closing(image, 1, kernel=kernel)
    
    return (closed.astype(np.int16) - image.astype(np.int16)).clip(0, 255).astype(np.uint8)


def hit_or_miss(
    image: np.ndarray,
    hit_kernel: np.ndarray,
    miss_kernel: np.ndarray
) -> np.ndarray:
    """
    Hit-or-miss transform for pattern matching.
    
    Args:
        image: Binary input image
        hit_kernel: Pattern to match in foreground (1s)
        miss_kernel: Pattern to match in background (1s)
    
    Returns:
        Binary image with matched locations
    """
    # Erode original with hit kernel
    hit_result = erode(image, 1, kernel=hit_kernel)
    
    # Erode complement with miss kernel
    complement = 255 - image
    miss_result = erode(complement, 1, kernel=miss_kernel)
    
    # Intersection
    return np.minimum(hit_result, miss_result)