"""
Gaussian blur implementation using separable filters for optimal performance.

Separable filtering reduces complexity from O(k²) to O(2k) where k is kernel size.
For a 15x15 kernel: 225 operations → 30 operations per pixel.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from utils import ensure_odd


def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 1D Gaussian kernel.
    
    Args:
        size: Kernel size (will be forced to odd)
        sigma: Standard deviation
    
    Returns:
        Normalized 1D kernel as float32 array
    """
    size = ensure_odd(max(3, int(size)))
    sigma = max(1e-6, float(sigma))
    
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    
    return kernel


def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel (for reference/comparison).
    
    Args:
        size: Kernel size (will be forced to odd)
        sigma: Standard deviation
    
    Returns:
        Normalized 2D kernel as float32 array
    """
    size = ensure_odd(max(3, int(size)))
    sigma = max(1e-6, float(sigma))
    
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    
    return kernel


def _convolve1d_horizontal(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 1D convolution along horizontal axis."""
    pad_w = len(kernel) // 2
    padded = np.pad(image, ((0, 0), (pad_w, pad_w)), mode="edge")
    windows = sliding_window_view(padded, len(kernel), axis=1)
    return np.tensordot(windows, kernel, axes=([2], [0]))


def _convolve1d_vertical(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 1D convolution along vertical axis."""
    pad_h = len(kernel) // 2
    padded = np.pad(image, ((pad_h, pad_h), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, len(kernel), axis=0)
    return np.tensordot(windows, kernel, axes=([2], [0]))


def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur using separable filtering.
    
    This is significantly faster than 2D convolution, especially for larger kernels.
    
    Args:
        image: Grayscale input image (2D numpy array)
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian
    
    Returns:
        Blurred image as uint8 array
    
    Raises:
        ValueError: If image is not grayscale (2D)
    """
    if image.ndim != 2:
        raise ValueError("apply_gaussian_blur expects a grayscale image")
    
    kernel = gaussian_kernel_1d(kernel_size, sigma)
    
    # Apply horizontal then vertical (order doesn't matter for Gaussian)
    result = _convolve1d_horizontal(image.astype(np.float32), kernel)
    result = _convolve1d_vertical(result, kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply box blur (mean filter) - faster than Gaussian but lower quality.
    
    Useful for quick preprocessing where quality isn't critical.
    
    Args:
        image: Grayscale input image
        kernel_size: Size of the box kernel
    
    Returns:
        Blurred image as uint8 array
    """
    if image.ndim != 2:
        raise ValueError("apply_box_blur expects a grayscale image")
    
    size = ensure_odd(max(3, int(kernel_size)))
    kernel = np.ones(size, dtype=np.float32) / size
    
    result = _convolve1d_horizontal(image.astype(np.float32), kernel)
    result = _convolve1d_vertical(result, kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)


# Convenience function for common sigma calculation
def sigma_from_kernel_size(kernel_size: int) -> float:
    """Calculate appropriate sigma for a given kernel size."""
    return kernel_size / 6.0