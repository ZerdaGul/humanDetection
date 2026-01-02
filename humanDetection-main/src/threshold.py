"""
Thresholding operations for binary image segmentation.
"""
from __future__ import annotations

import numpy as np


def binary_threshold(image: np.ndarray, thresh: int, max_val: int = 255) -> np.ndarray:
    """
    Apply binary threshold to grayscale image.
    
    Args:
        image: Grayscale input image (2D)
        thresh: Threshold value (0-255)
        max_val: Value for pixels above threshold
    
    Returns:
        Binary image (0 and max_val)
    
    Raises:
        ValueError: If image is not grayscale
    """
    if image.ndim != 2:
        raise ValueError("binary_threshold expects a grayscale image")
    
    return np.where(image >= thresh, max_val, 0).astype(np.uint8)


def binary_threshold_inv(image: np.ndarray, thresh: int, max_val: int = 255) -> np.ndarray:
    """
    Apply inverse binary threshold.
    
    Pixels below threshold become max_val, above become 0.
    """
    if image.ndim != 2:
        raise ValueError("binary_threshold_inv expects a grayscale image")
    
    return np.where(image < thresh, max_val, 0).astype(np.uint8)


def adaptive_threshold_mean(
    image: np.ndarray,
    block_size: int = 11,
    c: float = 2.0,
    max_val: int = 255
) -> np.ndarray:
    """
    Adaptive threshold using local mean.
    
    Each pixel is compared against the mean of its local neighborhood minus constant C.
    Better for images with varying illumination.
    
    Args:
        image: Grayscale input image
        block_size: Size of local neighborhood (odd number)
        c: Constant subtracted from mean
        max_val: Value for pixels above local threshold
    
    Returns:
        Binary image
    """
    if image.ndim != 2:
        raise ValueError("adaptive_threshold expects a grayscale image")
    
    # Ensure odd block size
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    block_size = max(3, block_size)
    
    # Use integral image for fast mean calculation
    img_float = image.astype(np.float64)
    integral = np.cumsum(np.cumsum(img_float, axis=0), axis=1)
    
    # Pad integral image for easier boundary handling
    padded = np.zeros((integral.shape[0] + 1, integral.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = integral
    
    radius = block_size // 2
    h, w = image.shape
    
    # Calculate local sums using integral image
    y1 = np.clip(np.arange(h) - radius, 0, h).astype(np.int32)
    y2 = np.clip(np.arange(h) + radius + 1, 0, h).astype(np.int32)
    x1 = np.clip(np.arange(w) - radius, 0, w).astype(np.int32)
    x2 = np.clip(np.arange(w) + radius + 1, 0, w).astype(np.int32)
    
    # Compute local means
    local_sum = (
        padded[y2[:, None], x2[None, :]] -
        padded[y1[:, None], x2[None, :]] -
        padded[y2[:, None], x1[None, :]] +
        padded[y1[:, None], x1[None, :]]
    )
    
    # Calculate actual neighborhood sizes for border handling
    widths = x2[None, :] - x1[None, :]
    heights = y2[:, None] - y1[:, None]
    counts = widths * heights
    
    local_mean = local_sum / np.maximum(counts, 1)
    
    # Apply threshold
    return np.where(image > local_mean - c, max_val, 0).astype(np.uint8)


def otsu_threshold(image: np.ndarray, max_val: int = 255) -> tuple[np.ndarray, int]:
    """
    Otsu's automatic thresholding method.
    
    Finds optimal threshold that minimizes intra-class variance.
    
    Args:
        image: Grayscale input image
        max_val: Value for pixels above threshold
    
    Returns:
        Tuple of (binary image, optimal threshold value)
    """
    if image.ndim != 2:
        raise ValueError("otsu_threshold expects a grayscale image")
    
    # Compute histogram
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    
    total_pixels = image.size
    hist_normalized = hist / total_pixels
    
    # Compute cumulative sums
    cumsum = np.cumsum(hist_normalized)
    cumsum_mean = np.cumsum(hist_normalized * np.arange(256))
    
    global_mean = cumsum_mean[-1]
    
    # Compute between-class variance for all thresholds
    # Avoid division by zero
    w0 = cumsum
    w1 = 1.0 - cumsum
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mu0 = cumsum_mean / np.maximum(w0, 1e-10)
        mu1 = (global_mean - cumsum_mean) / np.maximum(w1, 1e-10)
        
        between_variance = w0 * w1 * (mu0 - mu1) ** 2
    
    # Find threshold that maximizes between-class variance
    optimal_thresh = int(np.argmax(between_variance))
    
    binary = np.where(image >= optimal_thresh, max_val, 0).astype(np.uint8)
    
    return binary, optimal_thresh


def truncate_threshold(image: np.ndarray, thresh: int) -> np.ndarray:
    """
    Truncate threshold - pixels above threshold are set to threshold value.
    """
    if image.ndim != 2:
        raise ValueError("truncate_threshold expects a grayscale image")
    
    return np.minimum(image, thresh).astype(np.uint8)


def to_zero_threshold(image: np.ndarray, thresh: int) -> np.ndarray:
    """
    To-zero threshold - pixels below threshold become 0, others unchanged.
    """
    if image.ndim != 2:
        raise ValueError("to_zero_threshold expects a grayscale image")
    
    return np.where(image >= thresh, image, 0).astype(np.uint8)