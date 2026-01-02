import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


KERNEL_SHAPE = (3, 3)


def _sliding_windows(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    return sliding_window_view(padded, KERNEL_SHAPE)


def _apply(image: np.ndarray, mode: str) -> np.ndarray:
    windows = _sliding_windows(image)
    if mode == "erode":
        result = np.all(windows == 255, axis=(-2, -1))
    elif mode == "dilate":
        result = np.any(windows == 255, axis=(-2, -1))
    else:
        raise ValueError(f"Unsupported morphological operator: {mode}")
    return (result.astype(np.uint8)) * 255


def erode(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = image
    for _ in range(max(1, int(iterations))):
        result = _apply(result, "erode")
    return result


def dilate(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = image
    for _ in range(max(1, int(iterations))):
        result = _apply(result, "dilate")
    return result


def opening(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = image
    for _ in range(max(1, int(iterations))):
        result = dilate(erode(result, 1), 1)
    return result


def closing(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = image
    for _ in range(max(1, int(iterations))):
        result = erode(dilate(result, 1), 1)
    return result
