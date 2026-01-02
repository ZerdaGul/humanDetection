import numpy as np


def binary_threshold(image: np.ndarray, thresh: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("binary_threshold expects a grayscale image")

    threshold_value = int(thresh)
    output = np.zeros_like(image, dtype=np.uint8)
    output[image >= threshold_value] = 255
    return output
