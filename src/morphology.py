import numpy as np


KERNEL_SHAPE = (3, 3)


def _apply_kernel(image: np.ndarray, operator: str) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    output = np.zeros_like(image, dtype=np.uint8)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y : y + KERNEL_SHAPE[0], x : x + KERNEL_SHAPE[1]]
            if operator == "erode":
                output[y, x] = 255 if np.all(region == 255) else 0
            elif operator == "dilate":
                output[y, x] = 255 if np.any(region == 255) else 0
            else:
                raise ValueError(f"Unsupported morphological operator: {operator}")

    return output


def erode(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = image.copy()
    for _ in range(max(1, int(iterations))):
        result = _apply_kernel(result, "erode")
    return result


def dilate(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = image.copy()
    for _ in range(max(1, int(iterations))):
        result = _apply_kernel(result, "dilate")
    return result
