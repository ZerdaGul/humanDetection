from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

Neighbor = Tuple[int, int]


def connected_components(binary_image: np.ndarray) -> tuple[np.ndarray, List[Dict[str, Tuple[int, int] | int | float]]]:
    if binary_image.ndim != 2:
        raise ValueError("connected_components expects a single-channel image")

    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components: List[Dict[str, Tuple[int, int] | int | float]] = []

    mask = binary_image > 0
    label_id = 0
    neighbors: List[Neighbor] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue

            label_id += 1
            queue: deque[tuple[int, int]] = deque()
            queue.append((y, x))
            labels[y, x] = label_id

            min_y = max_y = y
            min_x = max_x = x
            sum_y = float(y)
            sum_x = float(x)
            area = 1

            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if not mask[ny, nx] or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = label_id
                    queue.append((ny, nx))

                    area += 1
                    sum_y += ny
                    sum_x += nx
                    min_y = min(min_y, ny)
                    min_x = min(min_x, nx)
                    max_y = max(max_y, ny)
                    max_x = max(max_x, nx)

            centroid = (sum_y / area, sum_x / area)
            components.append(
                {
                    "label": label_id,
                    "area": area,
                    "bbox": (min_y, min_x, max_y, max_x),
                    "centroid": centroid,
                }
            )

    return labels, components
