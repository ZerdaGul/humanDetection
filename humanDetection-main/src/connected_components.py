"""
Connected component analysis for binary images.

Implements two-pass labeling algorithm with union-find for efficiency.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict

import numpy as np


class ComponentInfo(TypedDict):
    """Type definition for component information."""
    label: int
    area: int
    bbox: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    centroid: Tuple[float, float]    # (y, x)


class UnionFind:
    """
    Disjoint set data structure for efficient label merging.
    
    Uses path compression and union by rank for nearly O(1) operations.
    """
    
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def connected_components(
    binary_image: np.ndarray,
    connectivity: int = 8
) -> Tuple[np.ndarray, List[ComponentInfo]]:
    """
    Label connected components in a binary image.
    
    Uses two-pass algorithm with union-find for efficient labeling.
    
    Args:
        binary_image: Binary input image (0 = background, >0 = foreground)
        connectivity: 4 or 8 connectivity
    
    Returns:
        Tuple of (label_image, component_list)
        - label_image: 2D array where each pixel has its component label
        - component_list: List of dicts with label, area, bbox, centroid
    
    Raises:
        ValueError: If image is not 2D or connectivity is invalid
    """
    if binary_image.ndim != 2:
        raise ValueError("connected_components expects a single-channel image")
    
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    
    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=np.int32)
    
    # Define neighbor offsets based on connectivity
    if connectivity == 8:
        # Check top-left, top, top-right, left
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    else:
        # Check top, left only
        neighbors = [(-1, 0), (0, -1)]
    
    mask = binary_image > 0
    uf = UnionFind(height * width // 4 + 1)  # Rough estimate of max labels
    
    next_label = 1
    
    # First pass: initial labeling with equivalence recording
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            
            # Collect labels from neighbors
            neighbor_labels = []
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if labels[ny, nx] > 0:
                        neighbor_labels.append(labels[ny, nx])
            
            if not neighbor_labels:
                # New component
                labels[y, x] = next_label
                # Ensure union-find has enough capacity
                if next_label >= len(uf.parent):
                    uf.parent.extend(range(len(uf.parent), next_label + 100))
                    uf.rank.extend([0] * 100)
                next_label += 1
            else:
                # Assign minimum label
                min_label = min(neighbor_labels)
                labels[y, x] = min_label
                
                # Record equivalences
                for lbl in neighbor_labels:
                    if lbl != min_label:
                        uf.union(min_label, lbl)
    
    # Second pass: resolve equivalences and collect statistics
    # Map old labels to new consecutive labels
    label_map = {}
    new_label = 0
    
    # Component statistics
    stats: Dict[int, Dict] = {}
    
    for y in range(height):
        for x in range(width):
            old_label = labels[y, x]
            if old_label == 0:
                continue
            
            # Find root label
            root = uf.find(old_label)
            
            if root not in label_map:
                new_label += 1
                label_map[root] = new_label
                stats[new_label] = {
                    "min_y": y, "max_y": y,
                    "min_x": x, "max_x": x,
                    "sum_y": 0.0, "sum_x": 0.0,
                    "area": 0
                }
            
            final_label = label_map[root]
            labels[y, x] = final_label
            
            # Update statistics
            s = stats[final_label]
            s["min_y"] = min(s["min_y"], y)
            s["max_y"] = max(s["max_y"], y)
            s["min_x"] = min(s["min_x"], x)
            s["max_x"] = max(s["max_x"], x)
            s["sum_y"] += y
            s["sum_x"] += x
            s["area"] += 1
    
    # Build component list
    components: List[ComponentInfo] = []
    for label_id in sorted(stats.keys()):
        s = stats[label_id]
        area = s["area"]
        components.append({
            "label": label_id,
            "area": area,
            "bbox": (s["min_y"], s["min_x"], s["max_y"], s["max_x"]),
            "centroid": (s["sum_y"] / area, s["sum_x"] / area)
        })
    
    return labels, components


def connected_components_with_stats(
    binary_image: np.ndarray,
    connectivity: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OpenCV-compatible interface returning stats and centroids as arrays.
    
    Returns:
        Tuple of (labels, stats, centroids)
        - labels: Label image
        - stats: Nx5 array [x, y, width, height, area] per component
        - centroids: Nx2 array [cx, cy] per component
    """
    labels, components = connected_components(binary_image, connectivity)
    
    n_labels = len(components) + 1  # +1 for background
    
    # Stats array: [x, y, width, height, area]
    stats = np.zeros((n_labels, 5), dtype=np.int32)
    centroids = np.zeros((n_labels, 2), dtype=np.float64)
    
    # Background (label 0)
    bg_mask = labels == 0
    bg_area = np.sum(bg_mask)
    if bg_area > 0:
        bg_y, bg_x = np.where(bg_mask)
        stats[0] = [bg_x.min(), bg_y.min(), 
                    bg_x.max() - bg_x.min() + 1,
                    bg_y.max() - bg_y.min() + 1,
                    bg_area]
        centroids[0] = [bg_x.mean(), bg_y.mean()]
    
    # Foreground components
    for comp in components:
        label = comp["label"]
        min_y, min_x, max_y, max_x = comp["bbox"]
        stats[label] = [
            min_x, min_y,
            max_x - min_x + 1,
            max_y - min_y + 1,
            comp["area"]
        ]
        cy, cx = comp["centroid"]
        centroids[label] = [cx, cy]
    
    return labels, stats, centroids


def filter_small_components(
    binary_image: np.ndarray,
    min_area: int,
    connectivity: int = 8
) -> np.ndarray:
    """
    Remove connected components smaller than min_area.
    
    Useful for noise removal after thresholding.
    
    Args:
        binary_image: Binary input image
        min_area: Minimum area to keep
        connectivity: 4 or 8 connectivity
    
    Returns:
        Binary image with small components removed
    """
    labels, components = connected_components(binary_image, connectivity)
    
    # Create mask of labels to keep
    output = np.zeros_like(binary_image)
    for comp in components:
        if comp["area"] >= min_area:
            output[labels == comp["label"]] = 255
    
    return output


def largest_component(
    binary_image: np.ndarray,
    connectivity: int = 8
) -> np.ndarray:
    """
    Keep only the largest connected component.
    
    Args:
        binary_image: Binary input image
        connectivity: 4 or 8 connectivity
    
    Returns:
        Binary image with only the largest component
    """
    labels, components = connected_components(binary_image, connectivity)
    
    if not components:
        return np.zeros_like(binary_image)
    
    # Find largest
    largest = max(components, key=lambda c: c["area"])
    
    output = np.zeros_like(binary_image)
    output[labels == largest["label"]] = 255
    
    return output