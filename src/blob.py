import math
from typing import Dict, List, Sequence, Tuple

Blob = Dict[str, object]


def _aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
    min_y, min_x, max_y, max_x = bbox
    height = max(1, max_y - min_y + 1)
    width = max(1, max_x - min_x + 1)
    return width / height


def filter_blobs(
    components: Sequence[Blob],
    min_area: int,
    max_area: int,
    aspect_ratio_range: Tuple[float, float],
) -> List[Blob]:
    filtered: List[Blob] = []
    min_ratio, max_ratio = aspect_ratio_range

    for comp in components:
        area = int(comp["area"])
        if area < min_area or area > max_area:
            continue

        bbox = comp["bbox"]
        ratio = _aspect_ratio(bbox) if isinstance(bbox, tuple) else 1.0
        if ratio < min_ratio or ratio > max_ratio:
            continue

        filtered.append({
            "label": comp["label"],
            "area": area,
            "bbox": bbox,
            "centroid": comp["centroid"],
        })

    return filtered


class BlobTracker:
    """Maintains temporal consistency of detected blobs."""

    def __init__(self, max_distance: float = 60.0, max_disappeared: int = 10) -> None:
        self.max_distance = max(1.0, float(max_distance))
        self.max_disappeared = max(1, int(max_disappeared))
        self.next_id = 0
        self.tracks: Dict[int, Dict[str, object]] = {}

    def _register(self, blob: Blob) -> None:
        self.tracks[self.next_id] = {
            "id": self.next_id,
            "centroid": blob["centroid"],
            "bbox": blob["bbox"],
            "area": blob["area"],
            "history": [blob["centroid"]],
            "disappeared": 0,
        }
        self.next_id += 1

    def _deregister(self, track_id: int) -> None:
        if track_id in self.tracks:
            del self.tracks[track_id]

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dy = p1[0] - p2[0]
        dx = p1[1] - p2[1]
        return math.sqrt(dy * dy + dx * dx)

    def update(self, blobs: Sequence[Blob]) -> List[Dict[str, object]]:
        if len(self.tracks) == 0:
            for blob in blobs:
                self._register(blob)
            return list(self.tracks.values())

        assigned_tracks = set()
        assigned_detections = set()

        for det_idx, blob in enumerate(blobs):
            best_track_id = None
            best_distance = float("inf")
            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                distance = self._distance(track["centroid"], blob["centroid"])  # type: ignore[arg-type]
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None and best_distance <= self.max_distance:
                track = self.tracks[best_track_id]
                track["centroid"] = blob["centroid"]
                track["bbox"] = blob["bbox"]
                track["area"] = blob["area"]
                history = track["history"]  # type: ignore[assignment]
                history.append(blob["centroid"])  # type: ignore[arg-type]
                if len(history) > 32:
                    history.pop(0)
                track["disappeared"] = 0
                assigned_tracks.add(best_track_id)
                assigned_detections.add(det_idx)
            else:
                self._register(blob)
                assigned_detections.add(det_idx)

        unassigned_tracks = [tid for tid in self.tracks.keys() if tid not in assigned_tracks]
        for track_id in unassigned_tracks:
            track = self.tracks[track_id]
            track["disappeared"] = track.get("disappeared", 0) + 1  # type: ignore[operator]
            if track["disappeared"] >= self.max_disappeared:
                self._deregister(track_id)

        return list(self.tracks.values())
