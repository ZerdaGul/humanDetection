"""
Advanced Blob Tracking with Re-Identification.

Features:
1. Kalman Filter with adaptive noise
2. Hungarian Algorithm for optimal assignment  
3. Appearance-based Re-ID (histogram matching)
4. Multi-feature cost function (IoU + Distance + Size + Appearance + Velocity)
5. Track state machine with interpolation
6. Occlusion-aware prediction
7. Velocity consistency scoring
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np


# ============================================================================
# Type Definitions
# ============================================================================

class Blob(TypedDict):
    """Type definition for detected blob."""
    label: int
    area: int
    bbox: Tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    centroid: Tuple[float, float]    # (y, x)


class Track(TypedDict):
    """Type definition for tracked object."""
    id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    area: int
    history: List[Tuple[float, float]]
    disappeared: int
    velocity: Tuple[float, float]
    age: int
    state: str
    hits: int
    confidence: float


# ============================================================================
# Adaptive Kalman Filter
# ============================================================================

class AdaptiveKalmanFilter2D:
    """
    Adaptive Kalman Filter that adjusts noise based on prediction error.
    Better handles sudden direction changes and occlusions.
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        process_noise: float = 1.0,
        measurement_noise: float = 1.0
    ) -> None:
        # State: [y, x, vy, vx]
        self.state = np.array([
            initial_position[0],
            initial_position[1],
            0.0, 0.0
        ], dtype=np.float64)
        
        self.dt = 1.0
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        # Base noise values
        self.base_process_noise = process_noise
        self.base_measurement_noise = measurement_noise
        
        # Adaptive noise multipliers
        self.process_noise_scale = 1.0
        self.measurement_noise_scale = 1.0
        
        self._update_noise_matrices()
        
        # State covariance
        self.P = np.eye(4, dtype=np.float64) * 100.0
        
        # Innovation history for adaptation
        self.innovation_history: List[float] = []
        self.max_innovation_history = 10
    
    def _update_noise_matrices(self) -> None:
        """Update Q and R matrices based on current noise scales."""
        q = self.base_process_noise * self.process_noise_scale
        self.Q = np.array([
            [q/4, 0, q/2, 0],
            [0, q/4, 0, q/2],
            [q/2, 0, q, 0],
            [0, q/2, 0, q]
        ], dtype=np.float64)
        
        r = self.base_measurement_noise * self.measurement_noise_scale
        self.R = np.array([
            [r, 0],
            [0, r]
        ], dtype=np.float64)
    
    def predict(self) -> Tuple[float, float]:
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (self.state[0], self.state[1])
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Update with measurement and adapt noise."""
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)
        
        # Innovation
        y = z - self.H @ self.state
        innovation_magnitude = np.sqrt(y[0]**2 + y[1]**2)
        
        # Store innovation for adaptation
        self.innovation_history.append(innovation_magnitude)
        if len(self.innovation_history) > self.max_innovation_history:
            self.innovation_history.pop(0)
        
        # Adapt noise based on innovation
        self._adapt_noise(innovation_magnitude)
        
        # Standard Kalman update
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        self.state = self.state + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        
        return (self.state[0], self.state[1])
    
    def _adapt_noise(self, innovation: float) -> None:
        """Adapt process noise based on prediction error."""
        if len(self.innovation_history) < 3:
            return
        
        avg_innovation = np.mean(self.innovation_history)
        
        # High innovation = model not fitting well = increase process noise
        if innovation > avg_innovation * 2:
            self.process_noise_scale = min(10.0, self.process_noise_scale * 1.5)
        elif innovation < avg_innovation * 0.5:
            self.process_noise_scale = max(0.5, self.process_noise_scale * 0.9)
        
        self._update_noise_matrices()
    
    def get_position(self) -> Tuple[float, float]:
        return (self.state[0], self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        return (self.state[2], self.state[3])
    
    def get_speed(self) -> float:
        return math.sqrt(self.state[2]**2 + self.state[3]**2)
    
    def get_predicted_position(self, steps: int = 1) -> Tuple[float, float]:
        state = self.state.copy()
        for _ in range(steps):
            state = self.F @ state
        return (state[0], state[1])
    
    def boost_uncertainty(self, factor: float = 2.0) -> None:
        """Increase uncertainty during occlusion."""
        self.P *= factor
        self.process_noise_scale = min(10.0, self.process_noise_scale * factor)
        self._update_noise_matrices()


# ============================================================================
# Appearance Model for Re-ID
# ============================================================================

class AppearanceModel:
    """
    Simple appearance model using intensity histogram.
    Used for re-identification after occlusion.
    """
    
    def __init__(self, n_bins: int = 16, history_size: int = 10):
        self.n_bins = n_bins
        self.history_size = history_size
        self.histogram_history: List[np.ndarray] = []
        self.smooth_histogram: Optional[np.ndarray] = None
    
    def update(self, bbox: Tuple[int, int, int, int], frame: Optional[np.ndarray]) -> None:
        """Update appearance model with new observation."""
        if frame is None:
            return
        
        min_y, min_x, max_y, max_x = bbox
        
        # Ensure valid bounds
        h, w = frame.shape[:2]
        min_y = max(0, min(min_y, h - 1))
        max_y = max(0, min(max_y, h))
        min_x = max(0, min(min_x, w - 1))
        max_x = max(0, min(max_x, w))
        
        if max_y <= min_y or max_x <= min_x:
            return
        
        # Extract region
        region = frame[min_y:max_y, min_x:max_x]
        if region.size == 0:
            return
        
        # Convert to grayscale if needed
        if region.ndim == 3:
            region = np.mean(region, axis=2)
        
        # Compute histogram
        hist, _ = np.histogram(region.ravel(), bins=self.n_bins, range=(0, 256))
        hist = hist.astype(np.float64)
        
        # Normalize
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        
        # Add to history
        self.histogram_history.append(hist)
        if len(self.histogram_history) > self.history_size:
            self.histogram_history.pop(0)
        
        # Update smooth histogram (exponential moving average)
        if self.smooth_histogram is None:
            self.smooth_histogram = hist.copy()
        else:
            alpha = 0.3
            self.smooth_histogram = alpha * hist + (1 - alpha) * self.smooth_histogram
    
    def compute_similarity(self, other: 'AppearanceModel') -> float:
        """Compute histogram similarity with another appearance model."""
        if self.smooth_histogram is None or other.smooth_histogram is None:
            return 0.5  # Neutral score if no histogram
        
        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(self.smooth_histogram * other.smooth_histogram))
        return float(bc)
    
    def compute_similarity_to_histogram(self, hist: np.ndarray) -> float:
        """Compute similarity to a given histogram."""
        if self.smooth_histogram is None:
            return 0.5
        
        # Normalize input
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        
        bc = np.sum(np.sqrt(self.smooth_histogram * hist))
        return float(bc)
    
    def get_histogram(self) -> Optional[np.ndarray]:
        return self.smooth_histogram.copy() if self.smooth_histogram is not None else None


def compute_blob_histogram(
    bbox: Tuple[int, int, int, int], 
    frame: Optional[np.ndarray],
    n_bins: int = 16
) -> Optional[np.ndarray]:
    """Compute histogram for a blob region."""
    if frame is None:
        return None
    
    min_y, min_x, max_y, max_x = bbox
    h, w = frame.shape[:2]
    
    min_y = max(0, min(min_y, h - 1))
    max_y = max(0, min(max_y, h))
    min_x = max(0, min(min_x, w - 1))
    max_x = max(0, min(max_x, w))
    
    if max_y <= min_y or max_x <= min_x:
        return None
    
    region = frame[min_y:max_y, min_x:max_x]
    if region.size == 0:
        return None
    
    if region.ndim == 3:
        region = np.mean(region, axis=2)
    
    hist, _ = np.histogram(region.ravel(), bins=n_bins, range=(0, 256))
    return hist.astype(np.float64)


# ============================================================================
# Hungarian Algorithm
# ============================================================================

def hungarian_algorithm(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Hungarian algorithm for optimal assignment."""
    if cost_matrix.size == 0:
        return []
    
    n_rows, n_cols = cost_matrix.shape
    size = max(n_rows, n_cols)
    padded = np.full((size, size), 1e9, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost_matrix
    
    for i in range(size):
        min_val = np.min(padded[i])
        if min_val < 1e8:
            padded[i] -= min_val
    
    for j in range(size):
        min_val = np.min(padded[:, j])
        if min_val < 1e8:
            padded[:, j] -= min_val
    
    assignments = _hungarian_augment(padded)
    
    result = []
    for row, col in assignments:
        if row < n_rows and col < n_cols:
            if cost_matrix[row, col] < 1e8:
                result.append((row, col))
    
    return result


def _hungarian_augment(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Augmenting path method."""
    n = cost.shape[0]
    INF = 1e9
    
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int32)
    way = np.zeros(n + 1, dtype=np.int32)
    
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        
        while p[j0] != 0:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            
            for j in range(1, n + 1):
                if not used[j]:
                    if i0 <= n and j <= n:
                        cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    else:
                        cur = INF
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
        
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    
    return [(p[j] - 1, j - 1) for j in range(1, n + 1) if p[j] != 0]


# ============================================================================
# Utility Functions
# ============================================================================

def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two bounding boxes."""
    min_y1, min_x1, max_y1, max_x1 = bbox1
    min_y2, min_x2, max_y2, max_x2 = bbox2
    
    inter_min_y = max(min_y1, min_y2)
    inter_min_x = max(min_x1, min_x2)
    inter_max_y = min(max_y1, max_y2)
    inter_max_x = min(max_x1, max_x2)
    
    if inter_max_y <= inter_min_y or inter_max_x <= inter_min_x:
        return 0.0
    
    inter_area = (inter_max_y - inter_min_y) * (inter_max_x - inter_min_x)
    area1 = (max_y1 - min_y1) * (max_x1 - min_x1)
    area2 = (max_y2 - min_y2) * (max_x2 - min_x2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_centroid_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Compute Euclidean distance."""
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def compute_size_similarity(area1: int, area2: int) -> float:
    """Compute size similarity (0 to 1)."""
    if area1 <= 0 or area2 <= 0:
        return 0.0
    return min(area1, area2) / max(area1, area2)


def compute_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
    """Compute width/height ratio."""
    min_y, min_x, max_y, max_x = bbox
    height = max(1, max_y - min_y)
    width = max(1, max_x - min_x)
    return width / height


def compute_velocity_consistency(
    predicted_pos: Tuple[float, float],
    actual_pos: Tuple[float, float],
    velocity: Tuple[float, float]
) -> float:
    """
    Compute how consistent the actual position is with predicted velocity.
    Returns 0 (inconsistent) to 1 (consistent).
    """
    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
    if speed < 0.5:  # Nearly stationary
        return 1.0
    
    error = compute_centroid_distance(predicted_pos, actual_pos)
    max_error = speed * 3  # Allow up to 3x speed as error
    
    consistency = 1.0 - min(1.0, error / max(1.0, max_error))
    return consistency


def compute_extent(area: int, bbox: Tuple[int, int, int, int]) -> float:
    """Compute fill ratio."""
    min_y, min_x, max_y, max_x = bbox
    bbox_area = max(1, (max_y - min_y + 1) * (max_x - min_x + 1))
    return area / bbox_area


# ============================================================================
# Track State
# ============================================================================

class TrackState:
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


# ============================================================================
# Enhanced Track with Re-ID
# ============================================================================

class EnhancedTrack:
    """Track with Kalman filter and appearance model."""
    
    def __init__(
        self,
        track_id: int,
        blob: Blob,
        frame: Optional[np.ndarray] = None,
        process_noise: float = 4.0,
        measurement_noise: float = 1.0
    ) -> None:
        self.id = track_id
        self.bbox = blob["bbox"]
        self.area = blob["area"]
        
        # Adaptive Kalman filter
        self.kf = AdaptiveKalmanFilter2D(
            initial_position=blob["centroid"],
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        
        # Appearance model for Re-ID
        self.appearance = AppearanceModel(n_bins=16, history_size=10)
        self.appearance.update(blob["bbox"], frame)
        
        # History
        self.history: List[Tuple[float, float]] = [blob["centroid"]]
        self.bbox_history: List[Tuple[int, int, int, int]] = [blob["bbox"]]
        self.velocity_history: List[Tuple[float, float]] = [(0.0, 0.0)]
        
        # State
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0
        self.confidence = 0.5
        
        # Prediction tracking
        self.last_predicted_pos: Optional[Tuple[float, float]] = None
        self.prediction_errors: List[float] = []
    
    @property
    def centroid(self) -> Tuple[float, float]:
        return self.kf.get_position()
    
    @property
    def velocity(self) -> Tuple[float, float]:
        return self.kf.get_velocity()
    
    @property
    def speed(self) -> float:
        return self.kf.get_speed()
    
    def predict(self) -> Tuple[float, float]:
        """Predict next position."""
        self.last_predicted_pos = self.kf.get_position()
        predicted_pos = self.kf.predict()
        
        self.age += 1
        self.time_since_update += 1
        
        # Decay confidence
        decay_rate = 0.9 if self.state == TrackState.CONFIRMED else 0.8
        self.confidence *= decay_rate
        
        # Update bbox prediction
        vy, vx = self.velocity
        min_y, min_x, max_y, max_x = self.bbox
        self.bbox = (
            int(min_y + vy),
            int(min_x + vx),
            int(max_y + vy),
            int(max_x + vx)
        )
        
        # Increase uncertainty during occlusion
        if self.time_since_update > 2:
            self.kf.boost_uncertainty(1.2)
        
        return predicted_pos
    
    def update(self, blob: Blob, frame: Optional[np.ndarray] = None) -> None:
        """Update track with matched detection."""
        # Track prediction error
        if self.last_predicted_pos is not None:
            error = compute_centroid_distance(self.last_predicted_pos, blob["centroid"])
            self.prediction_errors.append(error)
            if len(self.prediction_errors) > 20:
                self.prediction_errors.pop(0)
        
        # Update Kalman
        self.kf.update(blob["centroid"])
        
        # Update appearance
        self.appearance.update(blob["bbox"], frame)
        
        # Update state
        self.bbox = blob["bbox"]
        self.area = blob["area"]
        
        # Update history
        self.history.append(blob["centroid"])
        self.bbox_history.append(blob["bbox"])
        self.velocity_history.append(self.velocity)
        
        max_history = 50
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]
            self.bbox_history = self.bbox_history[-max_history:]
            self.velocity_history = self.velocity_history[-max_history:]
        
        # Update counters
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        # Update confidence based on consistency
        if self.prediction_errors:
            avg_error = np.mean(self.prediction_errors[-5:])
            if avg_error < 10:
                self.confidence = min(1.0, self.confidence + 0.15)
            elif avg_error < 30:
                self.confidence = min(1.0, self.confidence + 0.1)
            else:
                self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.confidence = min(1.0, self.confidence + 0.1)
        
        # State transitions
        if self.state == TrackState.TENTATIVE and self.hit_streak >= 3:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED
            self.hit_streak = 1
    
    def mark_missed(self) -> None:
        """Mark as missed."""
        self.hit_streak = 0
        if self.state == TrackState.CONFIRMED and self.time_since_update >= 5:
            self.state = TrackState.LOST
    
    def is_deleted(self, max_age: int = 30) -> bool:
        """Check if should be deleted."""
        if self.state == TrackState.TENTATIVE and self.time_since_update >= 3:
            return True
        if self.time_since_update >= max_age:
            return True
        return False
    
    def get_predicted_bbox(self) -> Tuple[int, int, int, int]:
        """Get predicted bbox."""
        pred_y, pred_x = self.kf.get_predicted_position(1)
        cy, cx = self.centroid
        dy, dx = pred_y - cy, pred_x - cx
        min_y, min_x, max_y, max_x = self.bbox
        return (int(min_y + dy), int(min_x + dx), int(max_y + dy), int(max_x + dx))
    
    def get_average_velocity(self, n: int = 5) -> Tuple[float, float]:
        """Get average velocity over last n frames."""
        if len(self.velocity_history) < 2:
            return (0.0, 0.0)
        recent = self.velocity_history[-n:]
        avg_vy = np.mean([v[0] for v in recent])
        avg_vx = np.mean([v[1] for v in recent])
        return (avg_vy, avg_vx)
    
    def get_motion_direction(self) -> float:
        """Get motion direction in radians."""
        vy, vx = self.velocity
        return math.atan2(vy, vx)
    
    def to_dict(self) -> Track:
        return {
            "id": self.id,
            "centroid": self.centroid,
            "bbox": self.bbox,
            "area": self.area,
            "history": self.history.copy(),
            "disappeared": self.time_since_update,
            "velocity": self.velocity,
            "age": self.age,
            "state": self.state,
            "hits": self.hits,
            "confidence": self.confidence
        }


# ============================================================================
# Advanced Blob Tracker
# ============================================================================

class BlobTracker:
    """
    Advanced tracker with Re-ID capability.
    """
    
    def __init__(
        self,
        max_distance: float = 100.0,
        max_iou_distance: float = 0.7,
        max_disappeared: int = 30,
        min_hits: int = 3,
        iou_weight: float = 0.25,
        distance_weight: float = 0.30,
        size_weight: float = 0.15,
        appearance_weight: float = 0.20,
        velocity_weight: float = 0.10,
        process_noise: float = 4.0,
        measurement_noise: float = 1.0
    ) -> None:
        self.max_distance = max(1.0, float(max_distance))
        self.max_iou_distance = max(0.1, min(1.0, float(max_iou_distance)))
        self.max_age = max(1, int(max_disappeared))
        self.min_hits = max(1, int(min_hits))
        
        # Normalize weights
        total = iou_weight + distance_weight + size_weight + appearance_weight + velocity_weight
        self.iou_weight = iou_weight / total
        self.distance_weight = distance_weight / total
        self.size_weight = size_weight / total
        self.appearance_weight = appearance_weight / total
        self.velocity_weight = velocity_weight / total
        
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        self.tracks: Dict[int, EnhancedTrack] = {}
        self.lost_tracks: Dict[int, EnhancedTrack] = {}  # For Re-ID
        self.next_id = 0
        self.frame_count = 0
        
        # Current frame reference
        self.current_frame: Optional[np.ndarray] = None
    
    def _create_track(self, blob: Blob) -> EnhancedTrack:
        """Create new track."""
        track = EnhancedTrack(
            track_id=self.next_id,
            blob=blob,
            frame=self.current_frame,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track
    
    def _delete_track(self, track_id: int) -> None:
        """Move track to lost tracks for potential Re-ID."""
        if track_id in self.tracks:
            track = self.tracks.pop(track_id)
            # Keep in lost tracks if it was confirmed
            if track.state == TrackState.CONFIRMED and track.hits >= 5:
                self.lost_tracks[track_id] = track
                # Limit lost tracks
                if len(self.lost_tracks) > 20:
                    oldest_id = min(self.lost_tracks.keys())
                    del self.lost_tracks[oldest_id]
    
    def _try_reid(self, blob: Blob) -> Optional[int]:
        """Try to re-identify blob with lost track."""
        if not self.lost_tracks:
            return None
        
        blob_hist = compute_blob_histogram(blob["bbox"], self.current_frame)
        if blob_hist is None:
            return None
        
        best_match_id = None
        best_score = 0.6  # Minimum similarity threshold
        
        for track_id, track in self.lost_tracks.items():
            # Check distance is reasonable
            dist = compute_centroid_distance(track.centroid, blob["centroid"])
            max_reid_dist = self.max_distance * 3  # Allow larger distance for re-id
            
            if dist > max_reid_dist:
                continue
            
            # Check appearance similarity
            similarity = track.appearance.compute_similarity_to_histogram(blob_hist)
            
            # Check size similarity
            size_sim = compute_size_similarity(track.area, blob["area"])
            
            # Combined score
            score = similarity * 0.7 + size_sim * 0.3
            
            if score > best_score:
                best_score = score
                best_match_id = track_id
        
        return best_match_id
    
    def _compute_cost(
        self,
        track: EnhancedTrack,
        blob: Blob,
        blob_hist: Optional[np.ndarray]
    ) -> float:
        """Compute matching cost between track and blob."""
        # IoU cost
        pred_bbox = track.get_predicted_bbox()
        iou = compute_iou(pred_bbox, blob["bbox"])
        iou_cost = 1.0 - iou
        
        # Distance cost
        dist = compute_centroid_distance(track.centroid, blob["centroid"])
        if dist > self.max_distance:
            return 1e9
        dist_cost = dist / self.max_distance
        
        # Size cost
        size_sim = compute_size_similarity(track.area, blob["area"])
        size_cost = 1.0 - size_sim
        
        # Appearance cost
        if blob_hist is not None:
            appearance_sim = track.appearance.compute_similarity_to_histogram(blob_hist)
        else:
            appearance_sim = 0.5
        appearance_cost = 1.0 - appearance_sim
        
        # Velocity consistency cost
        pred_pos = track.kf.get_predicted_position(1)
        vel_consistency = compute_velocity_consistency(
            pred_pos, blob["centroid"], track.velocity
        )
        velocity_cost = 1.0 - vel_consistency
        
        # Combined cost
        cost = (
            self.iou_weight * iou_cost +
            self.distance_weight * dist_cost +
            self.size_weight * size_cost +
            self.appearance_weight * appearance_cost +
            self.velocity_weight * velocity_cost
        )
        
        return cost
    
    def _cascade_matching(
        self,
        tracks: List[EnhancedTrack],
        blobs: List[Blob],
        blob_histograms: List[Optional[np.ndarray]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Cascade matching with appearance."""
        if len(tracks) == 0:
            return [], [], list(range(len(blobs)))
        if len(blobs) == 0:
            return [], list(range(len(tracks))), []
        
        # Sort by recency
        track_indices = sorted(range(len(tracks)), key=lambda i: tracks[i].time_since_update)
        
        matches = []
        unmatched_blobs = set(range(len(blobs)))
        unmatched_tracks = set(range(len(tracks)))
        
        max_cascade = min(self.max_age, 15)
        
        for cascade_level in range(max_cascade):
            if not unmatched_blobs:
                break
            
            level_tracks = [
                i for i in track_indices
                if i in unmatched_tracks and tracks[i].time_since_update == cascade_level
            ]
            
            if not level_tracks:
                continue
            
            level_blobs = list(unmatched_blobs)
            cost_matrix = np.full((len(level_tracks), len(level_blobs)), 1e9, dtype=np.float64)
            
            for ii, track_idx in enumerate(level_tracks):
                track = tracks[track_idx]
                for jj, blob_idx in enumerate(level_blobs):
                    blob = blobs[blob_idx]
                    blob_hist = blob_histograms[blob_idx]
                    cost_matrix[ii, jj] = self._compute_cost(track, blob, blob_hist)
            
            level_matches = hungarian_algorithm(cost_matrix)
            
            # Adaptive threshold based on cascade level
            threshold = 0.7 + cascade_level * 0.05  # Relax threshold for older tracks
            
            for ii, jj in level_matches:
                if cost_matrix[ii, jj] < threshold:
                    track_idx = level_tracks[ii]
                    blob_idx = level_blobs[jj]
                    matches.append((track_idx, blob_idx))
                    unmatched_tracks.discard(track_idx)
                    unmatched_blobs.discard(blob_idx)
        
        return matches, list(unmatched_tracks), list(unmatched_blobs)
    
    def update(
        self,
        blobs: Sequence[Blob],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """Update tracker."""
        self.frame_count += 1
        self.current_frame = frame
        blobs = list(blobs)
        
        # Precompute blob histograms
        blob_histograms = [
            compute_blob_histogram(b["bbox"], frame) for b in blobs
        ]
        
        # Predict all tracks
        track_list = list(self.tracks.values())
        for track in track_list:
            track.predict()
        
        # Cascade matching
        matches, unmatched_track_indices, unmatched_blob_indices = \
            self._cascade_matching(track_list, blobs, blob_histograms)
        
        # Update matched tracks
        for track_idx, blob_idx in matches:
            track_list[track_idx].update(blobs[blob_idx], frame)
        
        # Mark unmatched tracks
        for track_idx in unmatched_track_indices:
            track_list[track_idx].mark_missed()
        
        # Handle unmatched blobs
        for blob_idx in unmatched_blob_indices:
            blob = blobs[blob_idx]
            
            # Try Re-ID first
            reid_id = self._try_reid(blob)
            if reid_id is not None and reid_id in self.lost_tracks:
                # Recover lost track
                track = self.lost_tracks.pop(reid_id)
                track.update(blob, frame)
                self.tracks[track.id] = track
            else:
                # Create new track
                self._create_track(blob)
        
        # Delete old tracks
        to_delete = [tid for tid, t in self.tracks.items() if t.is_deleted(self.max_age)]
        for tid in to_delete:
            self._delete_track(tid)
        
        # Clean old lost tracks
        lost_to_delete = [
            tid for tid, t in self.lost_tracks.items()
            if t.time_since_update > self.max_age * 2
        ]
        for tid in lost_to_delete:
            del self.lost_tracks[tid]
        
        return [t.to_dict() for t in self.tracks.values()]
    
    def get_confirmed_tracks(self) -> List[Track]:
        return [t.to_dict() for t in self.tracks.values() if t.state == TrackState.CONFIRMED]
    
    def get_active_tracks(self) -> List[Track]:
        return [
            t.to_dict() for t in self.tracks.values()
            if t.state == TrackState.CONFIRMED or (t.state == TrackState.TENTATIVE and t.hit_streak >= 2)
        ]
    
    def reset(self) -> None:
        self.tracks.clear()
        self.lost_tracks.clear()
        self.next_id = 0
        self.frame_count = 0
        self.current_frame = None


# ============================================================================
# Blob Filtering
# ============================================================================

def filter_blobs(
    components: Sequence[Dict],
    min_area: int = 500,
    max_area: int = 50000,
    aspect_ratio_range: Tuple[float, float] = (0.2, 4.0),
    min_extent: float = 0.0
) -> List[Blob]:
    """Filter blobs by geometry."""
    min_ratio, max_ratio = aspect_ratio_range
    filtered: List[Blob] = []
    
    for comp in components:
        area = int(comp["area"])
        if area < min_area or area > max_area:
            continue
        
        bbox = comp["bbox"]
        if not isinstance(bbox, tuple):
            continue
        
        ratio = compute_aspect_ratio(bbox)
        if ratio < min_ratio or ratio > max_ratio:
            continue
        
        if min_extent > 0:
            extent = compute_extent(area, bbox)
            if extent < min_extent:
                continue
        
        filtered.append({
            "label": comp["label"],
            "area": area,
            "bbox": bbox,
            "centroid": comp["centroid"]
        })
    
    return filtered