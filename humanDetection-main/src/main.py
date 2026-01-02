"""
Human detection pipeline with multi-window visualization.

Her işlem aşaması ayrı pencerede gösterilir:
1. Original - Orijinal frame
2. Grayscale - Gri tonlama
3. Gaussian Blur - Bulanıklaştırma
4. Foreground - Arka plan çıkarma
5. Binary Threshold - Eşikleme
6. Morphology - Morfolojik işlemler
7. Detections - Son sonuç + tracking
8. Info Panel - İstatistikler
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from background import RunningAverageBackground, RunningAverageWithVariance
from blob import BlobTracker, filter_blobs, Track
from connected_components import connected_components
from gaussian import apply_gaussian_blur
from morphology import closing, dilate, erode, opening
from threshold import binary_threshold
from utils import load_settings, resize_frame_fast


# ============================================================================
# Configuration Helpers
# ============================================================================

def compute_adaptive_threshold(foreground: np.ndarray, cfg: Dict[str, Any]) -> int:
    """Compute threshold value, optionally using adaptive method."""
    base_value = int(cfg.get("value", 30))
    
    if cfg.get("adaptive", False):
        mean_val = float(np.mean(foreground))
        std_val = float(np.std(foreground))
        std_factor = cfg.get("std_factor", 1.5)
        offset = cfg.get("offset", 5.0)
        adaptive_value = mean_val + std_factor * std_val + offset
        return max(base_value, int(adaptive_value))
    
    return base_value


# ============================================================================
# Visualization Helpers
# ============================================================================

def add_label(image: np.ndarray, text: str) -> np.ndarray:
    """Add text label to image top-left corner."""
    if image.ndim == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display = image.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(display, (0, 0), (text_w + 10, text_h + 10), (0, 0, 0), -1)
    cv2.putText(display, text, (5, text_h + 5), font, font_scale, (0, 255, 0), thickness)
    
    return display


def draw_detections(
    frame: np.ndarray,
    tracks: List[Track],
    draw_trails: bool = True,
    show_state: bool = True
) -> np.ndarray:
    """Draw bounding boxes, IDs, and trails on frame."""
    display = frame.copy()
    
    for track in tracks:
        bbox = track.get("bbox", (0, 0, 0, 0))
        min_y, min_x, max_y, max_x = [int(v) for v in bbox]
        
        state = track.get("state", "confirmed")
        confidence = track.get("confidence", 1.0)
        
        # Color based on state
        if state == "confirmed":
            color = (0, 255, 0)  # Green - confirmed
        elif state == "tentative":
            color = (0, 255, 255)  # Yellow - tentative
        else:
            color = (128, 128, 128)  # Gray - lost
        
        # Draw bounding box
        thickness = 2 if state == "confirmed" else 1
        cv2.rectangle(display, (min_x, min_y), (max_x, max_y), color, thickness)
        
        # Draw centroid
        centroid = track.get("centroid", (0.0, 0.0))
        cx, cy = int(centroid[1]), int(centroid[0])
        cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
        
        # Draw velocity vector
        velocity = track.get("velocity", (0.0, 0.0))
        vy, vx = velocity
        if abs(vx) > 0.5 or abs(vy) > 0.5:
            end_x = int(cx + vx * 5)
            end_y = int(cy + vy * 5)
            cv2.arrowedLine(display, (cx, cy), (end_x, end_y), (255, 0, 255), 2)
        
        # Label with ID, state, and confidence
        if show_state:
            label = f"ID:{track['id']} {state[0].upper()} {confidence:.0%}"
        else:
            label = f"ID {track['id']}"
        
        cv2.putText(
            display, label, (min_x, max(min_y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
        )
        
        # Draw trail
        if draw_trails:
            history = track.get("history", [])
            if len(history) >= 2:
                points = [(int(p[1]), int(p[0])) for p in history]
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    thickness = max(1, int(alpha * 3))
                    # Gradient color
                    b = int(255 * (1 - alpha))
                    g = int(128 * alpha)
                    r = int(255 * alpha)
                    cv2.line(display, points[i-1], points[i], (b, g, r), thickness)
    
    return display


def create_info_panel(
    width: int,
    height: int,
    fps: float,
    thresh_value: int,
    n_blobs: int,
    n_tracks: int,
    n_confirmed: int,
    frame_count: int
) -> np.ndarray:
    """Create info panel with statistics."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    cyan = (255, 255, 0)
    
    y = 25
    line_h = 20
    
    cv2.putText(panel, "=== PIPELINE INFO ===", (10, y), font, 0.45, yellow, 1)
    y += line_h + 5
    
    cv2.putText(panel, f"Frame: {frame_count}", (10, y), font, 0.4, white, 1)
    y += line_h
    cv2.putText(panel, f"FPS: {fps:.1f}", (10, y), font, 0.4, green, 1)
    y += line_h
    cv2.putText(panel, f"Threshold: {thresh_value}", (10, y), font, 0.4, white, 1)
    y += line_h + 5
    
    cv2.putText(panel, "=== TRACKING ===", (10, y), font, 0.45, yellow, 1)
    y += line_h + 5
    
    cv2.putText(panel, f"Blobs: {n_blobs}", (10, y), font, 0.4, white, 1)
    y += line_h
    cv2.putText(panel, f"Total Tracks: {n_tracks}", (10, y), font, 0.4, white, 1)
    y += line_h
    cv2.putText(panel, f"Confirmed: {n_confirmed}", (10, y), font, 0.4, green, 1)
    y += line_h + 10
    
    cv2.putText(panel, "=== CONTROLS ===", (10, y), font, 0.45, yellow, 1)
    y += line_h + 5
    
    controls = [
        "Q - Quit",
        "R - Reset tracker",
        "T - Toggle trails",
        "S - Screenshot",
        "+/- Threshold"
    ]
    for ctrl in controls:
        cv2.putText(panel, ctrl, (10, y), font, 0.35, white, 1)
        y += line_h - 3
    
    return panel


# ============================================================================
# Multi-Window Display Manager  
# ============================================================================

class MultiWindowDisplay:
    """Manages multiple OpenCV windows."""
    
    def __init__(self, window_size: Tuple[int, int] = (320, 240)):
        self.window_size = window_size
        self.windows = [
            "1-Original",
            "2-Grayscale", 
            "3-GaussianBlur",
            "4-Foreground",
            "5-Threshold",
            "6-Morphology",
            "7-Detections",
            "8-Info"
        ]
        self._setup_windows()
    
    def _setup_windows(self):
        """Create and position windows in 4x2 grid."""
        w, h = self.window_size
        gap = 5
        title_bar = 30
        
        positions = [
            (0, 0),                          (w + gap, 0),
            (2 * (w + gap), 0),              (3 * (w + gap), 0),
            (0, h + gap + title_bar),        (w + gap, h + gap + title_bar),
            (2 * (w + gap), h + gap + title_bar), (3 * (w + gap), h + gap + title_bar)
        ]
        
        for i, name in enumerate(self.windows):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, w, h)
            if i < len(positions):
                cv2.moveWindow(name, positions[i][0], positions[i][1])
    
    def update(self, images: Dict[str, np.ndarray]):
        """Update all windows."""
        cv2.imshow(self.windows[0], add_label(images["original"], "Original"))
        cv2.imshow(self.windows[1], add_label(images["grayscale"], "Grayscale"))
        cv2.imshow(self.windows[2], add_label(images["blurred"], "Gaussian Blur"))
        cv2.imshow(self.windows[3], add_label(images["foreground"], "Foreground"))
        cv2.imshow(self.windows[4], add_label(images["binary"], "Binary Threshold"))
        cv2.imshow(self.windows[5], add_label(images["morphology"], "Morphology"))
        cv2.imshow(self.windows[6], add_label(images["detections"], "Detections"))
        cv2.imshow(self.windows[7], images["info"])
    
    def destroy(self):
        cv2.destroyAllWindows()


# ============================================================================
# Human Detector Class
# ============================================================================

class HumanDetector:
    """Human detection pipeline with enhanced tracking."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Background model
        bg_cfg = config.get("background", {})
        bg_type = bg_cfg.get("type", "running_average")
        
        if bg_type == "variance":
            self.background_model = RunningAverageWithVariance(
                learning_rate=bg_cfg.get("learning_rate", 0.05)
            )
        else:
            self.background_model = RunningAverageBackground(
                learning_rate=bg_cfg.get("learning_rate", 0.02)
            )
        
        # Enhanced Tracker
        blob_cfg = config.get("blob", {})
        self.tracker = BlobTracker(
            max_distance=blob_cfg.get("max_distance", 100.0),
            max_iou_distance=blob_cfg.get("max_iou_distance", 0.7),
            max_disappeared=blob_cfg.get("max_disappeared", 30),
            min_hits=blob_cfg.get("min_hits", 3),
            iou_weight=blob_cfg.get("iou_weight", 0.25),
            distance_weight=blob_cfg.get("distance_weight", 0.30),
            size_weight=blob_cfg.get("size_weight", 0.15),
            appearance_weight=blob_cfg.get("appearance_weight", 0.20),
            velocity_weight=blob_cfg.get("velocity_weight", 0.10),
            process_noise=blob_cfg.get("process_noise", 4.0),
            measurement_noise=blob_cfg.get("measurement_noise", 1.0)
        )
        
        self.triggered_ids: set[int] = set()
        self.frame_count = 0
        self.last_thresh_value = 0
        self.last_n_blobs = 0
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame and return all intermediate results."""
        cfg = self.config
        self.frame_count += 1
        results = {}
        
        # 1. Resize
        resize_cfg = cfg.get("resize", {})
        if resize_cfg.get("width") and resize_cfg.get("height"):
            frame = resize_frame_fast(frame, int(resize_cfg["width"]), int(resize_cfg["height"]))
        results["original"] = frame.copy()
        
        # 2. Grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results["grayscale"] = grayscale.copy()
        
        # 3. Gaussian Blur
        gauss_cfg = cfg.get("gaussian", {})
        blurred = apply_gaussian_blur(
            grayscale,
            gauss_cfg.get("kernel_size", 5),
            gauss_cfg.get("sigma", 1.0)
        )
        results["blurred"] = blurred.copy()
        
        # 4. Background Subtraction
        foreground, _ = self.background_model.apply(blurred)
        results["foreground"] = foreground.copy()
        
        # 5. Thresholding
        thresh_cfg = cfg.get("threshold", {})
        thresh_value = compute_adaptive_threshold(foreground, thresh_cfg)
        self.last_thresh_value = thresh_value
        binary = binary_threshold(foreground, thresh_value)
        results["binary"] = binary.copy()
        
        # 6. Morphology
        morph_cfg = cfg.get("morphology", {})
        processed = binary
        
        if morph_cfg.get("opening_iterations", 0) > 0:
            processed = opening(processed, iterations=morph_cfg.get("opening_iterations", 1),
                              kernel_size=morph_cfg.get("kernel_size", 3))
        if morph_cfg.get("erosion_iterations", 0) > 0:
            processed = erode(processed, iterations=morph_cfg.get("erosion_iterations", 1),
                            kernel_size=morph_cfg.get("kernel_size", 3))
        if morph_cfg.get("dilation_iterations", 0) > 0:
            processed = dilate(processed, iterations=morph_cfg.get("dilation_iterations", 1),
                             kernel_size=morph_cfg.get("kernel_size", 3))
        if morph_cfg.get("closing_iterations", 0) > 0:
            processed = closing(processed, iterations=morph_cfg.get("closing_iterations", 1),
                              kernel_size=morph_cfg.get("kernel_size", 3))
        
        results["morphology"] = processed.copy()
        
        # 7. Connected Components & Filtering
        _, components = connected_components(processed)
        
        blob_cfg = cfg.get("blob", {})
        aspect_range = blob_cfg.get("aspect_ratio_range", [0.2, 4.0])
        blobs = filter_blobs(
            components,
            min_area=blob_cfg.get("min_area", 500),
            max_area=blob_cfg.get("max_area", 50000),
            aspect_ratio_range=(float(aspect_range[0]), float(aspect_range[1]))
        )
        self.last_n_blobs = len(blobs)
        
        # 8. Enhanced Tracking
        tracks = self.tracker.update(blobs, frame=grayscale)    
        results["tracks"] = tracks
        results["confirmed_tracks"] = self.tracker.get_confirmed_tracks()
        
        return results
    
    def check_entry_events(self, tracks: List[Track]) -> List[int]:
        """Check for new confirmed entries."""
        det_cfg = self.config.get("detection", {})
        min_history = det_cfg.get("min_history", 3)
        min_area = det_cfg.get("min_area", 500)
        
        new_entries = []
        for track in tracks:
            track_id = track["id"]
            state = track.get("state", "tentative")
            history = track.get("history", [])
            area = track.get("area", 0)
            
            # Only trigger for confirmed tracks
            if (state == "confirmed" and
                len(history) >= min_history and 
                area >= min_area and 
                track_id not in self.triggered_ids):
                self.triggered_ids.add(track_id)
                new_entries.append(track_id)
        
        return new_entries
    
    def reset(self) -> None:
        """Reset detector state."""
        self.background_model.reset()
        self.tracker.reset()
        self.triggered_ids.clear()
        self.frame_count = 0


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(config: Dict[str, Any]) -> None:
    """Run detection pipeline with multi-window visualization."""
    
    video_source = config.get("video_source", 0)
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video_source}")
    
    resize_cfg = config.get("resize", {})
    win_w = resize_cfg.get("width", 320)
    win_h = resize_cfg.get("height", 240)
    
    detector = HumanDetector(config)
    display = MultiWindowDisplay(window_size=(win_w, win_h))
    
    draw_trails = config.get("draw_trails", True)
    fps_start = time.time()
    fps_count = 0
    current_fps = 0.0
    
    print("\n" + "="*55)
    print("  HUMAN DETECTION - ENHANCED TRACKING PIPELINE")
    print("="*55)
    print("  Q - Quit      R - Reset      T - Trails")
    print("  S - Screenshot    +/- Threshold adjust")
    print("="*55)
    print("  Tracker: Kalman Filter + Hungarian Algorithm")
    print("  Matching: IoU + Distance + Size")
    print("="*55 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Process
            results = detector.process_frame(frame)
            tracks = results["tracks"]
            confirmed_tracks = results["confirmed_tracks"]
            
            # Check entries
            new_entries = detector.check_entry_events(confirmed_tracks)
            for tid in new_entries:
                print(f"[EVENT] Human entered! Track ID: {tid}")
            
            # FPS
            fps_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_count / (time.time() - fps_start)
                fps_count = 0
                fps_start = time.time()
            
            # Draw detections (show all tracks, not just confirmed)
            detection_frame = draw_detections(results["original"], tracks, draw_trails)
            
            # Info panel
            info_panel = create_info_panel(
                win_w, win_h, current_fps,
                detector.last_thresh_value,
                detector.last_n_blobs,
                len(tracks),
                len(confirmed_tracks),
                detector.frame_count
            )
            
            # Update windows
            display.update({
                "original": results["original"],
                "grayscale": results["grayscale"],
                "blurred": results["blurred"],
                "foreground": results["foreground"],
                "binary": results["binary"],
                "morphology": results["morphology"],
                "detections": detection_frame,
                "info": info_panel
            })
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                detector.reset()
                print("[INFO] Tracker reset")
            elif key == ord('t') or key == ord('T'):
                draw_trails = not draw_trails
                print(f"[INFO] Trails: {'ON' if draw_trails else 'OFF'}")
            elif key == ord('s') or key == ord('S'):
                fname = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, detection_frame)
                print(f"[INFO] Saved: {fname}")
            elif key == ord('+') or key == ord('='):
                config.setdefault("threshold", {})["offset"] = config.get("threshold", {}).get("offset", 5.0) + 2.0
                print(f"[INFO] Threshold offset: {config['threshold']['offset']}")
            elif key == ord('-') or key == ord('_'):
                config.setdefault("threshold", {})["offset"] = max(0, config.get("threshold", {}).get("offset", 5.0) - 2.0)
                print(f"[INFO] Threshold offset: {config['threshold']['offset']}")
    
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        display.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human Detection - Enhanced Tracking")
    parser.add_argument("--config", "-c", default="settings.yaml")
    parser.add_argument("--source", "-s", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    try:
        config = load_settings(args.config)
    except FileNotFoundError:
        print(f"Config bulunamadi: {args.config}, varsayilan kullaniliyor...")
        config = {
            "video_source": 0,
            "resize": {"width": 320, "height": 240},
            "gaussian": {"kernel_size": 5, "sigma": 1.0},
            "background": {"learning_rate": 0.01},
            "threshold": {"value": 15, "adaptive": True, "std_factor": 1.2, "offset": 3.0},
            "morphology": {"opening_iterations": 1, "closing_iterations": 2, "kernel_size": 3},
            "blob": {
                "min_area": 700,
                "max_area": 50000,
                "aspect_ratio_range": [0.25, 0.65],
                "max_distance": 100.0,
                "max_age": 30,
                "min_hits": 3,
                "iou_weight": 0.35,
                "distance_weight": 0.45,
                "size_weight": 0.20
            },
            "detection": {"min_history": 3, "min_area": 700}
        }
    
    if args.source:
        try:
            config["video_source"] = int(args.source)
        except ValueError:
            config["video_source"] = args.source
    
    run_pipeline(config)


if __name__ == "__main__":
    main()