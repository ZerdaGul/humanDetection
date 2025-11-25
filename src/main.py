import argparse
from typing import Any, Dict, List

import cv2
import numpy as np

from background import RunningAverageBackground
from blob import BlobTracker, filter_blobs
from connected_components import connected_components
from gaussian import apply_gaussian_blur
from morphology import closing, dilate, erode, opening
from roi import DoorROI
from threshold import binary_threshold
from utils import load_settings, resize_frame


def build_roi(config: Dict[str, Any]) -> DoorROI:
    roi_cfg = config.get("roi", {})
    top_left = roi_cfg.get("top_left", [0, 0])
    bottom_right = roi_cfg.get("bottom_right", [100, 100])
    full_frame = bool(roi_cfg.get("full_frame", False))
    return DoorROI(top_left, bottom_right, roi_cfg.get("direction"), full_frame=full_frame)


def compute_threshold(foreground: np.ndarray, cfg: Dict[str, Any]) -> int:
    base_value = int(cfg.get("value", 30))
    if cfg.get("adaptive", False):
        mean_val = float(np.mean(foreground))
        std_val = float(np.std(foreground))
        adaptive_value = mean_val + cfg.get("std_factor", 1.5) * std_val + cfg.get("offset", 5.0)
        return max(base_value, int(adaptive_value))
    return base_value


def preprocess_frame(frame: np.ndarray, resize_cfg: Dict[str, Any]) -> np.ndarray:
    width = resize_cfg.get("width")
    height = resize_cfg.get("height")
    if width and height:
        frame = resize_frame(frame, int(width), int(height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def run_pipeline(config: Dict[str, Any]) -> None:
    video_source = config.get("video_source", 0)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")

    gaussian_cfg = config.get("gaussian", {"kernel_size": 5, "sigma": 1.0})
    threshold_cfg = config.get(
        "threshold",
        {"value": 30, "adaptive": True, "std_factor": 1.5, "offset": 5.0},
    )
    background_cfg = config.get("background", {"learning_rate": 0.02})
    morph_cfg = config.get(
        "morphology",
        {
            "opening_iterations": 1,
            "erosion_iterations": 0,
            "dilation_iterations": 0,
            "closing_iterations": 1,
        },
    )
    blob_cfg = config.get(
        "blob",
        {
            "min_area": 500,
            "max_area": 50000,
            "aspect_ratio_range": [0.2, 4.0],
            "max_distance": 60.0,
            "max_disappeared": 10,
        },
    )

    roi = build_roi(config)
    background_model = RunningAverageBackground(background_cfg.get("learning_rate", 0.02))
    tracker = BlobTracker(
        max_distance=blob_cfg.get("max_distance", 60.0),
        max_disappeared=blob_cfg.get("max_disappeared", 10),
    )

    display = config.get("display", True)
    resize_cfg = config.get("resize", {})
    triggered_tracks: set[int] = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = preprocess_frame(frame, resize_cfg)
            roi.update_bounds(gray.shape[0], gray.shape[1])
            blurred = apply_gaussian_blur(
                gray,
                gaussian_cfg.get("kernel_size", 5),
                gaussian_cfg.get("sigma", 1.0),
            )
            foreground, _ = background_model.apply(blurred)
            thresh_value = compute_threshold(foreground, threshold_cfg)
            binary_mask = binary_threshold(foreground, thresh_value)

            processed = binary_mask
            if morph_cfg.get("opening_iterations", 0) > 0:
                processed = opening(processed, morph_cfg.get("opening_iterations", 1))
            if morph_cfg.get("erosion_iterations", 0) > 0:
                processed = erode(processed, morph_cfg.get("erosion_iterations", 1))
            if morph_cfg.get("dilation_iterations", 0) > 0:
                processed = dilate(processed, morph_cfg.get("dilation_iterations", 1))
            if morph_cfg.get("closing_iterations", 0) > 0:
                processed = closing(processed, morph_cfg.get("closing_iterations", 1))

            _, components = connected_components(processed)
            aspect_range = blob_cfg.get("aspect_ratio_range", [0.2, 4.0])
            blobs = filter_blobs(
                components,
                blob_cfg.get("min_area", 500),
                blob_cfg.get("max_area", 50000),
                (float(aspect_range[0]), float(aspect_range[1])),
            )

            tracks = tracker.update(blobs)
            for track in tracks:
                history: List[Any] = track.get("history", [])  # type: ignore[assignment]
                if len(history) < 2:
                    continue
                previous = history[-2]
                current = history[-1]
                if roi.check_entry(previous, current) and track["id"] not in triggered_tracks:
                    print("Human entered the room")
                    triggered_tracks.add(track["id"])

            if display:
                display_frame = frame.copy()
                roi.draw(display_frame)
                for track in tracks:
                    bbox = track.get("bbox", (0, 0, 0, 0))
                    min_y, min_x, max_y, max_x = [int(v) for v in bbox]
                    cv2.rectangle(display_frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
                    centroid = track.get("centroid", (0.0, 0.0))
                    cv2.circle(display_frame, (int(centroid[1]), int(centroid[0])), 3, (0, 0, 255), -1)
                    cv2.putText(
                        display_frame,
                        f"ID {track['id']}",
                        (min_x, max(min_y - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Detections", display_frame)
                cv2.imshow("Mask", processed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Stopping detection loop...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual human entry detector")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_settings(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
