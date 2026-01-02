# Human Entry Detection (Manual Computer Vision)

This project implements a full classical computer vision pipeline that detects when a person enters a room monitored by a fixed camera. Every image-processing primitive—Gaussian blur, background subtraction, thresholding, morphology, and connected components labeling—is coded manually in Python/NumPy without relying on OpenCV's built-ins (except for frame capture, color conversion, and display utilities).

## Features
- Manual Gaussian kernel generation and fast vectorized 2D convolution
- Running-average background model with manual difference computation
- Adaptive thresholding plus handcrafted erosion, dilation, opening, and closing
- Custom 8-connected component labeling plus blob statistics
- Blob filtering plus a lightweight centroid tracker for stable detections
- Configurable pipeline through `config/settings.yaml`

## Project Layout
```
humanDetection/
  src/
    main.py
    gaussian.py
    threshold.py
    background.py
    morphology.py
    connected_components.py
    blob.py
    utils.py
  config/settings.yaml
  samples/test_video.mp4
  README.md
```
`samples/test_video.mp4` is a placeholder. Replace it with your actual door-feed recording or change `video_source` in the config to point at your camera index.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python pyyaml
```

## Configuration
`config/settings.yaml` exposes every stage of the pipeline:
- `video_source`: Path or camera index.
- `resize.width/height`: Optional frame resize before processing.
- `gaussian.kernel_size`, `gaussian.sigma`: Blur settings (kernel forced to odd size).
- `background.learning_rate`: Running average update factor (0–1).
- `threshold.value`: Base cutoff applied to the foreground difference.
- `threshold.adaptive`, `threshold.std_factor`, `threshold.offset`: Enable and tune adaptive thresholds computed from the current frame statistics.
- `morphology.opening_iterations`, `morphology.erosion_iterations`, `morphology.dilation_iterations`, `morphology.closing_iterations`: Control noise removal and blob solidification passes.
- `blob.*`: Area limits, allowable aspect ratio, tracker tuning.
- `detection.min_history`, `detection.min_area`: Require a track to persist for N frames and exceed a given area before firing the entry event.

Adjust these values to match your environment—for example, tweak the blob area limits to ignore noise and tune the detection thresholds until a person consistently triggers the alert.

## Running the Detector
```bash
python src/main.py --config config/settings.yaml
```
Press `q` to exit the display windows. Whenever a tracked blob is stable for the configured number of frames and large enough to be considered a person, the program prints `Human entered the room` and highlights the detection on screen. Use `Ctrl+C` in the terminal at any time for a graceful shutdown.

### Troubleshooting Performance / Mask Quality
- Lower the input resolution via `resize.width/height` for faster processing.
- Increase `threshold.std_factor` or `threshold.offset` if the mask still shows stray highlights; decrease them if people disappear.
- Increase `morphology.opening_iterations` to suppress isolated noise specks; increase `closing_iterations` to solidify the person silhouette before tracking.

## Notes
- All image processing beyond color conversion and display windows is implemented manually on NumPy arrays.
- The tracker maintains centroid history to enforce motion continuity and prevent duplicate entry events.
- For reproducible testing without a camera, place a short door video clip at `samples/test_video.mp4`.
