# Human Entry Detection (Manual Computer Vision)

This project implements a full classical computer vision pipeline that detects when a person enters a room monitored by a fixed camera. Every image-processing primitive—Gaussian blur, background subtraction, thresholding, morphology, and connected components labeling—is coded manually in Python/NumPy without relying on OpenCV's built-ins (except for frame capture, color conversion, and display utilities).

## Features
- Manual Gaussian kernel generation and 2D convolution
- Running-average background model with manual difference computation
- Binary thresholding, erosion, and dilation written from scratch
- Custom 8-connected component labeling plus blob statistics
- Blob filtering, lightweight centroid tracker, and ROI entry detection
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
    roi.py
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
- `threshold.value`: Binary cutoff applied to the foreground difference.
- `morphology.erosion_iterations`, `morphology.dilation_iterations`: Manual morphology passes.
- `blob.*`: Area limits, allowable aspect ratio, tracker tuning.
- `roi.top_left`, `roi.bottom_right`: Door boundary in `(row, column)` coordinates.

Adjust these values to match your environment—for example, enlarge the ROI coordinates to cover your doorway and tweak the minimum area to ignore noise.

## Running the Detector
```bash
python src/main.py --config config/settings.yaml
```
Press `q` to exit the display windows. Whenever a tracked blob crosses into the ROI, the program prints `Human entered the room` and highlights the detection on screen.

## Notes
- All image processing beyond color conversion and display windows is implemented manually on NumPy arrays.
- The tracker maintains centroid history to enforce motion continuity and prevent duplicate entry events.
- For reproducible testing without a camera, place a short door video clip at `samples/test_video.mp4`.
