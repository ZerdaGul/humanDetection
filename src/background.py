from __future__ import annotations

import numpy as np


class RunningAverageBackground:
    """Maintains a running-average background model for subtraction."""

    def __init__(self, learning_rate: float = 0.05) -> None:
        self.learning_rate = max(1e-4, min(float(learning_rate), 1.0))
        self.background: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame.ndim != 2:
            raise ValueError("Background model expects grayscale input")

        frame_float = frame.astype(np.float32)

        if self.background is None:
            self.background = frame_float.copy()
            return np.zeros_like(frame, dtype=np.uint8), frame.copy()

        self.background = (
            (1.0 - self.learning_rate) * self.background + self.learning_rate * frame_float
        )
        difference = np.abs(frame_float - self.background)
        foreground = np.clip(difference, 0, 255).astype(np.uint8)
        return foreground, self.background.astype(np.uint8)

    def reset(self) -> None:
        self.background = None
