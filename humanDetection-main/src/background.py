"""
Background subtraction models for motion detection.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BackgroundModel(ABC):
    """Abstract base class for background models."""
    
    @abstractmethod
    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a frame and return foreground mask and background estimate.
        
        Args:
            frame: Grayscale input frame
        
        Returns:
            Tuple of (foreground_mask, background_estimate)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the background model."""
        pass


class RunningAverageBackground(BackgroundModel):
    """
    Running average background model with adaptive learning rate.
    
    Simple but effective for static camera scenarios with gradual
    illumination changes.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        min_learning_rate: float = 0.01,
        max_learning_rate: float = 0.5
    ) -> None:
        """
        Args:
            learning_rate: Base learning rate for background update (0-1)
            min_learning_rate: Minimum adaptive learning rate
            max_learning_rate: Maximum adaptive learning rate
        """
        self.base_learning_rate = np.clip(learning_rate, 0.001, 1.0)
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.background: np.ndarray | None = None
        self._frame_count = 0
    
    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame.ndim != 2:
            raise ValueError("Background model expects grayscale input")
        
        frame_float = frame.astype(np.float32)
        self._frame_count += 1
        
        # Initialize background with first frame
        if self.background is None:
            self.background = frame_float.copy()
            return np.zeros_like(frame, dtype=np.uint8), frame.copy()
        
        # Adaptive learning rate - faster at start, slower when stable
        if self._frame_count < 30:
            # Fast adaptation during initialization
            alpha = min(0.5, self.base_learning_rate * 5)
        else:
            alpha = self.base_learning_rate
        
        # Update background
        self.background = (1.0 - alpha) * self.background + alpha * frame_float
        
        # Compute foreground
        difference = np.abs(frame_float - self.background)
        foreground = np.clip(difference, 0, 255).astype(np.uint8)
        
        return foreground, self.background.astype(np.uint8)
    
    def reset(self) -> None:
        self.background = None
        self._frame_count = 0


class RunningAverageWithVariance(BackgroundModel):
    """
    Running average with variance estimation for adaptive thresholding.
    
    Maintains both mean and variance of background, allowing for
    pixel-wise adaptive detection thresholds.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        variance_learning_rate: float = 0.01,
        std_threshold: float = 2.5
    ) -> None:
        """
        Args:
            learning_rate: Learning rate for mean update
            variance_learning_rate: Learning rate for variance update
            std_threshold: Number of standard deviations for foreground detection
        """
        self.alpha_mean = np.clip(learning_rate, 0.001, 1.0)
        self.alpha_var = np.clip(variance_learning_rate, 0.001, 1.0)
        self.std_threshold = max(1.0, std_threshold)
        
        self.mean: np.ndarray | None = None
        self.variance: np.ndarray | None = None
        self._frame_count = 0
    
    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame.ndim != 2:
            raise ValueError("Background model expects grayscale input")
        
        frame_float = frame.astype(np.float32)
        self._frame_count += 1
        
        # Initialize
        if self.mean is None:
            self.mean = frame_float.copy()
            self.variance = np.ones_like(frame_float) * 100.0  # Initial variance
            return np.zeros_like(frame, dtype=np.uint8), frame.copy()
        
        # Compute difference
        diff = frame_float - self.mean
        diff_squared = diff ** 2
        
        # Adaptive foreground detection using variance
        std = np.sqrt(np.maximum(self.variance, 1.0))
        foreground_mask = np.abs(diff) > (self.std_threshold * std)
        
        # Update only background pixels (selective update)
        background_mask = ~foreground_mask
        
        # Update mean
        self.mean = np.where(
            background_mask,
            (1 - self.alpha_mean) * self.mean + self.alpha_mean * frame_float,
            self.mean
        )
        
        # Update variance
        self.variance = np.where(
            background_mask,
            (1 - self.alpha_var) * self.variance + self.alpha_var * diff_squared,
            self.variance
        )
        
        # Clamp variance
        self.variance = np.clip(self.variance, 1.0, 5000.0)
        
        # Output foreground intensity
        foreground = np.clip(np.abs(diff), 0, 255).astype(np.uint8)
        
        return foreground, self.mean.astype(np.uint8)
    
    def reset(self) -> None:
        self.mean = None
        self.variance = None
        self._frame_count = 0
    
    def get_variance_image(self) -> np.ndarray | None:
        """Get current variance estimate (for visualization)."""
        if self.variance is None:
            return None
        return np.clip(np.sqrt(self.variance), 0, 255).astype(np.uint8)


class FrameDifferenceBackground(BackgroundModel):
    """
    Simple frame difference model.
    
    Compares current frame with previous frame. Fast but sensitive
    to noise and misses stationary objects.
    """
    
    def __init__(self, threshold: int = 25) -> None:
        self.threshold = max(1, threshold)
        self.prev_frame: np.ndarray | None = None
    
    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame.ndim != 2:
            raise ValueError("Background model expects grayscale input")
        
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return np.zeros_like(frame, dtype=np.uint8), frame.copy()
        
        # Compute absolute difference
        diff = np.abs(frame.astype(np.int16) - self.prev_frame.astype(np.int16))
        foreground = np.clip(diff, 0, 255).astype(np.uint8)
        
        # Store current frame as previous
        background = self.prev_frame.copy()
        self.prev_frame = frame.copy()
        
        return foreground, background
    
    def reset(self) -> None:
        self.prev_frame = None


class HybridBackground(BackgroundModel):
    """
    Hybrid model: Running Average + Frame Difference
    
    İki yöntemi birleştirerek hem durağan hem de 
    düz hareket eden nesneleri tespit eder.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        frame_diff_weight: float = 0.5
    ) -> None:
        """
        Args:
            learning_rate: Background güncelleme hızı (düşük = daha stabil)
            frame_diff_weight: Frame difference ağırlığı (0-1)
        """
        self.learning_rate = np.clip(learning_rate, 0.001, 0.1)
        self.frame_diff_weight = np.clip(frame_diff_weight, 0.0, 1.0)
        
        self.background: np.ndarray | None = None
        self.prev_frame: np.ndarray | None = None
        self._frame_count = 0
    
    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame.ndim != 2:
            raise ValueError("Background model expects grayscale input")
        
        frame_float = frame.astype(np.float32)
        self._frame_count += 1
        
        # İlk frame
        if self.background is None:
            self.background = frame_float.copy()
            self.prev_frame = frame_float.copy()
            return np.zeros_like(frame, dtype=np.uint8), frame.copy()
        
        # 1. Running average foreground
        bg_diff = np.abs(frame_float - self.background)
        
        # 2. Frame difference (önceki frame ile fark)
        frame_diff = np.abs(frame_float - self.prev_frame)
        
        # 3. İkisini birleştir (maksimum al)
        combined = np.maximum(
            bg_diff * (1 - self.frame_diff_weight),
            frame_diff * self.frame_diff_weight * 2  # Frame diff'i güçlendir
        )
        
        # Background'u SADECE hareket olmayan yerlerde güncelle
        motion_mask = combined > 25  # Hareket olan piksel
        update_mask = ~motion_mask
        
        self.background = np.where(
            update_mask,
            (1 - self.learning_rate) * self.background + self.learning_rate * frame_float,
            self.background  # Hareket varsa güncelleme
        )
        
        # Önceki frame'i kaydet
        self.prev_frame = frame_float.copy()
        
        foreground = np.clip(combined, 0, 255).astype(np.uint8)
        return foreground, self.background.astype(np.uint8)
    
    def reset(self) -> None:
        self.background = None
        self.prev_frame = None
        self._frame_count = 0


class TemporalMedianBackground(BackgroundModel):
    """
    Temporal median background model.
    
    Maintains a history of frames and uses median as background.
    More robust to outliers than running average, but uses more memory.
    """
    
    def __init__(self, history_size: int = 30) -> None:
        """
        Args:
            history_size: Number of frames to keep in history
        """
        self.history_size = max(5, history_size)
        self.history: list[np.ndarray] = []
        self.background: np.ndarray | None = None
    
    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if frame.ndim != 2:
            raise ValueError("Background model expects grayscale input")
        
        # Add frame to history
        self.history.append(frame.astype(np.float32))
        
        # Maintain history size
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Compute median background
        if len(self.history) >= 3:
            stacked = np.stack(self.history, axis=0)
            self.background = np.median(stacked, axis=0).astype(np.float32)
        else:
            self.background = self.history[-1].copy()
        
        # Compute foreground
        diff = np.abs(frame.astype(np.float32) - self.background)
        foreground = np.clip(diff, 0, 255).astype(np.uint8)
        
        return foreground, self.background.astype(np.uint8)
    
    def reset(self) -> None:
        self.history.clear()
        self.background = None