from typing import Sequence, Tuple

import cv2


Point = Tuple[float, float]


class DoorROI:
    """Door region for entry detection."""

    def __init__(
        self,
        top_left: Sequence[int],
        bottom_right: Sequence[int],
        direction: str | None = None,
    ) -> None:
        self.top_left = (int(top_left[0]), int(top_left[1]))
        self.bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        self.direction = direction or "either"

    def contains(self, point: Point | None) -> bool:
        if point is None:
            return False
        y, x = point
        return (
            self.top_left[1] <= x <= self.bottom_right[1]
            and self.top_left[0] <= y <= self.bottom_right[0]
        )

    def check_entry(self, previous: Point | None, current: Point | None) -> bool:
        prev_inside = self.contains(previous)
        curr_inside = self.contains(current)
        return (not prev_inside) and curr_inside

    def draw(self, frame) -> None:
        cv2.rectangle(
            frame,
            (self.top_left[1], self.top_left[0]),
            (self.bottom_right[1], self.bottom_right[0]),
            (0, 255, 0),
            2,
        )
