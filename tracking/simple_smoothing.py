import cv2
import numpy as np


class SimpleSmoothing:
    def __init__(self, smooth_factor: float) -> None:
        self.__smooth_factor = smooth_factor

    def apply(
        self,
        prev_mask: np.ndarray,
        new_mask: np.ndarray
    ) -> np.ndarray:
        smoothed = cv2.addWeighted(
            prev_mask, 1.0 - self.__smooth_factor,
            new_mask, self.__smooth_factor, 0
        )
        current_object_mask = np.any(new_mask > 0, axis=-1)

        return np.where(current_object_mask[..., None], smoothed, new_mask)
