import cv2
import numpy as np


class ImgUtils:
    @staticmethod
    def mix(
        img1: np.ndarray,
        img2: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        result = img1.copy()
        non_zero = np.any(img2 > 0, axis=-1)

        if np.any(non_zero):
            result[non_zero] = cv2.addWeighted(
                src1=img1,
                alpha=alpha,
                src2=img2,
                beta=1.0 - alpha,
                gamma=0
            )[non_zero]

        return result
