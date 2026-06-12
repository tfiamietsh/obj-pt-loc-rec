import numpy as np


class ImgUtils:
    @staticmethod
    def mix(
        img1: np.ndarray,
        img2: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        mask = np.any(img2 > 0, axis=2)
        result = img1.copy()

        if mask.any():
            result[mask] = (img1[mask].astype(np.float32) * alpha +
                            img2[mask].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        return result
