import numpy as np
from abc import ABC


class AbstractPipeline(ABC):
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        pass
