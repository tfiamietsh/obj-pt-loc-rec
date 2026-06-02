import numpy as np
from abc import ABC, abstractmethod


class AbstractEngine(ABC):
    """Абстрактный класс для инференса моделей"""

    def __init__(self, model_path: str, device: str = "CPU") -> None:
        self.model_path = model_path
        self.device = device
        self._init_model()

    @abstractmethod
    def _init_model(self) -> None:
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        pass
