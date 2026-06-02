import numpy as np
from engines.abstract_engine import AbstractEngine


class DummyEngine(AbstractEngine):
    """Класс-пустышка"""

    def __init__(self) -> None:
        super().__init__("")

    def _init_model(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        return [np.zeros((1, 13, 224, 224))]
