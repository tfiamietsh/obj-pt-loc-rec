import cv2
import numpy as np


class BaseWindow:
    def __init__(self, name: str) -> None:
        self.__backbuffer = None
        self.__name = name

    def _set_title(self, title: str) -> None:
        cv2.setWindowTitle(self.__name, title)

    def _set_backbuffer(self, backbuffer: np.ndarray) -> None:
        self.__backbuffer = backbuffer

    @property
    def _backbuffer(self) -> np.ndarray:
        return self.__backbuffer

    def render(self) -> None:
        if self.__backbuffer is not None:
            cv2.imshow(self.__name, self.__backbuffer)
