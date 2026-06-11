import cv2
import threading
import numpy as np


class VideoStream:
    def __init__(self, src: str | int) -> None:
        self.__stream = cv2.VideoCapture(src)
        self.__stopped = False
        self.__grabbed, self.__frame = self.__stream.read()

    def start(self) -> None:
        threading.Thread(target=self.update, args=(), daemon=True).start()

    def update(self):
        while not self.__stopped:
            grabbed, frame = self.__stream.read()

            if not grabbed:
                self.__stopped = True
                break

            self.__frame = frame

    def read(self) -> tuple[bool, np.ndarray]:
        return self.__grabbed, self.__frame

    def stop(self) -> None:
        self.__stopped = True
        self.__stream.release()

    def is_open(self) -> bool:
        return self.__grabbed and not self.__stopped
