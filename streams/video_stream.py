import cv2
import time
import threading
import numpy as np


class VideoStream:
    def __init__(self, src: str | int) -> None:
        backend = cv2.CAP_DSHOW if isinstance(src, int) else 0
        capture = cv2.VideoCapture(src, backend)

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        _, self.__frame = capture.read()
        self.__stopped = False
        self.__capture = capture

    def start(self) -> None:
        threading.Thread(target=self.__update, args=(), daemon=True).start()

    def read_frame(self) -> tuple[bool, np.ndarray]:
        return self.__frame

    def stop(self) -> None:
        self.__stopped = True
        self.__capture.release()

    def is_open(self) -> bool:
        return self.__capture.isOpened() and not self.__stopped

    def __update(self):
        while not self.__stopped:
            if self.__capture.isOpened():
                status, self.__frame = self.__capture.read()

                if not status:
                    self.__stopped = True
            time.sleep(0.005)
