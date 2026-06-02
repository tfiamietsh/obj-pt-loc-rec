from time import time
from collections import deque


class FpsCounter:
    def __init__(self, avg_of: int = 5) -> None:
        self.__elapsed_times = deque(maxlen=avg_of)
        self.__prev_time = time()
        self.__fps = 0.0

    @property
    def fps(self) -> float:
        return self.__fps

    def update_fps(self):
        current_time = time()

        self.__elapsed_times.append(current_time - self.__prev_time)
        self.__prev_time = current_time
        self.__fps = 1.0 / (sum(self.__elapsed_times) / len(self.__elapsed_times))
