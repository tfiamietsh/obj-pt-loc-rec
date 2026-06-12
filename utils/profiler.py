from collections import deque
from time import perf_counter


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Profiler(metaclass=SingletonMeta):
    __hierarchy = {
        "Viewport": "App",
        "VideoStream": "Viewport",
        "SegmentationStream": "App",
        "YOLOSeg": "SegmentationStream"
    }

    def __init__(
        self,
        n: int = 100,
        hierarchy: dict[str, str] = __hierarchy
    ) -> None:
        self.__history = {}
        self.__max_len = n
        self.__start_time = None
        self.__current_name = None
        self.__hierarchy = hierarchy

    @classmethod
    def time(cls, name: str = None) -> None:
        Profiler().__time(name)

    @classmethod
    def summary(cls) -> None:
        Profiler().__summary()

    def __time(self, name: str) -> None:
        now = perf_counter()

        if self.__current_name is not None:
            elapsed_ms = (now - self.__start_time) * 1000

            if self.__current_name not in self.__history:
                self.__history[self.__current_name] = deque(
                    maxlen=self.__max_len
                )

            self.__history[self.__current_name].append(elapsed_ms)

        if name is not None:
            self.__current_name = name
            self.__start_time = perf_counter()
        else:
            self.__current_name = None

    def __calc_avg(self) -> dict[str, float]:
        avgs = {}

        for name, vals in self.__history.items():
            if len(vals) > 0:
                avgs[name] = sum(vals) / len(vals)

        return avgs

    def __calc_offset(self, step: str) -> str:
        if self.__hierarchy is None:
            return ''

        key = step.split(".", 1)[0]
        multiplyer = 0
        while key in self.__hierarchy:
            key = self.__hierarchy[key]
            multiplyer += 1

        return "    " * multiplyer

    def __summary(self) -> None:
        print("\n" + "-" * 40)
        print("Профилирование")
        print("-" * 40)
        for name, avg in self.__calc_avg().items():
            shifted_name = f"{self.__calc_offset(name)}{name}"

            print(
                f"{shifted_name: <60} |",
                f"Ср. время: {avg: >10.4f} мс"
            )
        print("-" * 50)
