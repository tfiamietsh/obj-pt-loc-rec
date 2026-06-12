import numpy as np
from typing import Any, Callable
from utils.profiler import Profiler
from utils.fps_counter import FpsCounter
from windows.base_window import BaseWindow
from streams.video_stream import VideoStream


class Viewport(BaseWindow):
    def __init__(self, config: Any) -> None:
        super().__init__("Viewport")

        self.__video_stream = VideoStream(config["camera_id"])
        self.__fps_counter = FpsCounter()
        self.__video_stream.start()
        self.__prerender_fn = None

    def render(self) -> None:
        Profiler.time("Viewport.render.__prerender_fn")
        if self.__prerender_fn is not None:
            self._set_backbuffer(self.__prerender_fn(self._backbuffer))

        Profiler.time("Viewport.render.__update_title")
        self.__update_title()
        Profiler.time("Viewport.render.super().render")
        super().render()

        Profiler.time("Viewport.render.__fps_counter.update_fps")
        self.__fps_counter.update_fps()
        Profiler.time()

    def render_frame(self, frame: np.ndarray) -> None:
        self._set_backbuffer(frame)

        Profiler.time("Viewport.render.__update_title")
        self.__update_title()
        Profiler.time("Viewport.render.super().render")
        super().render()

        Profiler.time("Viewport.render.__fps_counter.update_fps")
        self.__fps_counter.update_fps()
        Profiler.time()

    def set_prerender_fn(self, prerender_fn: Callable) -> None:
        self.__prerender_fn = prerender_fn

    def read_frame(self) -> bool:
        Profiler.time("Viewport.read_frame.__video_stream.read_frame")
        frame = self.__video_stream.read_frame()

        Profiler.time("Viewport.read_frame._set_backbuffer")
        self._set_backbuffer(frame)

        Profiler.time()
        return self.__video_stream.is_open()

    def close(self) -> None:
        self.__video_stream.stop()

    def __update_title(self) -> None:
        self._set_title(f"Viewport | FPS: {self.__fps_counter.fps: .2f}")
