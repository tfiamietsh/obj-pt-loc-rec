from typing import Any, Callable
from utils.fps_counter import FpsCounter
from utils.video_stream import VideoStream
from windows.base_window import BaseWindow


class Viewport(BaseWindow):
    def __init__(
        self,
        config: Any,
        name_prefix: str,
        name_suffix: str
    ) -> None:
        super().__init__("Viewport")

        self.__video_stream = VideoStream(config["camera_id"])
        self.__fps_counter = FpsCounter()
        self.__name_prefix = name_prefix
        self.__name_suffix = name_suffix
        self.__video_stream.start()
        self.__prerender_fn = None

    def render(self, process_fn: Callable = None) -> None:
        if process_fn is not None:
            self._set_backbuffer(process_fn(self._backbuffer))

        self.__update_title()
        super().render()

        self.__fps_counter.update_fps()

    def set_prerender_fn(self, prerender_fn: Callable) -> None:
        self.__prerender_fn = prerender_fn

    def read_frame(self) -> bool:
        _, frame = self.__video_stream.read()

        self._set_backbuffer(frame)

        return self.__video_stream.is_open()

    def get_fps(self) -> float:
        return self.__fps_counter.fps

    def close(self) -> None:
        self.__video_stream.stop()

    def __update_title(self) -> None:
        self._set_title(f"Viewport | FPS: {self.__fps_counter.fps: .2f}")
