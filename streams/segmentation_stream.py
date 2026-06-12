import time
import threading
import numpy as np
from typing import Any
from utils.profiler import Profiler
from pipelines.pipeline_factory import PipelineFactory


class SegmentationStream:
    def __init__(self, config: Any, colors_bgr: np.ndarray) -> None:
        device = config.get("device", "CPU")

        self.__pipeline = PipelineFactory.build_pipeline(
            config=config,
            colors_bgr=colors_bgr,
            device=device
        )
        self.__stopped = False
        self.__mask = None
        self.__running = False
        self.__thread = None
        self.__current_frame = None
        self.__detected_objects = None
        self.__lock = threading.Lock()

    def start(self) -> None:
        self.__running = True
        threading.Thread(target=self.__update, args=(), daemon=True).start()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        with self.__lock:
            self.__current_frame = frame.copy()

            mask = self.__mask.copy() if self.__mask is not None else None
            objects = self.__detected_objects.copy() if self.__detected_objects is not None else None

            return mask, objects

    def stop(self) -> None:
        self.__running = False
        if self.__thread and self.__thread.is_alive():
            self.__thread.join()

    def __update(self):
        while self.__running:
            with self.__lock:
                frame_to_process = self.__current_frame.copy() if self.__current_frame is not None else None

            if frame_to_process is not None:
                Profiler.time("SegmentationStream.__update.__pipeline.process_frame")
                mask, detected_objects = \
                    self.__pipeline.process_frame(frame_to_process)
                Profiler.time()

                with self.__lock:
                    self.__mask = mask
                    self.__detected_objects = detected_objects

            time.sleep(0.005)
