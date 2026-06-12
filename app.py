import cv2
import json
import numpy as np
from utils.profiler import Profiler
from utils.img_utils import ImgUtils
from windows.viewport import Viewport
from windows.info_window import InfoWindow
from engines.dummy_engine import DummyEngine
from tasks.deeplab_segmenter import DeepLabSegmenter
from streams.segmentation_stream import SegmentationStream
from tracking.historical_object_tracker import HistoricalObjectTracker


class App:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            config = json.load(file)

        reference_seg = DeepLabSegmenter(DummyEngine())

        self.__alpha = config["alpha"]
        self.__info_window = InfoWindow(
            classes=reference_seg.classes,
            colors_bgr=reference_seg.colors_bgr
        )
        self.__segmentation_stream = SegmentationStream(
            config=config,
            colors_bgr=reference_seg.colors_bgr
        )
        self.__backend = config.get("backend", "openvino")
        self.__colors = np.clip(reference_seg.colors_bgr * 255, 0, 255).astype(np.uint8)
        self.__shutdown_key = "q"
        self.__viewport = Viewport(config)
        self.__viewport.set_prerender_fn(self.__process_frame)
        self.__tracker = HistoricalObjectTracker()
        self.__last_detected_objects = None

    def __process_frame(self, frame: np.ndarray):
        Profiler.time("App.__process_frame.__segmentation_stream.process_frame")
        mask, detected_objects = self.__segmentation_stream.process_frame(frame)

        if detected_objects is not None:
            self.__last_detected_objects = detected_objects

        Profiler.time("App.__process_frame.__tracker.process")
        mask, objects = self.__tracker.process(mask, self.__last_detected_objects)
        self.__last_detected_objects = objects

        Profiler.time("App.__process_frame.ImgUtils.mix")
        masked_frame = ImgUtils.mix(frame, self.__colors[mask], self.__alpha)
        Profiler.time()

        return masked_frame

    def main_loop(self) -> None:
        self.__info_window.render()
        self.__segmentation_stream.start()

        while True:
            Profiler.time("App.main_loop.__viewport.read_frame")
            if not self.__viewport.read_frame():
                break

            Profiler.time("App.main_loop.__viewport.render")
            self.__viewport.render()
            Profiler.time()

            if cv2.waitKey(1) & 0xFF == ord(self.__shutdown_key):
                break

    def shutdown(self) -> None:
        self.__segmentation_stream.stop()
        self.__viewport.close()
        cv2.destroyAllWindows()
