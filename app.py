import cv2
import json
import numpy as np
from engines.dummy_engine import DummyEngine
from tasks.deeplab_segmenter import DeepLabSegmenter
from pipelines.pipeline_factory import PipelineFactory
from tracking.optical_flow import OpticalFlow
from windows.info_window import InfoWindow
from windows.viewport import Viewport
from utils.img_utils import ImgUtils


class App:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            config = json.load(file)

        device = config.get("device", "CPU")
        reference_seg = DeepLabSegmenter(DummyEngine())

        self.__alpha = config["alpha"]
        self.__frame_skip = config.get("frame_skip", 1)
        self.__pipeline = PipelineFactory.build_pipeline(
            config=config,
            colors_bgr=reference_seg.colors_bgr,
            device=device
        )
        self.__info_window = InfoWindow(
            classes=reference_seg.classes,
            colors_bgr=reference_seg.colors_bgr
        )
        self.__smooth_factor = config.get("smooth_factor", 0.6)
        self.__backend = config.get("backend", "openvino")
        self.__shutdown_key = "q"
        self.__viewport = Viewport(
            config=config,
            name_prefix=" | ".join([
                "Viewport",
                f"{self.__pipeline}",
                self.__backend
            ]),
            name_suffix=f"press '{self.__shutdown_key}' to exit"
        )
        self.__viewport.set_prerender_fn(self.__process_frame)
        self.__tracker = OpticalFlow()
        self.__frame_idx = 0
        self.__mask = None

    def __process_frame(self, frame: np.ndarray):
        self.__mask, _ = \
            self.__pipeline.process_frame(frame)

        return ImgUtils.mix(frame, self.__mask, self.__alpha)

    def main_loop(self) -> None:
        while True:
            if not self.__viewport.read_frame():
                break

            self.__viewport.render(self.__process_frame)

            self.__info_window.set_fps(self.__viewport.get_fps())
            self.__info_window.render()

            if cv2.waitKey(1) & 0xFF == ord(self.__shutdown_key):
                break

            self.__frame_idx += 1

    def shutdown(self) -> None:
        self.__viewport.close()
        cv2.destroyAllWindows()
