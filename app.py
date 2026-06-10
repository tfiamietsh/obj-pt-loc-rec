import cv2
import json
from engines.dummy_engine import DummyEngine
from tasks.deeplab_segmenter import DeepLabSegmenter
from pipelines.pipeline_factory import PipelineFactory
from tracking.optical_flow import OpticalFlow
from windows.info_window import InfoWindow
from utils.fps_counter import FpsCounter
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
        self.__cap = cv2.VideoCapture(config["camera_id"])
        self.__backend = config.get("backend", "openvino")
        self.__fps_counter = FpsCounter()
        self.__shutdown_key = "q"
        self.__app_name = " | ".join([
            "Viewport",
            f"{self.__pipeline}",
            self.__backend,
            f"press '{self.__shutdown_key}' to exit"
        ])
        self.__mask = None
        self.__frame_idx = 0
        self.__tracker = OpticalFlow()

    def main_loop(self) -> None:
        while self.__cap and self.__cap.isOpened():
            ret, frame = self.__cap.read()
            if not ret:
                break

            self.__mask, _ = \
                self.__pipeline.process_frame(frame)

            masked_frame = ImgUtils.mix(frame, self.__mask, self.__alpha)

            cv2.imshow(self.__app_name, masked_frame)

            self.__fps_counter.update_fps()
            self.__info_window.set_fps(self.__fps_counter.fps)

            if cv2.waitKey(1) & 0xFF == ord(self.__shutdown_key):
                break

            self.__frame_idx += 1

    def shutdown(self) -> None:
        if self.__cap is not None:
            self.__cap.release()
            cv2.destroyAllWindows()
