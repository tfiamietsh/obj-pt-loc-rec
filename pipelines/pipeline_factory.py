import numpy as np
from typing import Any
from tasks.yolo_seg import YOLOSeg
from tasks.yolo_detector import YOLODetector
from engines.engine_factory import EngineFactory
from tasks.deeplab_segmenter import DeepLabSegmenter
from pipelines.abstract_pipeline import AbstractPipeline
from pipelines.one_staged_pipeline import OneStagedPipeline
from pipelines.two_staged_pipeline import TwoStagedPipeline


class PipelineFactory:
    @staticmethod
    def build_pipeline(
        config: dict[str, Any],
        colors_bgr: np.ndarray,
        device: str
    ) -> AbstractPipeline:
        """Фабрика сборки пайплайна на основе конфигурации"""

        mode = config.get("pipeline_mode", "two_staged").lower()

        if mode == "one_staged":
            engine = EngineFactory.get_engine(
                config=config,
                model_path=config["models"]["yolo_seg"],
                device=device
            )

            yolo_task = YOLOSeg(
                engine=engine,
                colors_bgr=colors_bgr,
                conf_threshold=config.get("conf_threshold", 0.3)
            )

            return OneStagedPipeline(yolo_task)
        elif mode == "two_staged":
            engine_det = EngineFactory.get_engine(
                config=config,
                model_path=config["models"]["yolo_detector"],
                device=device,
                target_size=(480, 640)
            )
            engine_seg = EngineFactory.get_engine(
                config=config,
                model_path=config["models"]["deeplab_segmenter"],
                device=device,
                dynamic_batch=True
            )

            detector = YOLODetector(engine_det)
            segmenter = DeepLabSegmenter(engine_seg)

            return TwoStagedPipeline(detector, segmenter)
        else:
            raise ValueError(f"Неизвестный режим пайплайна: {mode}")
