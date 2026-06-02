import numpy as np
from tasks.yolo_seg import YOLOSeg
from pipelines.abstract_pipeline import AbstractPipeline


class OneStagedPipeline(AbstractPipeline):
    """Одностадийный пайплайн (YOLOv8-seg/YOLO11-seg)"""

    def __init__(self, yolo_seg: YOLOSeg) -> None:
        self.__yolo_seg = yolo_seg

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Обработать кадр

        :param frame: ненормализованный кадр uint8
        :return: маска сегментации uint8
        """
        return self.__yolo_seg(frame)

    def __str__(self) -> str:
        return "One Staged"
