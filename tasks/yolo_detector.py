import numpy as np
from engines.abstract_engine import AbstractEngine


class YOLODetector:
    """Детектор на базе YOLO"""

    def __init__(self, engine: AbstractEngine) -> None:
        self.__engine = engine
        self.__classes = {
            i: cls for i, cls in enumerate([
                "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                "train", "truck", "boat", "traffic light", "fire hydrant",
                "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie",
                "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog",
                "pizza", "donut", "cake", "chair", "couch", "potted plant",
                "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush"
            ])
        }

    @property
    def classes(self) -> dict[int, str]:
        return self.__classes

    def __call__(
        self,
        frame_fp32: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Запуск локализатора YOLO

        :param frame_fp32: нормализованный кадр в NCHW [1, 3, H, W]
        :return: данные о найденных объектах: n_dets, boxes, scores, classes
        """
        x = np.expand_dims(frame_fp32.transpose(2, 0, 1), axis=0)
        outputs = self.__engine(x)

        return outputs[0], outputs[1], outputs[2], outputs[3]
