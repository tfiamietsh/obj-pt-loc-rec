import numpy as np
from matplotlib.colors import hex2color
from engines.abstract_engine import AbstractEngine


class DeepLabSegmenter:
    """Попиксельный сегментатор деталей автомобиля DeepLab"""

    def __init__(self, engine: AbstractEngine) -> None:
        self.__engine = engine
        self.__metaclass = "car"
        self.__image_size = (224, 224)
        self.__raw_classes = [
            {"name": "background", "color": "#000000"},
            {"name": "hood", "color": "#4355CB"},
            {"name": "trunk", "color": "#E5F83A"},
            {"name": "windshield", "color": "#9C969D"},
            {"name": "rear window", "color": "#320698"},
            {"name": "headlight", "color": "#2E7F62"},
            {"name": "tail light", "color": "#903765"},
            {"name": "front door", "color": "#E62D30"},
            {"name": "rear door", "color": "#7C93DA"},
            {"name": "front bumper", "color": "#4AF778"},
            {"name": "rear bumper", "color": "#50C878"},
            {"name": "wheel", "color": "#8206DB"},
            {"name": "mirror", "color": "#DEC723"}
        ]
        self.__classes = [color["name"] for color in self.__raw_classes]
        self.__colors_rgb = self.__get_rgb_colors(self.__raw_classes)
        self.__colors_bgr = self.__colors_rgb[:, ::-1].copy()

    @property
    def classes(self) -> list[str]:
        return self.__classes

    @property
    def metaclass(self) -> str:
        return self.__metaclass

    @property
    def image_size(self) -> tuple[int, int]:
        return self.__image_size

    @property
    def colors_rgb(self) -> np.array:
        return self.__colors_rgb

    @property
    def colors_bgr(self) -> np.array:
        return self.__colors_bgr

    @staticmethod
    def __get_rgb_colors(classes_info: list[dict[str, str]]) -> np.ndarray:
        return np.array([
            hex2color(class_info.get("color", "#000000"))
            for class_info in classes_info
        ], dtype=np.float32)

    def __call__(self, crops_batch: np.ndarray) -> np.ndarray:
        """
        Запуск сегментатора DeepLab

        :param crops_batch: вырезанный мини-батч [N, 3, 224, 224]
        :return: маска сегментации
        """
        return self.__engine(crops_batch)[0]
