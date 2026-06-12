import cv2
import numpy as np
from windows.base_window import BaseWindow


class InfoWindow(BaseWindow):
    def __init__(
        self,
        classes: list[str],
        colors_bgr: np.ndarray
    ) -> None:
        super().__init__("Info")

        backbuffer = np.zeros((264, 192, 3), np.float32)
        for i, class_name in enumerate(classes):
            text_color = (1., 1., 1.)
            box_color = tuple(map(float, colors_bgr[i]))

            backbuffer = self.__put_text(
                text=class_name,
                buffer=backbuffer,
                pos=(4, 16 + 20 * i),
                color=text_color
            )
            backbuffer = cv2.rectangle(
                img=backbuffer,
                pt1=(120, 5 + 20 * i),
                pt2=(185, 16 + 20 * i),
                color=box_color,
                thickness=-1
            )

        self._set_backbuffer(backbuffer)

    @staticmethod
    def __put_text(
        text: str,
        buffer: np.ndarray,
        pos: tuple[int, int],
        color: tuple[float, float, float]
    ) -> np.ndarray:
        return cv2.putText(
            img=buffer,
            text=text,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )
