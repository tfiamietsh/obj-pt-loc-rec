import cv2
import numpy as np


class InfoWindow:
    def __init__(
        self,
        classes: list[str],
        colors_rgb: np.ndarray
    ) -> None:
        self.__window = np.zeros((264, 192, 3), np.float32)

        for i, class_name in enumerate(classes):
            text_color = (1., 1., 1.)
            box_color = tuple(map(float, colors_rgb[i]))

            self.__put_text(
                text=class_name,
                pos=(4, 36 + 20 * i),
                color=text_color
            )
            self.__window = cv2.rectangle(
                img=self.__window,
                pt1=(120, 25 + 20 * i),
                pt2=(185, 36 + 20 * i),
                color=box_color,
                thickness=-1
            )

    def set_fps(self, fps: float) -> None:
        info_display = self.__window.copy()
        info_display[:18, :140] = 0.0

        cv2.putText(
            img=info_display,
            text=f"FPS: {fps: .2f}",
            org=(4, 16),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(1., 1., 1.),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        cv2.imshow("Info", info_display)

    def __put_text(
        self,
        text: str,
        pos: tuple[int, int],
        color: tuple[float, float, float]
    ) -> None:
        self.__window = cv2.putText(
            img=self.__window,
            text=text,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA
        )
