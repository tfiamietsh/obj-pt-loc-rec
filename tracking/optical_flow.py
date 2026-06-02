import cv2
import numpy as np


class OpticalFlow:
    def __init__(self) -> None:
        self.__tracked_objects = []
        self.__prev_gray = None

    def update_tracked_objects(self, tracked_objects: list[dict]) -> None:
        self.__tracked_objects = tracked_objects

    def apply(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        mask = np.zeros_like(frame)

        if self.__prev_gray is None or len(self.__tracked_objects) == 0:
            return mask

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for obj in self.__tracked_objects:
            bx1, by1, bx2, by2 = obj["bbox"]

            if bx2 - bx1 < 5 or by2 - by1 < 5:
                continue

            old_roi = self.__prev_gray[by1:by2, bx1:bx2]
            new_roi = gray[by1:by2, bx1:bx2]

            shift = cv2.phaseCorrelate(
                src1=old_roi.astype(np.float32),
                src2=new_roi.astype(np.float32)
            )
            (dx, dy), _ = shift

            if not (np.isnan(dx) or np.isnan(dy)):
                bx1 = int(max(0, bx1 + dx))
                by1 = int(max(0, by1 + dy))
                bx2 = int(min(frame.shape[1], bx2 + dx))
                by2 = int(min(frame.shape[0], by2 + dy))
                obj["bbox"] = [bx1, by1, bx2, by2]

            h_obj, w_obj = by2 - by1, bx2 - bx1
            if w_obj > 0 and h_obj > 0:
                resized_crop = cv2.resize(
                    obj["crop_mask"].astype(np.uint8),
                    (w_obj, h_obj),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                mask[by1:by2, bx1:bx2][resized_crop] = obj["color"]

        self.__prev_gray = gray

        return mask
