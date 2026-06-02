import cv2
import numpy as np
from engines.abstract_engine import AbstractEngine


class YOLOSeg:
    """Instance-сегментация через YOLO-seg"""

    def __init__(
        self,
        engine: AbstractEngine,
        colors_bgr: np.ndarray,
        conf_threshold: float = 0.5
    ) -> None:
        self.__engine = engine
        self.__colors = np.clip(colors_bgr * 255, 0, 255).astype(np.uint8)
        self.__conf_threshold = conf_threshold

    @staticmethod
    def __preprocess(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Предобработка кадра

        :param frame: ненормализованный кадр uint8
        :return: нормализованный кадр float32 с размерами
        """
        h, w = frame.shape[:2]
        input_img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img.transpose(2, 0, 1), axis=0)

        return input_img, (h, w)

    def __postprocess(
        self,
        preds: list[np.ndarray],
        orig_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Конвертация прототипов в маски сегментации

        :param preds: предсказанный выход модели
        :param orig_shape: вырезанный мини-батч [N, 3, 224, 224]
        :return: маска сегментации uint8
        """
        orig_h, orig_w = orig_shape
        output0 = np.squeeze(preds[0]).T  # размер: (8400, 116)
        proto = np.squeeze(preds[1])  # размер: (32, 160, 160)

        boxes = output0[:, :4]
        scores = output0[:, 4:-32]
        mask_coefficients = output0[:, -32:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        keep = confidences > self.__conf_threshold
        if not np.any(keep):
            return np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        boxes, class_ids, mask_coefficients, confidences = \
            boxes[keep], class_ids[keep], \
            mask_coefficients[keep], confidences[keep]

        x_center, y_center, width, height = \
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = x_center - width / 2, y_center - height / 2

        cv_boxes = np.stack([x1, y1, width, height], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            bboxes=cv_boxes,
            scores=confidences.tolist(),
            score_threshold=self.__conf_threshold,
            nms_threshold=0.45
        )

        final_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        if len(indices) == 0:
            return final_mask

        indices = indices.flatten()
        proto_flat = proto.reshape(32, -1)
        coeffs = mask_coefficients[indices]

        masks = 1 / (1 + np.exp(-np.matmul(coeffs, proto_flat)))
        masks = masks.reshape(len(indices), 160, 160)

        for i, idx in enumerate(indices):
            seg_mask = cv2.resize(
                src=masks[i],
                dsize=(orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            ) > 0.5

            bx1 = int(max(0, x1[idx] * orig_w / 640))
            by1 = int(max(0, y1[idx] * orig_h / 640))
            bx2 = int(min(orig_w, (x1[idx] + width[idx]) * orig_w / 640))
            by2 = int(min(orig_h, (y1[idx] + height[idx]) * orig_h / 640))

            crop_mask = np.zeros_like(seg_mask)
            crop_mask[by1:by2, bx1:bx2] = True
            seg_mask &= crop_mask

            # сдвиг индекса на 1, так как 0 - background
            color_idx = (class_ids[idx] + 1) % len(self.__colors)
            final_mask[seg_mask] = self.__colors[color_idx]

        return final_mask

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Запуск модели YOLO-seg

        :param frame: ненормализованный кадр uint8
        :return: маска сегментации uint8
        """
        input_tensor, orig_shape = self.__preprocess(frame)
        preds = self.__engine(input_tensor)

        return self.__postprocess(preds, orig_shape)
