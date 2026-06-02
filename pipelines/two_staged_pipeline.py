import cv2
import numpy as np
from tasks.yolo_detector import YOLODetector
from tasks.deeplab_segmenter import DeepLabSegmenter
from pipelines.abstract_pipeline import AbstractPipeline


class TwoStagedPipeline(AbstractPipeline):
    """Двухстадийный пайплайн (YOLOv6 + DeepLabV3+)"""

    def __init__(
        self,
        detector: YOLODetector,
        segmenter: DeepLabSegmenter
    ) -> None:
        self.__detector = detector
        self.__segmenter = segmenter

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Обработать кадр

        :param frame: ненормализованный кадр uint8
        :return: маска сегментации uint8
        """
        h_frame, w_frame = frame.shape[:2]
        frame_fp32 = frame.astype(np.float32) / 255.0

        num_dets, det_boxes, det_scores, det_classes = \
            self.__detector(frame_fp32)

        valid_crops, valid_pairs = [], []
        score_idx_pairs = []

        for i in range(int(num_dets[0])):
            class_id = int(det_classes[0, i])

            if self.__is_target_metaclass(class_id):
                score_idx_pairs.append((det_scores[0, i], i))

        if not score_idx_pairs:
            return np.zeros_like(frame)

        score_idx_pairs.sort()
        for score, j in score_idx_pairs:
            x1, y1, x2, y2 = map(int, det_boxes[0, j])

            if -1 < x1 < x2 and -1 < y1 < y2:
                crop = frame_fp32[y1:y2, x1:x2]
                resized_crop = cv2.resize(
                    src=crop,
                    dsize=self.__segmenter.image_size
                ).transpose(2, 0, 1)

                valid_crops.append(resized_crop)
                valid_pairs.append((score, j))

        if not valid_crops:
            return np.zeros_like(frame)

        X = np.stack(valid_crops, axis=0)
        segmenter_outputs = self.__segmenter(X)

        mask = np.zeros_like(frame_fp32)
        for i, (_, j) in enumerate(valid_pairs):
            x1, y1, x2, y2 = map(int, np.round(det_boxes[0, j]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)

            target_h, target_w = y2 - y1, x2 - x1
            if target_h <= 0 or target_w <= 0:
                continue

            preds = np.argmax(segmenter_outputs[i], axis=0)
            resized_mask = cv2.resize(
                src=self.__segmenter.colors_bgr[preds],
                dsize=(target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            )

            temp = np.zeros_like(frame_fp32)
            temp[y1:y2, x1:x2] = resized_mask[:target_h, :target_w]

            indices = temp > 1e-6
            mask[indices] = temp[indices]

        return np.clip(mask * 255, 0, 255).astype(np.uint8)

    def __is_target_metaclass(self, class_id: int) -> bool:
        return class_id in self.__detector.classes and \
            self.__detector.classes[class_id] == self.__segmenter.metaclass

    def __str__(self) -> str:
        return "Two Staged"
