import numpy as np


class HistoricalObjectTracker:
    def __init__(
        self,
        max_missing_frames: int = 3,
        iou_threshold: float = 0.2,
        weight_gain: float = 0.25,
        weight_decay: float = 0.03,
        stability_threshold: float = 0.6
    ) -> None:
        self.__stability_threshold = stability_threshold
        self.__max_missing_frames = max_missing_frames
        self.__iou_threshold = iou_threshold
        self.__weight_decay = weight_decay
        self.__weight_gain = weight_gain
        self.__tracked_objects = []
        self.__next_id = 0

    def process(
        self,
        mask: np.ndarray = None,
        detected_objects: list[dict] = None
    ) -> tuple[np.ndarray, list[dict]]:
        if mask is None:
            final_mask = np.zeros((480, 640), dtype=np.uint8)
        else:
            final_mask = np.zeros_like(mask)

        updated_tracks = []
        matched_track_ids = set()

        has_detections = (detected_objects is not None) and (len(detected_objects) > 0)

        if has_detections:
            for current_obj in detected_objects:
                det_box = current_obj["bbox"]

                best_iou = 0
                best_idx = -1

                for idx, track in enumerate(self.__tracked_objects):
                    iou = self.__calc_iou(det_box, track["object"]["bbox"])
                    if iou > best_iou and iou > self.__iou_threshold:
                        best_iou = iou
                        best_idx = idx

                bx1, by1, bx2, by2 = det_box
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                w, h = float(bx2 - bx1), float(by2 - by1)

                if best_idx != -1:
                    # обновляем состояние объекта
                    track = self.__tracked_objects[best_idx]
                    matched_track_ids.add(track["id"])

                    track["confidence"] = min(1.0, track["confidence"] + self.__weight_gain)
                    track["missing_frames"] = 0

                    # накапливаем и усредняем скорость
                    inst_vx = cx - track["cx"]
                    inst_vy = cy - track["cy"]
                    alpha = 0.4
                    track["vx"] = (1 - alpha) * track["vx"] + alpha * inst_vx
                    track["vy"] = (1 - alpha) * track["vy"] + alpha * inst_vy

                    # сохраняем исходную маску с индексами классов в истории
                    track["last_valid_crop"] = current_obj["crop_mask"].copy()

                    # обновляем геометрию
                    track["cx"], track["cy"], track["w"], track["h"] = cx, cy, w, h
                    track["object"] = current_obj
                    updated_tracks.append(track)
                else:
                    # добавляем маску в историю
                    new_track = {
                        "id": self.__next_id,
                        "confidence": self.__weight_gain,
                        "missing_frames": 0,
                        "cx": cx, "cy": cy, "w": w, "h": h,
                        "vx": 0.0, "vy": 0.0,
                        "last_valid_crop": current_obj["crop_mask"].copy(),
                        "object": current_obj
                    }
                    self.__next_id += 1
                    matched_track_ids.add(new_track["id"])
                    updated_tracks.append(new_track)

        # двигаем потерянные объекты
        for track in self.__tracked_objects:
            if track["id"] in matched_track_ids:
                continue

            track["missing_frames"] += 1
            track["confidence"] = max(0.0, track["confidence"] - self.__weight_decay)

            if track["confidence"] >= self.__stability_threshold and track[
                "missing_frames"] <= self.__max_missing_frames:
                # продвигаем центр по средней скорости
                track["cx"] += track["vx"]
                track["cy"] += track["vy"]

                bx1 = int(track["cx"] - track["w"] / 2.0)
                by1 = int(track["cy"] - track["h"] / 2.0)
                bx2 = int(track["cx"] + track["w"] / 2.0)
                by2 = int(track["cy"] + track["h"] / 2.0)
                track["object"]["bbox"] = [bx1, by1, bx2, by2]

                # достаем сохраненную маску из истории
                track["object"]["crop_mask"] = track["last_valid_crop"].copy()

                updated_tracks.append(track)

        # отсекаем объекты по уверенности
        valid_tracks = [
            t for t in updated_tracks
            if t["confidence"] >= self.__stability_threshold or t["missing_frames"] > 0
        ]

        # послойная отрисовка
        render_order = sorted(
            valid_tracks,
            key=lambda t: (t["object"]["bbox"][2] - t["object"]["bbox"][0]) *
                          (t["object"]["bbox"][3] - t["object"]["bbox"][1]),
            reverse=True
        )

        for track in render_order:
            self.__render_object_mask(final_mask, track)

        # сохраняем пул объектов в историю для следующего кадра
        self.__tracked_objects = updated_tracks

        return final_mask, [t["object"] for t in valid_tracks]

    @staticmethod
    def __render_object_mask(final_mask: np.ndarray, track: dict) -> None:
        """Копируем crop_mask объекта на итоговую маску кадра"""
        bx1, by1, bx2, by2 = track["object"]["bbox"]
        img_h, img_w = final_mask.shape[:2]

        bx1_c, by1_c = max(0, bx1), max(0, by1)
        bx2_c, by2_c = min(img_w, bx2), min(img_h, by2)

        mask_area = final_mask[by1_c:by2_c, bx1_c:bx2_c]
        crop_mask = track["object"]["crop_mask"]

        if mask_area.size > 0:
            t_h, t_w = mask_area.shape[:2]
            crop_resized = crop_mask[:t_h, :t_w]

            if crop_resized.shape != (t_h, t_w):
                tmp = np.zeros((t_h, t_w), dtype=crop_mask.dtype)
                tmp[:crop_resized.shape[0], :crop_resized.shape[1]] = crop_resized
                crop_resized = tmp

            mask_area[crop_resized > 0] = crop_resized[crop_resized > 0]

    @staticmethod
    def __calc_iou(box_a: list[int], box_b: list[int]) -> float:
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        intersection = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection

        return intersection / union if union > 0.0 else 0.0
