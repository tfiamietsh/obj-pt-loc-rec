import numpy as np


class FrameHistoryTracker:
    def __init__(
        self,
        max_missing_frames: int = 7,
        iou_threshold: float = 0.2,
        alpha_speed: float = 0.6
    ) -> None:
        self.__iou_threshold = iou_threshold
        self.__max_missing_frames = max_missing_frames
        self.__alpha_speed = alpha_speed
        self.__objects = None

    def process(self, mask: np.ndarray, detected_objects: list[dict] = None) -> tuple[np.ndarray, list[dict]]:
        if mask is None:
            empty_mask = np.zeros((480, 640), dtype=np.uint8)
            return empty_mask, []

        if detected_objects is None or len(detected_objects) == 0:
            return self.__predict_mask(mask)
        else:
            filtered_detections = self.__filter_duplicate_detections(detected_objects)
            return self.__process_mask(mask, filtered_detections)

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

    def __filter_duplicate_detections(self, detected_objects: list[dict]) -> list[dict]:
        if len(detected_objects) <= 1:
            return detected_objects

        sorted_objs = sorted(
            detected_objects,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            reverse=True
        )

        filtered = []
        for obj in sorted_objs:
            box = obj["bbox"]
            is_duplicate = False

            for saved_obj in filtered:
                if obj["class_id"] == saved_obj["class_id"]:
                    if self.__calc_iou(box, saved_obj["bbox"]) > 0.70:
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered.append(obj)

        return filtered

    def __predict_mask(self, base_mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        if self.__objects is None:
            empty_mask = np.zeros_like(base_mask) if base_mask is not None else np.zeros((480, 640), dtype=np.uint8)
            return empty_mask, []

        predicted_tracks = []
        h, w = base_mask.shape[:2] if base_mask is not None else (480, 640)
        final_mask = np.zeros((h, w), dtype=np.uint8)

        for track in self.__objects:
            track["missing_frames"] += 1
            track["confidence"] = max(0.0, track["confidence"] - 0.15)

            if track["missing_frames"] <= self.__max_missing_frames and track["confidence"] > 0.1:
                track["cx"] += track["vx"]
                track["cy"] += track["vy"]
                track["w"] = max(10, track["w"] + track["vw"])
                track["h"] = max(10, track["h"] + track["vh"])

                bx1 = int(track["cx"] - track["w"] / 2)
                by1 = int(track["cy"] - track["h"] / 2)
                bx2 = int(track["cx"] + track["w"] / 2)
                by2 = int(track["cy"] + track["h"] / 2)

                track["object"]["bbox"] = [bx1, by1, bx2, by2]

                bx1_c, by1_c = max(0, bx1), max(0, by1)
                bx2_c, by2_c = min(w, bx2), min(h, by2)

                mask_area = final_mask[by1_c:by2_c, bx1_c:bx2_c]
                crop_mask = track["object"]["crop_mask"]

                if mask_area.size > 0:
                    t_h, t_w = mask_area.shape[:2]
                    crop_resized = crop_mask[:t_h, :t_w]
                    if crop_resized.shape != (t_h, t_w):
                        tmp = np.zeros((t_h, t_w), dtype=crop_mask.dtype)
                        tmp[:crop_resized.shape[0], :crop_resized.shape[1]] = crop_resized
                        crop_resized = tmp

                    mask_area[crop_resized > 0] = track["object"]["class_id"]
                    track["object"]["crop_mask"] = crop_resized

                predicted_tracks.append(track)

        self.__objects = predicted_tracks
        return final_mask, [t["object"] for t in predicted_tracks]

    def __process_mask(self, mask: np.ndarray, detected_objects: list[dict]) -> tuple[np.ndarray, list[dict]]:
        if self.__objects is None:
            self.__objects = []
            for obj in detected_objects:
                bx1, by1, bx2, by2 = obj["bbox"]
                self.__objects.append({
                    "confidence": 1.0,
                    "missing_frames": 0,
                    "cx": (bx1 + bx2) / 2,
                    "cy": (by1 + by2) / 2,
                    "w": bx2 - bx1,
                    "h": by2 - by1,
                    "vx": 0.0, "vy": 0.0, "vw": 0.0, "vh": 0.0,
                    "object": obj
                })
            return mask, detected_objects

        final_mask = mask.copy()
        updated_tracks = []
        matched_history_indices = set()

        for current in detected_objects:
            idx = self.__find_best_iou_idx(current["bbox"])

            bx1, by1, bx2, by2 = current["bbox"]
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            w, h = bx2 - bx1, by2 - by1

            if idx != -1:
                track = self.__objects[idx]
                matched_history_indices.add(idx)

                raw_vx = cx - track["cx"]
                raw_vy = cy - track["cy"]
                raw_vw = w - track["w"]
                raw_vh = h - track["h"]

                vx = (1 - self.__alpha_speed) * track["vx"] + self.__alpha_speed * raw_vx
                vy = (1 - self.__alpha_speed) * track["vy"] + self.__alpha_speed * raw_vy
                vw = (1 - self.__alpha_speed) * track["vw"] + self.__alpha_speed * raw_vw
                vh = (1 - self.__alpha_speed) * track["vh"] + self.__alpha_speed * raw_vh

                if current["class_id"] == track["object"]["class_id"]:
                    track["confidence"] = min(1.0, track["confidence"] + 0.2)
                else:
                    track["confidence"] -= 0.15

                    if track["confidence"] >= 0.4:
                        current["class_id"] = track["object"]["class_id"]
                    else:
                        track["object"]["class_id"] = current["class_id"]
                        track["confidence"] = 0.4

                track.update({
                    "confidence": track["confidence"],
                    "missing_frames": 0,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "vx": vx, "vy": vy, "vw": vw, "vh": vh,
                    "object": current
                })
                updated_tracks.append(track)
            else:
                new_track = {
                    "confidence": 1.0,
                    "missing_frames": 0,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "vx": 0.0, "vy": 0.0, "vw": 0.0, "vh": 0.0,
                    "object": current
                }
                updated_tracks.append(new_track)

        for idx, track in enumerate(self.__objects):
            if idx in matched_history_indices:
                continue

            track["missing_frames"] += 1
            track["confidence"] -= 0.15

            if track["missing_frames"] <= self.__max_missing_frames and track["confidence"] > 0.1:
                track["cx"] += track["vx"]
                track["cy"] += track["vy"]
                track["w"] = max(10, track["w"] + track["vw"])
                track["h"] = max(10, track["h"] + track["vh"])

                bx1 = int(track["cx"] - track["w"] / 2)
                by1 = int(track["cy"] - track["h"] / 2)
                bx2 = int(track["cx"] + track["w"] / 2)
                by2 = int(track["cy"] + track["h"] / 2)

                track["object"]["bbox"] = [bx1, by1, bx2, by2]

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
                        tmp[:crop_resized.shape, :crop_resized.shape] = crop_resized
                        crop_resized = tmp

                    mask_area[crop_resized > 0] = track["object"]["class_id"]
                    track["object"]["crop_mask"] = crop_resized

                updated_tracks.append(track)

        self.__objects = updated_tracks
        return final_mask, [t["object"] for t in updated_tracks]

    def __find_best_iou_idx(self, bbox: list[int]) -> int:
        best_idx = -1
        best_iou = 0

        for idx, prev in enumerate(self.__objects):
            iou = self.__calc_iou(bbox, prev["object"]["bbox"])

            if iou > best_iou and iou > self.__iou_threshold:
                best_idx = idx
                best_iou = iou

            if iou > 0.85 and prev["object"]["class_id"] == prev["object"]["class_id"]:
                return idx

        return best_idx
