from __future__ import annotations

import io
import warnings
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from detector import (
    DetectionResult,
    Desk,
    Seat,
    _assign_people_to_seats,
    _build_desks_from_chairs,
    _build_desks_from_tables,
    _draw_annotations,
    _is_valid_desk_box,
    _iou,
    _merge_overlapping_boxes,
)

_PERSON_QUERIES = ["student person", "person"]
_DESK_QUERIES = ["school desk", "wooden desk", "classroom desk", "desk"]
_CHAIR_QUERIES = ["chair", "school chair", "wooden chair"]


class GroundingDinoDetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
                self.device
            )
        self.model.eval()

    def _label_matches(self, label: str, queries: list[str]) -> bool:
        label_lower = label.lower()
        return any(q in label_lower for q in queries)

    def process(self, image_bytes: bytes) -> DetectionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        text = " . ".join(_PERSON_QUERIES + _DESK_QUERIES + _CHAIR_QUERIES)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[(image.height, image.width)],
        )[0]

        persons: list[tuple[int, int, int, int]] = []
        chairs: list[tuple[int, int, int, int]] = []
        tables: list[tuple[int, int, int, int]] = []

        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            if score < 0.20:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            if self._label_matches(label, _PERSON_QUERIES):
                persons.append((x1, y1, x2, y2))
            elif self._label_matches(label, _DESK_QUERIES):
                tables.append((x1, y1, x2, y2))
            elif self._label_matches(label, _CHAIR_QUERIES):
                chairs.append((x1, y1, x2, y2))

        tables = [t for t in tables if _is_valid_desk_box(t, img_np.shape)]
        tables = _merge_overlapping_boxes(tables, iou_threshold=0.45)
        tables = [t for t in tables if _is_valid_desk_box(t, img_np.shape)]

        table_desks = _build_desks_from_tables(tables)
        chair_desks = _build_desks_from_chairs(chairs)

        desks = list(table_desks)
        existing_boxes = [d.box for d in desks]
        for cd in chair_desks:
            if not any(_iou(cd.box, eb) > 0.3 for eb in existing_boxes):
                desks.append(cd)

        if len(table_desks) < 2 and len(chair_desks) >= 2:
            desks = chair_desks

        _assign_people_to_seats(desks, persons, img_np.shape)

        total = sum(len(d.seats) for d in desks)
        occupied = sum(d.occupied_count for d in desks)
        free = max(0, total - occupied)

        annotated = _draw_annotations(img_np, desks, persons)

        return DetectionResult(
            desks=desks,
            persons_detected=len(persons),
            total_seats=total,
            occupied_seats=occupied,
            free_seats=free,
            annotated_image=annotated,
        )
