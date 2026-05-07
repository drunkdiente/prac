from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

_CLS_PERSON = 0
_CLS_CHAIR = 56
_CLS_DINING_TABLE = 60

_DESK_CAPACITY = 2
_DESK_GAP_FACTOR = 2.5
_ROW_Y_FACTOR = 0.6
_DESK_VERTICAL_EXPAND = 1.6
_DESK_HORIZONTAL_EXPAND = 0.15


@dataclass
class Seat:
    index: int
    side: str
    box: tuple[int, int, int, int]
    occupied: bool = False
    person_box: Optional[tuple[int, int, int, int]] = None


@dataclass
class Desk:
    box: tuple[int, int, int, int]
    source: str
    seats: list[Seat] = field(default_factory=list)

    @property
    def occupied_count(self) -> int:
        return sum(1 for s in self.seats if s.occupied)

    @property
    def free_count(self) -> int:
        return len(self.seats) - self.occupied_count


@dataclass
class DetectionResult:
    desks: list[Desk]
    persons_detected: int
    total_seats: int
    occupied_seats: int
    free_seats: int
    annotated_image: np.ndarray


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _box_bottom_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, float(y2))


def _expand_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
    *,
    horizontal_factor: float,
    vertical_factor: float,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    return (
        max(0, x1 - int(box_w * horizontal_factor)),
        max(0, y1 - int(box_h * vertical_factor)),
        min(width - 1, x2 + int(box_w * horizontal_factor)),
        min(height - 1, y2 + int(box_h * 0.2)),
    )


def _make_two_seats(box: tuple[int, int, int, int]) -> list[Seat]:
    x1, y1, x2, y2 = box
    middle = (x1 + x2) // 2
    return [
        Seat(index=1, side="left", box=(x1, y1, middle, y2)),
        Seat(index=2, side="right", box=(middle, y1, x2, y2)),
    ]


def _build_desks_from_tables(tables: list[tuple[int, int, int, int]]) -> list[Desk]:
    desks: list[Desk] = []
    for table in sorted(tables, key=lambda b: (_box_center(b)[1], _box_center(b)[0])):
        desks.append(Desk(box=table, source="table", seats=_make_two_seats(table)))
    return desks


def _build_desks_from_chairs(chairs: list[tuple[int, int, int, int]]) -> list[Desk]:
    if len(chairs) < 2:
        return []

    chairs = sorted(chairs, key=lambda b: (_box_center(b)[1], _box_center(b)[0]))
    used = [False] * len(chairs)
    desks: list[Desk] = []

    for i, c1 in enumerate(chairs):
        if used[i]:
            continue
        cx1, cy1 = _box_center(c1)
        w1 = c1[2] - c1[0]
        h1 = c1[3] - c1[1]
        best_j = -1
        best_dist = float("inf")

        for j in range(i + 1, len(chairs)):
            if used[j]:
                continue
            c2 = chairs[j]
            cx2, cy2 = _box_center(c2)
            dy = abs(cy2 - cy1)
            dx = cx2 - cx1

            if dy > _ROW_Y_FACTOR * h1:
                continue
            if dx < 0 or dx > _DESK_GAP_FACTOR * w1:
                continue

            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j >= 0:
            used[i] = True
            used[best_j] = True
            c2 = chairs[best_j]
            left, right = (c1, c2) if _box_center(c1)[0] < _box_center(c2)[0] else (c2, c1)
            x1 = min(left[0], right[0])
            y1 = min(left[1], right[1])
            x2 = max(left[2], right[2])
            y2 = max(left[3], right[3])
            desk_box = (x1, y1, x2, y2)
            desks.append(Desk(box=desk_box, source="chairs", seats=_make_two_seats(desk_box)))

    return desks


def _assign_people_to_seats(
    desks: list[Desk],
    persons: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int, int],
) -> None:
    slots: list[tuple[Desk, Seat, tuple[int, int, int, int]]] = []
    for desk in desks:
        for seat in desk.seats:
            expanded = _expand_box(
                seat.box,
                image_shape,
                horizontal_factor=_DESK_HORIZONTAL_EXPAND,
                vertical_factor=_DESK_VERTICAL_EXPAND,
            )
            slots.append((desk, seat, expanded))

    scored: list[tuple[float, int, int]] = []
    for pi, person in enumerate(persons):
        pcx, pcy = _box_bottom_center(person)
        for si, (_, _, exp_box) in enumerate(slots):
            ex1, ey1, ex2, ey2 = exp_box
            if ex1 <= pcx <= ex2 and ey1 <= pcy <= ey2:
                scx, scy = _box_center(exp_box)
                scored.append((math.hypot(pcx - scx, pcy - scy), pi, si))

    scored.sort(key=lambda t: t[0])

    assigned_persons: set[int] = set()
    assigned_slots: set[int] = set()
    for _, pi, si in scored:
        if pi in assigned_persons or si in assigned_slots:
            continue
        _, seat, _ = slots[si]
        seat.occupied = True
        seat.person_box = persons[pi]
        assigned_persons.add(pi)
        assigned_slots.add(si)


def _draw_annotations(
    img: np.ndarray,
    desks: list[Desk],
    persons: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = {
        "free": (0, 200, 0),
        "taken": (0, 0, 220),
        "person": (255, 140, 0),
        "desk_label": (255, 255, 255),
    }

    for di, desk in enumerate(desks):
        x1, y1, x2, y2 = desk.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 50), 2)
        cv2.putText(
            out,
            f"Desk {di + 1}",
            (x1, max(16, y1 - 6)),
            font,
            0.55,
            colors["desk_label"],
            2,
            cv2.LINE_AA,
        )

        for seat in desk.seats:
            color = colors["taken"] if seat.occupied else colors["free"]
            sx1, sy1, sx2, sy2 = seat.box
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), color, 2)
            label = "Occupied" if seat.occupied else "Free"
            cv2.putText(out, label, (sx1, sy2 + 16), font, 0.45, color, 1, cv2.LINE_AA)

    for box in persons:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), colors["person"], 2)

    return out


class ClassroomDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self._model = YOLO(model_path)

    def process(self, image_bytes: bytes) -> DetectionResult:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")

        results = self._model(img, verbose=False)[0]

        persons: list[tuple[int, int, int, int]] = []
        chairs: list[tuple[int, int, int, int]] = []
        tables: list[tuple[int, int, int, int]] = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == _CLS_PERSON and conf >= 0.30:
                persons.append((x1, y1, x2, y2))
            elif cls == _CLS_CHAIR and conf >= 0.25:
                chairs.append((x1, y1, x2, y2))
            elif cls == _CLS_DINING_TABLE and conf >= 0.20:
                tables.append((x1, y1, x2, y2))

        desks = _build_desks_from_tables(tables)
        if not desks:
            desks = _build_desks_from_chairs(chairs)

        _assign_people_to_seats(desks, persons, img.shape)

        total = sum(len(d.seats) for d in desks)
        occupied = sum(d.occupied_count for d in desks)
        free = max(0, total - occupied)

        annotated = _draw_annotations(img, desks, persons)

        return DetectionResult(
            desks=desks,
            persons_detected=len(persons),
            total_seats=total,
            occupied_seats=occupied,
            free_seats=free,
            annotated_image=annotated,
        )
