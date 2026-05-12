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
_DESK_GAP_FACTOR = 3.0
_ROW_Y_FACTOR = 0.7
_DESK_VERTICAL_EXPAND = 0.8
_DESK_HORIZONTAL_EXPAND = 0.25

# Фильтрация dining-table боксов (парты шире, чем выше)
_MIN_DESK_ASPECT = 1.0
_MAX_DESK_ASPECT = 3.5
_MIN_DESK_SIZE_RATIO = 0.03
_MAX_DESK_SIZE_RATIO = 0.50


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _is_valid_desk_box(box: tuple[int, int, int, int], img_shape: tuple[int, ...]) -> bool:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    aspect = w / h
    if not (_MIN_DESK_ASPECT <= aspect <= _MAX_DESK_ASPECT):
        return False
    img_h, img_w = img_shape[:2]
    min_side = min(img_w, img_h)
    min_size = min_side * _MIN_DESK_SIZE_RATIO
    max_size = min_side * _MAX_DESK_SIZE_RATIO
    if w < min_size or h < min_size:
        return False
    if w > max_size or h > max_size:
        return False
    return True


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.45
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        found = False
        for i, m in enumerate(merged):
            if _iou(box, m) > iou_threshold:
                merged[i] = (
                    min(m[0], box[0]),
                    min(m[1], box[1]),
                    max(m[2], box[2]),
                    max(m[3], box[3]),
                )
                found = True
                break
        if not found:
            merged.append(box)
    # итеративно до стабилизации
    for _ in range(3):
        prev = merged
        merged = []
        for box in prev:
            found = False
            for i, m in enumerate(merged):
                if _iou(box, m) > iou_threshold:
                    merged[i] = (
                        min(m[0], box[0]),
                        min(m[1], box[1]),
                        max(m[2], box[2]),
                        max(m[3], box[3]),
                    )
                    found = True
                    break
            if not found:
                merged.append(box)
        if len(merged) == len(prev):
            break
    return merged


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
    # расширяем вниз (куда сидит человек) сильнее, чем вверх
    return (
        max(0, x1 - int(box_w * horizontal_factor)),
        max(0, y1 - int(box_h * vertical_factor * 0.3)),
        min(width - 1, x2 + int(box_w * horizontal_factor)),
        min(height - 1, y2 + int(box_h * vertical_factor)),
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

    # Fallback: человек не попал в expanded место, но его bottom-center
    # внутри bbox парты или перекрывается с ней — привязываем к ближайшему свободному
    for pi, person in enumerate(persons):
        if pi in assigned_persons:
            continue
        px, py = _box_bottom_center(person)
        best_si = -1
        best_dist = float("inf")
        for si, (desk, seat, _) in enumerate(slots):
            if si in assigned_slots:
                continue
            dx1, dy1, dx2, dy2 = desk.box
            in_desk = dx1 <= px <= dx2 and dy1 <= py <= dy2
            overlap = _iou(person, desk.box) > 0.05
            if in_desk or overlap:
                scx, scy = _box_center(seat.box)
                dist = math.hypot(px - scx, py - scy)
                if dist < best_dist:
                    best_dist = dist
                    best_si = si
        if best_si >= 0:
            _, seat, _ = slots[best_si]
            seat.occupied = True
            seat.person_box = person
            assigned_persons.add(pi)
            assigned_slots.add(best_si)


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)


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
        "desk_table": (200, 200, 50),
        "desk_chairs": (50, 150, 200),
    }

    for di, desk in enumerate(desks):
        x1, y1, x2, y2 = desk.box
        desk_color = colors["desk_table"] if desk.source == "table" else colors["desk_chairs"]
        cv2.rectangle(out, (x1, y1), (x2, y2), desk_color, 2)
        _draw_text_with_bg(
            out,
            f"Desk {di + 1}",
            (x1, max(20, y1 - 4)),
            font,
            0.55,
            colors["desk_label"],
            2,
            bg_color=(0, 0, 0),
        )

        for seat in desk.seats:
            color = colors["taken"] if seat.occupied else colors["free"]
            sx1, sy1, sx2, sy2 = seat.box
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), color, 2)
            label = "Occupied" if seat.occupied else "Free"
            _draw_text_with_bg(
                out,
                label,
                (sx1, min(img.shape[0] - 4, sy2 + 14)),
                font,
                0.45,
                color,
                1,
                bg_color=(0, 0, 0),
            )

    for box in persons:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), colors["person"], 2)

    return out


class ClassroomDetector:
    def __init__(self, model_path: str = "yolo11m.pt"):
        self._model = YOLO(model_path)

    def process(self, image_bytes: bytes) -> DetectionResult:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")

        results = self._model(img, verbose=False, imgsz=1280, conf=0.20, iou=0.45)[0]

        persons: list[tuple[int, int, int, int]] = []
        chairs: list[tuple[int, int, int, int]] = []
        tables: list[tuple[int, int, int, int]] = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == _CLS_PERSON and conf >= 0.25:
                persons.append((x1, y1, x2, y2))
            elif cls == _CLS_CHAIR and conf >= 0.25:
                chairs.append((x1, y1, x2, y2))
            elif cls == _CLS_DINING_TABLE and conf >= 0.15:
                tables.append((x1, y1, x2, y2))

        # Фильтрация + merge dining tables
        tables = [t for t in tables if _is_valid_desk_box(t, img.shape)]
        tables = _merge_overlapping_boxes(tables, iou_threshold=0.45)
        tables = [t for t in tables if _is_valid_desk_box(t, img.shape)]

        table_desks = _build_desks_from_tables(tables)
        chair_desks = _build_desks_from_chairs(chairs)

        # Комбинируем: стулья добавляем только если не перекрываются со столами
        desks = list(table_desks)
        existing_boxes = [d.box for d in desks]
        for cd in chair_desks:
            if not any(_iou(cd.box, eb) > 0.3 for eb in existing_boxes):
                desks.append(cd)

        # Если столов практически нет — полный fallback на стулья
        if len(table_desks) < 2 and len(chair_desks) >= 2:
            desks = chair_desks

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
