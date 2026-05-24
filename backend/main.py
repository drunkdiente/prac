from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .detector import ClassroomDetector, DetectionResult
except ImportError:
    from backend.detector import ClassroomDetector, DetectionResult


def _get_detector():
    _detector_type = os.getenv("DETECTOR_TYPE", "yolo").lower()
    if _detector_type == "gdino":
        try:
            from .gdino_detector import GroundingDinoDetector
        except ImportError:
            from backend.gdino_detector import GroundingDinoDetector
        return GroundingDinoDetector()
    return ClassroomDetector(model_path=os.getenv("YOLO_MODEL", "yolo11m.pt"))


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Classroom Occupancy Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = _get_detector()


def _box_to_dict(box: tuple[int, int, int, int]) -> dict[str, int]:
    x1, y1, x2, y2 = box
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _result_to_response(result: DetectionResult) -> dict:
    ok, buffer = cv2.imencode(".jpg", result.annotated_image)
    if not ok:
        raise HTTPException(status_code=500, detail="Cannot encode annotated image")

    desks = []
    for index, desk in enumerate(result.desks, start=1):
        desks.append(
            {
                "index": index,
                "source": desk.source,
                "box": _box_to_dict(desk.box),
                "occupied_count": desk.occupied_count,
                "free_count": desk.free_count,
                "seats": [
                    {
                        "index": seat.index,
                        "side": seat.side,
                        "occupied": seat.occupied,
                        "box": _box_to_dict(seat.box),
                        "person_box": _box_to_dict(seat.person_box)
                        if seat.person_box
                        else None,
                    }
                    for seat in desk.seats
                ],
            }
        )

    image_base64 = base64.b64encode(buffer).decode("ascii")
    return {
        "persons_detected": result.persons_detected,
        "desks_detected": len(result.desks),
        "total_seats": result.total_seats,
        "occupied_seats": result.occupied_seats,
        "free_seats": result.free_seats,
        "desks": desks,
        "annotated_image": f"data:image/jpeg;base64,{image_base64}",
    }


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = detector.process(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _result_to_response(result)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
