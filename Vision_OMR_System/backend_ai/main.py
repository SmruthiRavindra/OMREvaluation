"""
main.py
-------
FastAPI entry point for the Vision-Based OMR Evaluation Engine.

Pipeline:
  1. Receive image upload (JPEG / PNG)
  2. Preprocess  → bilateral filter + perspective warp
  3. Localise    → YOLOv8 inference + class-aware NMS
  4. Classify    → secondary threshold verification per bubble
  5. Aggregate   → build structured response with counts, USN, timings
"""

import time
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.preprocess import preprocess_image
from core.localization import run_yolo_inference
from core.classification import classify_all, BubbleState
from core.scoring import (
    score_sheet,
    SheetLayout,
    ScoreReport as _ScoreReport,
    AnswerStatus,
)

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vision-Based OMR Evaluation Engine",
    version="2.0.0",
    description="End-to-end OMR sheet processing: image cleanup → bubble detection → grading.",
)

# ── In-memory answer key store (keyed by session_id) ────────────────────────
# In production, persist this in PostgreSQL via the data gateway.
_answer_keys: Dict[str, Dict[int, str]] = {}


# ── Response schemas ────────────────────────────────────────────────────────

class BubbleResult(BaseModel):
    """Per-bubble classification result."""
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    state: str           # filled | empty | ambiguous
    fill_ratio: float
    needs_review: bool


class EvaluationResponse(BaseModel):
    """Full evaluation result returned to the client."""
    usn: Optional[str] = None
    filled_count: int
    empty_count: int
    ambiguous_count: int
    needs_manual_review: bool
    bubbles: List[BubbleResult]
    processing_time_ms: int


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "backend-ai"}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_sheet(file: UploadFile = File(...)):
    """
    Run the full OMR evaluation pipeline on an uploaded sheet image.
    """
    # ── Validate ────────────────────────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid payload. File must be an image (JPEG/PNG).",
        )

    try:
        t_start = time.perf_counter()

        # ── 1. Read raw bytes ───────────────────────────────────────────────
        contents = await file.read()

        # ── 2. Preprocess: denoise + perspective warp ───────────────────────
        clean_img = preprocess_image(contents)

        # ── 3. Localise: YOLOv8 + custom class-aware NMS ───────────────────
        detections = run_yolo_inference(clean_img)

        # ── 4. Separate USN detections from bubble detections ───────────────
        usn_detections = [d for d in detections if d.class_name == "usn"]
        bubble_detections = [d for d in detections if d.class_name != "usn"]

        # Extract USN region (placeholder — OCR integration is a future step)
        usn_value = None
        if usn_detections:
            # For now, mark that a USN box was detected.
            # Full OCR (e.g. Tesseract / EasyOCR) would crop this region
            # and read the student ID. That's a future enhancement.
            usn_value = "DETECTED"

        # ── 5. Classify: secondary pixel-ratio verification ─────────────────
        classifications = classify_all(clean_img, bubble_detections)

        # ── 6. Aggregate results ────────────────────────────────────────────
        filled_count = sum(
            1 for c in classifications if c.state == BubbleState.FILLED
        )
        empty_count = sum(
            1 for c in classifications if c.state == BubbleState.EMPTY
        )
        ambiguous_count = sum(
            1 for c in classifications if c.state == BubbleState.AMBIGUOUS
        )
        needs_manual_review = ambiguous_count > 0

        bubbles = [
            BubbleResult(
                bbox=list(c.detection.bbox),
                confidence=round(c.detection.confidence, 4),
                class_id=c.detection.class_id,
                class_name=c.detection.class_name,
                state=c.state.value,
                fill_ratio=round(c.fill_ratio, 4),
                needs_review=c.needs_review,
            )
            for c in classifications
        ]

        t_end = time.perf_counter()
        processing_time_ms = int((t_end - t_start) * 1000)

        return EvaluationResponse(
            usn=usn_value,
            filled_count=filled_count,
            empty_count=empty_count,
            ambiguous_count=ambiguous_count,
            needs_manual_review=needs_manual_review,
            bubbles=bubbles,
            processing_time_ms=processing_time_ms,
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline execution failure: {str(e)}",
        )


# ── Answer Key management ───────────────────────────────────────────────────

class AnswerKeyRequest(BaseModel):
    """Upload correct answers for a session."""
    session_id: str
    answers: Dict[int, str]  # { 1: "A", 2: "C", 3: "B", ... }


class AnswerKeyResponse(BaseModel):
    session_id: str
    total_questions: int
    saved: bool


@app.post("/answer-key", response_model=AnswerKeyResponse)
async def upload_answer_key(req: AnswerKeyRequest):
    """
    Register the correct answer key for a given exam session.

    The key is stored in memory and used by the /score endpoint
    to grade evaluated sheets.
    """
    _answer_keys[req.session_id] = req.answers
    return AnswerKeyResponse(
        session_id=req.session_id,
        total_questions=len(req.answers),
        saved=True,
    )


@app.get("/answer-key/{session_id}")
async def get_answer_key(session_id: str):
    """Retrieve the stored answer key for a session."""
    key = _answer_keys.get(session_id)
    if key is None:
        raise HTTPException(status_code=404, detail=f"No answer key for session '{session_id}'.")
    return {"session_id": session_id, "answers": key, "total_questions": len(key)}


# ── Scoring endpoint ────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """Score a sheet against its session's answer key."""
    session_id: str
    questions_per_column: int = 30
    num_columns: int = 1
    options: str = "ABCD"


class QuestionResultResponse(BaseModel):
    question_number: int
    marked_options: List[str]
    correct_option: Optional[str]
    status: str
    has_ambiguous: bool


class ScoreResponse(BaseModel):
    total_questions: int
    answered: int
    correct: int
    incorrect: int
    unanswered: int
    multiple_marked: int
    ambiguous: int
    score_percent: float
    per_question: List[QuestionResultResponse]


@app.post("/score", response_model=ScoreResponse)
async def score_evaluated_sheet(
    file: UploadFile = File(...),
    session_id: str = "default",
    questions_per_column: int = 30,
    num_columns: int = 1,
    options: str = "ABCD",
):
    """
    Full pipeline + scoring in one call.

    Runs the evaluation pipeline and then scores the results against
    the answer key stored for the given session_id.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    answer_key = _answer_keys.get(session_id)

    try:
        contents = await file.read()
        clean_img = preprocess_image(contents)
        detections = run_yolo_inference(clean_img)
        bubble_detections = [d for d in detections if d.class_name != "usn"]
        classifications = classify_all(clean_img, bubble_detections)

        layout = SheetLayout(
            questions_per_column=questions_per_column,
            num_columns=num_columns,
            options=options,
        )

        report = score_sheet(classifications, answer_key, layout)

        return ScoreResponse(
            total_questions=report.total_questions,
            answered=report.answered,
            correct=report.correct,
            incorrect=report.incorrect,
            unanswered=report.unanswered,
            multiple_marked=report.multiple_marked,
            ambiguous=report.ambiguous,
            score_percent=report.score_percent,
            per_question=[
                QuestionResultResponse(
                    question_number=q.question_number,
                    marked_options=q.marked_options,
                    correct_option=q.correct_option,
                    status=q.status.value,
                    has_ambiguous=q.has_ambiguous,
                )
                for q in report.per_question
            ],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scoring pipeline failure: {str(e)}",
        )


# ── Debug / Visual Testing ──────────────────────────────────────────────────

# Color map for visualization
_VIZ_COLORS = {
    "filled":    (34, 197, 94),    # green
    "empty":     (148, 163, 184),  # gray
    "ambiguous": (245, 158, 11),   # amber
    "usn":       (99, 102, 241),   # indigo
}


def _annotate_image(
    image: np.ndarray,
    detections,
    classifications=None,
) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    annotated = image.copy()
    cls_map = {}
    if classifications:
        for cr in classifications:
            key = (int(cr.detection.x1), int(cr.detection.y1),
                   int(cr.detection.x2), int(cr.detection.y2))
            cls_map[key] = cr

    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        key = (x1, y1, x2, y2)

        # Determine color and label
        if det.class_name == "usn":
            color = _VIZ_COLORS["usn"]
            label = f"USN {det.confidence:.0%}"
        elif key in cls_map:
            cr = cls_map[key]
            color = _VIZ_COLORS.get(cr.state.value, (255, 255, 255))
            label = f"{cr.state.value} {cr.fill_ratio:.0%}"
        else:
            color = _VIZ_COLORS.get(det.class_name, (255, 255, 255))
            label = f"{det.class_name} {det.confidence:.0%}"

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return annotated


def _img_to_base64(image: np.ndarray) -> str:
    """Encode a BGR image as base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


class DebugResponse(BaseModel):
    """Debug endpoint response with annotated image and full results."""
    original_image_b64: str
    preprocessed_image_b64: str
    annotated_image_b64: str
    total_detections: int
    bubble_detections: int
    usn_detections: int
    filled_count: int
    empty_count: int
    ambiguous_count: int
    needs_manual_review: bool
    processing_time_ms: int
    bubbles: List[BubbleResult]


@app.post("/debug/evaluate")
async def debug_evaluate(file: UploadFile = File(...)):
    """
    Debug endpoint: runs the full pipeline and returns annotated images
    alongside the JSON results. Used for visual verification.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        t_start = time.perf_counter()
        contents = await file.read()

        # Decode original
        orig_arr = np.frombuffer(contents, np.uint8)
        orig_img = cv2.imdecode(orig_arr, cv2.IMREAD_COLOR)

        # Preprocess
        clean_img = preprocess_image(contents)

        # Localize
        detections = run_yolo_inference(clean_img)
        usn_dets = [d for d in detections if d.class_name == "usn"]
        bubble_dets = [d for d in detections if d.class_name != "usn"]

        # Classify
        classifications = classify_all(clean_img, bubble_dets)

        # Annotate
        annotated = _annotate_image(clean_img, detections, classifications)

        # Counts
        filled = sum(1 for c in classifications if c.state == BubbleState.FILLED)
        empty = sum(1 for c in classifications if c.state == BubbleState.EMPTY)
        ambig = sum(1 for c in classifications if c.state == BubbleState.AMBIGUOUS)

        t_end = time.perf_counter()

        bubbles = [
            BubbleResult(
                bbox=list(c.detection.bbox),
                confidence=round(c.detection.confidence, 4),
                class_id=c.detection.class_id,
                class_name=c.detection.class_name,
                state=c.state.value,
                fill_ratio=round(c.fill_ratio, 4),
                needs_review=c.needs_review,
            )
            for c in classifications
        ]

        return {
            "original_image_b64": _img_to_base64(orig_img) if orig_img is not None else "",
            "preprocessed_image_b64": _img_to_base64(clean_img),
            "annotated_image_b64": _img_to_base64(annotated),
            "total_detections": len(detections),
            "bubble_detections": len(bubble_dets),
            "usn_detections": len(usn_dets),
            "filled_count": filled,
            "empty_count": empty,
            "ambiguous_count": ambig,
            "needs_manual_review": ambig > 0,
            "processing_time_ms": int((t_end - t_start) * 1000),
            "bubbles": [b.model_dump() for b in bubbles],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Web Dashboard ────────────────────────────────────────────────────────────

# Serve the static dashboard
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the testing dashboard HTML page."""
    html_path = _STATIC_DIR / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found. Create backend_ai/static/dashboard.html")
    return html_path.read_text(encoding="utf-8")