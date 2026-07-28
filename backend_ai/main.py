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
import os
import json
import base64
import io
import uuid
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Worker thread count for concurrent batch sheet processing
_BATCH_MAX_WORKERS: int = int(os.getenv("BATCH_MAX_WORKERS", "4"))

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    RealDictCursor = None

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.preprocess import preprocess_image, preprocess_image_detect
from core.localization import run_yolo_inference
from core.classification import classify_all, BubbleState
from core import extract_usn_from_roi
from core.pdf_parser import extract_pages_from_pdf

# Read once at startup. Set OMR_DEBUG_DUMP=true to enable disk writes.
OMR_DEBUG_DUMP: bool = os.getenv("OMR_DEBUG_DUMP", "false").lower() in ("1", "true", "yes")

from core.scoring import (
    score_sheet,
    SheetLayout,
    ScoreReport as _ScoreReport,
    AnswerStatus,
    map_bubbles_to_grid,
)

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vision-Based OMR Evaluation Engine",
    version="2.0.0",
    description="End-to-end OMR sheet processing: image cleanup → bubble detection → grading.",
)

def get_db_connection():
    if not HAS_PSYCOPG2:
        return None
    try:
        return psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", 5432)),
            dbname=os.getenv("PGDATABASE", "omr_db"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres")
        )
    except Exception:
        return None

def get_answer_key_for_session(session_id: Optional[str], version: Optional[str] = None) -> Optional[Dict[int, str]]:
    if not session_id:
        return None
    ver = (version or "DEFAULT").upper()
    conn = get_db_connection()
    if conn is not None:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT version, answers FROM exam_versions WHERE session_id = %s",
                    (session_id,)
                )
                rows = cur.fetchall()
            conn.close()

            if rows:
                # Look for explicit version match
                for r in rows:
                    if r['version'].upper() == ver:
                        raw_ans = r['answers']
                        if isinstance(raw_ans, str):
                            raw_ans = json.loads(raw_ans)
                        return {int(k): str(v).upper() for k, v in raw_ans.items()}

                # Fallback to DEFAULT or first available version
                for r in rows:
                    if r['version'].upper() in ("DEFAULT", "A"):
                        raw_ans = r['answers']
                        if isinstance(raw_ans, str):
                            raw_ans = json.loads(raw_ans)
                        return {int(k): str(v).upper() for k, v in raw_ans.items()}

                first_ans = rows[0]['answers']
                if isinstance(first_ans, str):
                    first_ans = json.loads(first_ans)
                return {int(k): str(v).upper() for k, v in first_ans.items()}
        except Exception as e:
            print(f"[DB Error get_answer_key_for_session]: {e}")

    # Fallback to in-memory store
    if session_id in _fallback_answer_keys:
        session_val = _fallback_answer_keys[session_id]
        if ver in session_val:
            return session_val[ver]
        if "DEFAULT" in session_val:
            return session_val["DEFAULT"]
        if session_val:
            return next(iter(session_val.values()))
    return None



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
    image_resolution: Optional[str] = None
    warp_status: Optional[str] = None
    is_warped: bool = False
    score_report: Optional[dict] = None

class BatchEvaluationResponse(BaseModel):
    """Response array for a multi-page PDF batch."""
    total_pages: int
    results: List[dict]


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "backend-ai"}


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def match_usn_against_roster(ocr_usn: str, roster_list: List[str]) -> str:
    if not ocr_usn or not roster_list:
        return ocr_usn
    clean_ocr = ocr_usn.strip().upper()
    best_match = None
    min_dist = 9999
    for registered_usn in roster_list:
        clean_reg = registered_usn.strip().upper()
        if clean_ocr == clean_reg:
            return registered_usn
        dist = levenshtein_distance(clean_ocr, clean_reg)
        if dist < min_dist:
            min_dist = dist
            best_match = registered_usn
    if best_match and min_dist <= 5:
        return best_match
    return ocr_usn


# ---------------------------------------------------------------------------
# Input Quality Guards (TC#2 / TC#3 / TC#19)
# ---------------------------------------------------------------------------

# TC#2 threshold: trip if bubble count < fraction × expected total bubbles.
# For a 15×2×4 sheet (120 total), trips below 30 detections.
# NOTE: recalibrate once a TC#21 "best-case" baseline detection count is known.
_MIN_BUBBLE_FRACTION: float = 0.25

# TC#3 threshold: trip if detected question rows < fraction × questions_per_column.
# For 15 rows/column, trips below ~8 rows (half-page crop).
_MIN_ROW_FRACTION: float = 0.50


def _check_bubble_coverage(
    bubble_dets: list,
    layout: "SheetLayout",
    ambiguous_capture: bool = False,
) -> dict | None:
    """
    Returns a rejection dict if the capture is unusable, otherwise None.

    Three cases handled:
      TC#19 — ambiguous_capture=True   → multiple sheets in frame
      TC#2  — too few total detections  → blank page / wrong document
      TC#3  — too few question rows     → half-page / cropped capture

    The function is a pure probe — it does not modify any image data or
    alter any state.  The main pipeline continues normally when None is
    returned.
    """
    # TC#19 — multiple sheets in one photo (checked first, highest priority)
    if ambiguous_capture:
        return {
            "rejected": True,
            "reason": "multiple_sheets_detected",
            "detail": (
                "More than one sheet outline was detected in the image. "
                "Please photograph a single OMR sheet at a time."
            ),
        }

    expected_total = layout.total_bubbles   # e.g. 15 q × 2 cols × 4 opts = 120
    actual_count   = len(bubble_dets)

    # TC#2 — near-zero detections (blank page / wrong document)
    if actual_count < _MIN_BUBBLE_FRACTION * expected_total:
        return {
            "rejected": True,
            "reason": "insufficient_detections",
            "detail": (
                f"Only {actual_count} bubble(s) detected "
                f"(expected ≥ {int(_MIN_BUBBLE_FRACTION * expected_total)}). "
                "Ensure the OMR sheet fills the camera frame."
            ),
        }

    # TC#3 — half-page / cropped capture: check unique question rows
    if layout.questions_per_column > 0 and actual_count > 0:
        y_centres = sorted((d.y1 + d.y2) / 2 for d in bubble_dets)
        # Estimate median bubble height for adaptive row-gap threshold
        heights = sorted(d.y2 - d.y1 for d in bubble_dets)
        med_h = heights[len(heights) // 2] if heights else 20.0
        row_gap = max(med_h * 0.5, 6.0)
        rows = 1
        for prev, cur in zip(y_centres, y_centres[1:]):
            if cur - prev > row_gap:
                rows += 1
        min_rows = max(1, int(_MIN_ROW_FRACTION * layout.questions_per_column))
        if rows < min_rows:
            return {
                "rejected": True,
                "reason": "incomplete_sheet",
                "detail": (
                    f"Only {rows} question row(s) detected "
                    f"(expected ≥ {min_rows}). "
                    "Ensure the full OMR sheet is visible and not cropped."
                ),
            }

    return None  # sheet looks plausible — continue normal pipeline


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_sheet(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form("default"),
    version: Optional[str] = Form("DEFAULT"),
    roster: Optional[str] = Form(None),
    assigned_usn: Optional[str] = Form(None)
):
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
        import json
        roster_list = json.loads(roster) if roster else []

        # ── 1. Read raw bytes ───────────────────────────────────────────────
        contents = await file.read()
        orig_arr = np.frombuffer(contents, np.uint8)
        orig_img = cv2.imdecode(orig_arr, cv2.IMREAD_COLOR)
        h_orig, w_orig = (orig_img.shape[0], orig_img.shape[1]) if orig_img is not None else (0, 0)
        image_res = f"{w_orig}x{h_orig}" if w_orig and h_orig else None

        # ── 2. Preprocess: denoise + perspective warp ───────────────────────
        #    Use preprocess_image_detect to also get the multi-sheet flag (TC#19)
        clean_img, is_warped, multi_sheet = preprocess_image_detect(contents)
        warp_status = "WARPED_SUCCESS" if is_warped else "UNWARPED_FALLBACK"

        # ── 3. Localise: YOLOv8 + custom class-aware NMS ───────────────────
        detections = run_yolo_inference(clean_img)

        # Check for upside-down orientation using USN bounding box
        usn_detections = [d for d in detections if d.class_name == "usn"]
        if usn_detections:
            usn_det = usn_detections[0]
            h_img = clean_img.shape[0]
            usn_center_y = (usn_det.y1 + usn_det.y2) / 2
            if usn_center_y > h_img / 2:
                clean_img = cv2.rotate(clean_img, cv2.ROTATE_180)
                detections = run_yolo_inference(clean_img)
                usn_detections = [d for d in detections if d.class_name == "usn"]

        # ── 4. Separate USN detections from bubble detections ───────────────
        bubble_detections = [d for d in detections if d.class_name != "usn"]

        # ── Input Quality Guard (TC#2 / TC#3 / TC#19) ──────────────────────
        layout_for_guard = SheetLayout(
            questions_per_column=15, num_columns=2, options="ABCD"
        )
        rejection = _check_bubble_coverage(
            bubble_detections, layout_for_guard, ambiguous_capture=multi_sheet
        )
        if rejection:
            return JSONResponse(status_code=422, content=rejection)

        # Extract USN region using OCR
        usn_value = assigned_usn
        if not usn_value and usn_detections:
            det = usn_detections[0]
            usn_value = extract_usn_from_roi(clean_img, det.x1, det.y1, det.x2, det.y2)
            if usn_value and roster_list:
                usn_value = match_usn_against_roster(usn_value, roster_list)


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

        # ── 7. Automatic Scoring against session Answer Key ─────────────────
        score_report_dict = None
        answer_key = get_answer_key_for_session(session_id, version) if session_id else None
        if not answer_key and session_id:
            answer_key = get_answer_key_for_session(session_id, "DEFAULT")
        if answer_key:
            layout = SheetLayout(
                questions_per_column=15,
                num_columns=2,
                options="ABCD",
            )
            usn_y2 = usn_detections[0].y2 if usn_detections else None
            report = score_sheet(classifications, answer_key, layout, usn_y2=usn_y2)
            score_report_dict = {
                "total_questions": report.total_questions,
                "answered": report.answered,
                "correct": report.correct,
                "incorrect": report.incorrect,
                "unanswered": report.unanswered,
                "multiple_marked": report.multiple_marked,
                "ambiguous": report.ambiguous,
                "score_percent": report.score_percent,
                "per_question": [
                    {
                        "question_number": q.question_number,
                        "marked_options": q.marked_options,
                        "correct_option": q.correct_option,
                        "status": q.status.value,
                        "has_ambiguous": q.has_ambiguous,
                    }
                    for q in report.per_question
                ]
            }

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
            image_resolution=image_res,
            warp_status=warp_status,
            is_warped=is_warped,
            score_report=score_report_dict,
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

# ── Batch Endpoint ──────────────────────────────────────────────────────────

@app.post("/evaluate-batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form("default"),
    version: Optional[str] = Form("DEFAULT"),
    questions_per_column: int = Form(15),
    num_columns: int = Form(2),
    options: str = Form("ABCD")
):
    """
    Accepts either an image or a multi-page PDF.
    Extracts pages and runs the pipeline on each sequentially.
    """
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Missing Content-Type")

    try:
        contents = await file.read()
        images = []
        
        # 1. Parse File
        if file.content_type == "application/pdf":
            images = extract_pages_from_pdf(contents, dpi=150, max_pages=100)
            if not images:
                raise ValueError("Could not extract any pages from PDF.")
        elif file.content_type.startswith("image/"):
            img_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image.")
            images = [img]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF or JPEG/PNG.")

        # 2. Process Each Page using _process_single_sheet
        batch_results = []
        answer_key = get_answer_key_for_session(session_id or "default", version)
        layout = SheetLayout(
            questions_per_column=questions_per_column,
            num_columns=num_columns,
            options=options,
        )

        for i, img in enumerate(images):
            _, buffer = cv2.imencode(".jpg", img)
            page_fname = f"{file.filename}_page_{i+1}.jpg" if file.filename else f"page_{i+1}.jpg"
            res = _process_single_sheet(
                idx=i,
                filename=page_fname,
                content=buffer.tobytes(),
                answer_key=answer_key,
                layout=layout,
                assigned_usns=None,
                roster_list=[],
                version=version,
            )
            res.pop("_idx", None)
            res["page_index"] = i + 1
            batch_results.append(res)

        return BatchEvaluationResponse(
            total_pages=len(batch_results),
            results=batch_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# ── Async Batch Ingestion & Polling ─────────────────────────────────────────

_tasks: Dict[str, dict] = {}
_tasks_lock = threading.Lock()


class BatchEvaluationStartResponse(BaseModel):
    task_id: str
    total_files: int
    status: str


def _update_db_task_status(task_id: str, status: str, total: int, done: int, results: list = None, error: str = None):
    conn = get_db_connection()
    if conn is not None:
        try:
            with conn.cursor() as cur:
                res_json = json.dumps(results) if results is not None else '[]'
                err_json = json.dumps([error]) if error else '[]'
                cur.execute(
                    """
                    INSERT INTO batch_tasks (task_id, status, total_sheets, processed_sheets, results, errors, updated_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW())
                    ON CONFLICT (task_id) DO UPDATE SET
                      status = EXCLUDED.status,
                      total_sheets = EXCLUDED.total_sheets,
                      processed_sheets = EXCLUDED.processed_sheets,
                      results = EXCLUDED.results,
                      errors = EXCLUDED.errors,
                      updated_at = NOW()
                    """,
                    (task_id, status, total, done, res_json, err_json)
                )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Error _update_db_task_status]: {e}")

def _get_db_task_status(task_id: str) -> Optional[dict]:
    conn = get_db_connection()
    if conn is not None:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT task_id, status, total_sheets, processed_sheets, results, errors FROM batch_tasks WHERE task_id = %s", (task_id,))
                row = cur.fetchone()
            conn.close()
            if row:
                total = row['total_sheets'] or 1
                done = row['processed_sheets']
                progress_pct = int((done / total) * 100) if total > 0 else 0
                return {
                    "task_id": row['task_id'],
                    "status": row['status'],
                    "progress": f"{done}/{total}",
                    "progress_pct": progress_pct,
                    "done": done,
                    "total": total,
                    "results": row['results'] if isinstance(row['results'], list) else json.loads(row['results'] or '[]'),
                    "errors": row['errors'] if isinstance(row['errors'], list) else json.loads(row['errors'] or '[]')
                }
        except Exception as e:
            print(f"[DB Error _get_db_task_status]: {e}")
    return None

def _process_single_sheet(
    idx: int,
    filename: str,
    content: bytes,
    answer_key: Optional[Dict[int, str]],
    layout: SheetLayout,
    assigned_usns: Optional[List[str]],
    roster_list: List[str],
    version: str,
) -> dict:
    """
    Process a single OMR sheet image end-to-end and return its evaluation result dict.
    Contains the exact pipeline logic used for each sheet in a batch.
    """
    t_start = time.perf_counter()

    orig_arr = np.frombuffer(content, np.uint8)
    orig_img = cv2.imdecode(orig_arr, cv2.IMREAD_COLOR)
    h_orig, w_orig = (orig_img.shape[0], orig_img.shape[1]) if orig_img is not None else (0, 0)
    image_res = f"{w_orig}x{h_orig}" if w_orig and h_orig else None

    clean_img, is_warped, multi_sheet = preprocess_image_detect(content)
    warp_status = "WARPED_SUCCESS" if is_warped else "UNWARPED_FALLBACK"
    detections = run_yolo_inference(clean_img)

    # Check for upside-down orientation using USN bounding box
    usn_dets = [d for d in detections if d.class_name == "usn"]
    if usn_dets:
        usn_det = usn_dets[0]
        h_img = clean_img.shape[0]
        usn_center_y = (usn_det.y1 + usn_det.y2) / 2
        if usn_center_y > h_img / 2:
            clean_img = cv2.rotate(clean_img, cv2.ROTATE_180)
            detections = run_yolo_inference(clean_img)
            usn_dets = [d for d in detections if d.class_name == "usn"]

    bubble_dets = [d for d in detections if d.class_name != "usn"]

    # ── Input Quality Guard (TC#2 / TC#3 / TC#19) ──────────────────
    rejection = _check_bubble_coverage(
        bubble_dets, layout, ambiguous_capture=multi_sheet
    )
    if rejection:
        t_end = time.perf_counter()
        return {
            "filename": filename,
            "usn": None,
            **rejection,
            "score_report": None,
            "annotated_image_b64": None,
            "preprocessed_image_b64": None,
            "original_image_b64": None,
            "bubbles": [],
            "processing_time_ms": int((t_end - t_start) * 1000),
            "_idx": idx,
        }

    usn_value = None
    if assigned_usns and idx < len(assigned_usns):
        usn_value = assigned_usns[idx]
    elif usn_dets:
        d = usn_dets[0]
        usn_value = extract_usn_from_roi(clean_img, d.x1, d.y1, d.x2, d.y2)
        if usn_value and roster_list:
            usn_value = match_usn_against_roster(usn_value, roster_list)

    classifications = classify_all(clean_img, bubble_dets)

    usn_y2 = usn_dets[0].y2 if usn_dets else None
    score_report_dict = None
    if answer_key:
        report = score_sheet(classifications, answer_key, layout, usn_y2=usn_y2)
        score_report_dict = {
            "total_questions": report.total_questions,
            "answered": report.answered,
            "correct": report.correct,
            "incorrect": report.incorrect,
            "unanswered": report.unanswered,
            "multiple_marked": report.multiple_marked,
            "ambiguous": report.ambiguous,
            "score_percent": report.score_percent,
            "per_question": [
                {
                    "question_number": q.question_number,
                    "marked_options": q.marked_options,
                    "correct_option": q.correct_option,
                    "status": q.status.value,
                    "has_ambiguous": q.has_ambiguous,
                } for q in report.per_question
            ]
        }

    filled_cnt = sum(1 for c in classifications if c.state == BubbleState.FILLED)
    empty_cnt = sum(1 for c in classifications if c.state == BubbleState.EMPTY)
    ambig_cnt = sum(1 for c in classifications if c.state == BubbleState.AMBIGUOUS)

    # Generate annotated preview image for page-by-page verification
    valid_classifications = []
    grid = map_bubbles_to_grid(classifications, layout, usn_y2)
    for q_num, opts in grid.items():
        for opt_letter, cr in opts.items():
            valid_classifications.append(cr)
    usn_det = usn_dets[0] if usn_dets else None
    annotated = _annotate_image(clean_img, usn_det, valid_classifications)
    _, annotated_buf = cv2.imencode(".jpg", annotated)
    annotated_img_b64 = base64.b64encode(annotated_buf).decode("utf-8")

    # Convert classifications to bubbles list for interactive correction
    bubbles_res = [
        {
            "bbox": list(c.detection.bbox),
            "confidence": round(c.detection.confidence, 4),
            "class_id": c.detection.class_id,
            "class_name": c.detection.class_name,
            "state": c.state.value,
            "fill_ratio": round(c.fill_ratio, 4),
            "needs_review": c.needs_review,
        }
        for c in classifications
    ]

    t_end = time.perf_counter()
    return {
        "filename": filename,
        "usn": usn_value,
        "version": version,
        "filled_count": filled_cnt,
        "empty_count": empty_cnt,
        "ambiguous_count": ambig_cnt,
        "needs_manual_review": ambig_cnt > 0,
        "total_detections": len(classifications),
        "score_report": score_report_dict,
        "annotated_image_b64": annotated_img_b64,
        "preprocessed_image_b64": annotated_img_b64,
        "original_image_b64": annotated_img_b64,
        "bubbles": bubbles_res,
        "processing_time_ms": int((t_end - t_start) * 1000),
        "image_resolution": image_res,
        "warp_status": warp_status,
        "is_warped": is_warped,
        "_idx": idx,
    }


def run_batch_evaluation_sync(
    task_id: str,
    files_data: List[tuple],
    session_id: str,
    questions_per_column: int,
    num_columns: int,
    options: str,
    version: str = "DEFAULT",
    roster_list: List[str] = None,
    assigned_usns: List[str] = None
):
    try:
        answer_key = get_answer_key_for_session(session_id, version)
        layout = SheetLayout(
            questions_per_column=questions_per_column,
            num_columns=num_columns,
            options=options
        )
        
        # 1. Expand all input files (including PDF pages) into a list of single sheets
        all_sheets = []
        for filename, content in files_data:
            if filename.lower().endswith(".pdf"):
                images = extract_pages_from_pdf(content, dpi=150, max_pages=100)
                for p_idx, img in enumerate(images):
                    _, buffer = cv2.imencode(".jpg", img)
                    all_sheets.append((f"{filename}_page_{p_idx+1}.jpg", buffer.tobytes()))
            else:
                all_sheets.append((filename, content))
                
        if not all_sheets:
            with _tasks_lock:
                _tasks[task_id]["status"] = "completed"
                _tasks[task_id]["progress"] = "0/0"
                _tasks[task_id]["progress_pct"] = 100
            _update_db_task_status(task_id, "completed", 0, 0, [])
            return

        with _tasks_lock:
            _tasks[task_id]["total"] = len(all_sheets)
            _tasks[task_id]["progress"] = f"0/{len(all_sheets)}"
        _update_db_task_status(task_id, "processing", len(all_sheets), 0, [])

        # 2. Process all sheets concurrently using ThreadPoolExecutor
        futures = {}
        with ThreadPoolExecutor(max_workers=_BATCH_MAX_WORKERS) as pool:
            for idx, (filename, content) in enumerate(all_sheets):
                future = pool.submit(
                    _process_single_sheet,
                    idx,
                    filename,
                    content,
                    answer_key,
                    layout,
                    assigned_usns,
                    roster_list or [],
                    version,
                )
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    result.pop("_idx", None)
                except Exception as exc:
                    result = {
                        "filename": all_sheets[idx][0],
                        "error": f"Processing error: {str(exc)}",
                        "processing_time_ms": 0,
                    }

                with _tasks_lock:
                    _tasks[task_id]["results"].append(result)
                    _tasks[task_id]["done"] += 1
                    done_cnt = _tasks[task_id]["done"]
                    progress_pct = int((done_cnt / len(all_sheets)) * 100)
                    _tasks[task_id]["progress"] = f"{done_cnt}/{len(all_sheets)}"
                    _tasks[task_id]["progress_pct"] = progress_pct
                    _tasks[task_id]["status"] = "processing" if done_cnt < len(all_sheets) else "completed"
                    curr_results = list(_tasks[task_id]["results"])
                    curr_status = _tasks[task_id]["status"]

                if curr_status == "completed" or done_cnt % 5 == 0 or done_cnt == len(all_sheets):
                    _update_db_task_status(task_id, curr_status, len(all_sheets), done_cnt, curr_results)

    except Exception as e:
        with _tasks_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(e)
        _update_db_task_status(task_id, "failed", len(files_data), 0, [], error=str(e))


async def run_batch_evaluation_async(
    task_id: str,
    files_data: List[tuple],
    session_id: str,
    questions_per_column: int,
    num_columns: int,
    options: str,
    version: str = "DEFAULT",
    roster_list: List[str] = None,
    assigned_usns: List[str] = None
):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        run_batch_evaluation_sync,
        task_id,
        files_data,
        session_id,
        questions_per_column,
        num_columns,
        options,
        version,
        roster_list,
        assigned_usns
    )


@app.post("/api/v1/batch-evaluate", response_model=BatchEvaluationStartResponse, status_code=202)
async def batch_evaluate_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session_id: str = Form("default"),
    version: str = Form("DEFAULT"),
    questions_per_column: int = Form(15),
    num_columns: int = Form(2),
    options: str = Form("ABCD"),
    roster: Optional[str] = Form(None),
    assigned_usns: Optional[str] = Form(None)
):
    files_data = []
    for file in files:
        content = await file.read()
        files_data.append((file.filename, content))
        
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {
            "status": "queued",
            "progress": f"0/{len(files_data)}",
            "progress_pct": 0,
            "done": 0,
            "total": len(files_data),
            "results": []
        }
    _update_db_task_status(task_id, "queued", len(files_data), 0, [])

    import json
    roster_list = json.loads(roster) if roster else []
    assigned_usns_list = json.loads(assigned_usns) if assigned_usns else None
        
    background_tasks.add_task(
        run_batch_evaluation_async,
        task_id,
        files_data,
        session_id,
        questions_per_column,
        num_columns,
        options,
        version,
        roster_list,
        assigned_usns_list
    )
    
    return BatchEvaluationStartResponse(
        task_id=task_id,
        total_files=len(files_data),
        status="queued"
    )


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        db_task = _get_db_task_status(task_id)
        if db_task:
            return db_task
        raise HTTPException(status_code=404, detail="Task not found")
    return task


# ── Answer Key management ───────────────────────────────────────────────────

class AnswerKeyRequest(BaseModel):
    """Upload correct answers for a session and version (A/B/C/D)."""
    session_id: str
    version: Optional[str] = "DEFAULT"
    answers: Dict[int, str]  # { 1: "A", 2: "C", 3: "B", ... }


class AnswerKeyResponse(BaseModel):
    session_id: str
    version: str = "DEFAULT"
    total_questions: int
    saved: bool


_fallback_answer_keys: Dict[str, Dict[str, Dict[int, str]]] = {}

@app.post("/answer-key", response_model=AnswerKeyResponse)
async def upload_answer_key(req: AnswerKeyRequest):
    """
    Register the correct answer key for a given exam session and version.
    Persists to PostgreSQL if available, otherwise falls back to memory.
    """
    sess_id = req.session_id
    ver = (req.version or "DEFAULT").upper()
    clean_answers = {int(k): str(v).upper() for k, v in req.answers.items()}

    conn = get_db_connection()
    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO exam_versions (session_id, version, answers, created_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (session_id, version)
                    DO UPDATE SET answers = EXCLUDED.answers, created_at = NOW()
                    """,
                    (sess_id, ver, json.dumps(clean_answers))
                )
            conn.commit()
            conn.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to persist answer key to DB: {e}")
    else:
        if sess_id not in _fallback_answer_keys:
            _fallback_answer_keys[sess_id] = {}
        _fallback_answer_keys[sess_id][ver] = clean_answers

    return AnswerKeyResponse(
        session_id=sess_id,
        version=ver,
        total_questions=len(clean_answers),
        saved=True,
    )


@app.get("/answer-key/{session_id}")
async def get_answer_key(session_id: str, version: Optional[str] = None):
    """Retrieve the stored answer key for a session and version."""
    conn = get_db_connection()
    if conn is not None:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT version, answers FROM exam_versions WHERE session_id = %s",
                    (session_id,)
                )
                rows = cur.fetchall()
            conn.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

        if not rows:
            raise HTTPException(status_code=404, detail=f"No answer key for session '{session_id}'.")

        if version:
            ver = version.upper()
            key = get_answer_key_for_session(session_id, ver)
            if key is None:
                raise HTTPException(status_code=404, detail=f"No answer key for session '{session_id}' version '{ver}'.")
            return {"session_id": session_id, "version": ver, "answers": key, "total_questions": len(key)}

        default_key = get_answer_key_for_session(session_id, "DEFAULT") or {}
        versions_dict = {}
        for r in rows:
            v_name = r['version'].upper()
            ans = r['answers']
            if isinstance(ans, str):
                ans = json.loads(ans)
            parsed_ans = {int(k): str(v).upper() for k, v in ans.items()}
            versions_dict[v_name] = {"answers": parsed_ans, "total_questions": len(parsed_ans)}

        return {
            "session_id": session_id,
            "version": "DEFAULT",
            "answers": default_key,
            "total_questions": len(default_key),
            "versions": versions_dict
        }
    else:
        if session_id not in _fallback_answer_keys:
            raise HTTPException(status_code=404, detail=f"No answer key for session '{session_id}'.")
        session_val = _fallback_answer_keys[session_id]
        if version:
            ver = version.upper()
            key = session_val.get(ver)
            if key is None:
                raise HTTPException(status_code=404, detail=f"No answer key for session '{session_id}' version '{ver}'.")
            return {"session_id": session_id, "version": ver, "answers": key, "total_questions": len(key)}

        default_key = session_val.get("DEFAULT", {})
        versions_dict = {v: {"answers": keys, "total_questions": len(keys)} for v, keys in session_val.items()}
        return {
            "session_id": session_id,
            "version": "DEFAULT",
            "answers": default_key,
            "total_questions": len(default_key),
            "versions": versions_dict
        }



# ── Scoring endpoint ────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """Score a sheet against its session's answer key."""
    session_id: str
    version: Optional[str] = "DEFAULT"
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
    version: Optional[str] = "DEFAULT",
    questions_per_column: int = 15,
    num_columns: int = 2,
    options: str = "ABCD",
):
    """
    Full pipeline + scoring in one call.

    Runs the evaluation pipeline and then scores the results against
    the answer key stored for the given session_id and version.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    answer_key = get_answer_key_for_session(session_id, version)

    try:
        contents = await file.read()
        clean_img = preprocess_image(contents)
        detections = run_yolo_inference(clean_img)
        usn_dets = [d for d in detections if d.class_name == "usn"]
        bubble_detections = [d for d in detections if d.class_name != "usn"]
        classifications = classify_all(clean_img, bubble_detections)

        layout = SheetLayout(
            questions_per_column=questions_per_column,
            num_columns=num_columns,
            options=options,
        )

        usn_y2 = usn_dets[0].y2 if usn_dets else None
        report = score_sheet(classifications, answer_key, layout, usn_y2=usn_y2)

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

class RescoreRequest(BaseModel):
    """Request payload for manual re-scoring (overrides)."""
    session_id: str = "default"
    version: Optional[str] = "DEFAULT"
    questions_per_column: int = 15
    num_columns: int = 2
    options: str = "ABCD"
    bubbles: List[BubbleResult]


@app.post("/re-score", response_model=ScoreResponse)
async def rescore_sheet(req: RescoreRequest):
    """
    Fast-path scoring endpoint that bypasses inference.
    Accepts the global bubble array from the frontend (with manual overrides applied)
    and instantly recalculates the score.
    """
    answer_key = get_answer_key_for_session(req.session_id, req.version)
    if not answer_key:
        raise HTTPException(status_code=400, detail="Answer key not found for session.")

    from core.localization import BubbleDetection
    from core.classification import ClassificationResult, BubbleState

    classifications = []
    for b in req.bubbles:
        det = BubbleDetection(
            x1=b.bbox[0],
            y1=b.bbox[1],
            x2=b.bbox[2],
            y2=b.bbox[3],
            confidence=b.confidence,
            class_id=b.class_id,
            class_name=b.class_name
        )
        cr = ClassificationResult(
            detection=det,
            state=BubbleState(b.state),
            fill_ratio=b.fill_ratio
        )
        classifications.append(cr)

    layout = SheetLayout(
        questions_per_column=req.questions_per_column,
        num_columns=req.num_columns,
        options=req.options,
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
    usn_detection = None,
    valid_classifications = None,
) -> np.ndarray:
    """Draw bounding boxes and labels ONLY for valid USN and OMR bubble classifications."""
    annotated = image.copy()
    h_img, w_img = annotated.shape[:2]

    # Draw USN Box if detected
    if usn_detection:
        x1, y1, x2, y2 = int(usn_detection.x1), int(usn_detection.y1), int(usn_detection.x2), int(usn_detection.y2)
        x1, y1 = max(0, min(w_img - 1, x1)), max(0, min(h_img - 1, y1))
        x2, y2 = max(0, min(w_img - 1, x2)), max(0, min(h_img - 1, y2))
        color = _VIZ_COLORS["usn"]
        label = f"USN {usn_detection.confidence:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ly1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, ly1), (min(w_img - 1, x1 + tw + 4), y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw valid bubbles
    if valid_classifications:
        for cr in valid_classifications:
            det = cr.detection
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            x1, y1 = max(0, min(w_img - 1, x1)), max(0, min(h_img - 1, y1))
            x2, y2 = max(0, min(w_img - 1, x2)), max(0, min(h_img - 1, y2))
            color = _VIZ_COLORS.get(cr.state.value, (255, 255, 255))
            label = f"{cr.state.value} {cr.fill_ratio:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ly1 = max(0, y1 - th - 6)
            cv2.rectangle(annotated, (x1, ly1), (min(w_img - 1, x1 + tw + 4), y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def _img_to_base64(image: np.ndarray) -> str:
    """Encode a BGR image as base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


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
    usn: Optional[str] = None
    score_report: Optional[dict] = None


@app.post("/debug/evaluate")
async def debug_evaluate(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form("default"),
    version: Optional[str] = Form("DEFAULT"),
    questions_per_column: int = Form(15),
    num_columns: int = Form(2),
    options: str = Form("ABCD"),
    roster: Optional[str] = Form(None),
    assigned_usn: Optional[str] = Form(None)
):
    """
    Debug endpoint: runs the full pipeline and returns annotated images
    alongside the JSON results. Used for visual verification.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        t_start = time.perf_counter()
        import json
        roster_list = json.loads(roster) if roster else []
        contents = await file.read()

        # Preprocess
        orig_arr = np.frombuffer(contents, np.uint8)
        orig_img = cv2.imdecode(orig_arr, cv2.IMREAD_COLOR)
        h_orig, w_orig = (orig_img.shape[0], orig_img.shape[1]) if orig_img is not None else (0, 0)
        image_res = f"{w_orig}x{h_orig}" if w_orig and h_orig else None

        clean_img, is_warped, multi_sheet = preprocess_image_detect(contents)
        warp_status = "WARPED_SUCCESS" if is_warped else "UNWARPED_FALLBACK"

        # Localize
        detections = run_yolo_inference(clean_img)

        # Check for upside-down orientation using USN bounding box
        usn_dets = [d for d in detections if d.class_name == "usn"]
        if usn_dets:
            usn_det = usn_dets[0]
            h_img = clean_img.shape[0]
            usn_center_y = (usn_det.y1 + usn_det.y2) / 2
            if usn_center_y > h_img / 2:
                clean_img = cv2.rotate(clean_img, cv2.ROTATE_180)
                # Rotate original image too for correct display in debug
                if orig_img is not None:
                    orig_img = cv2.rotate(orig_img, cv2.ROTATE_180)
                detections = run_yolo_inference(clean_img)
                usn_dets = [d for d in detections if d.class_name == "usn"]

        bubble_dets = [d for d in detections if d.class_name != "usn"]

        # ── Input Quality Guard (TC#2 / TC#3 / TC#19) ──────────────────
        layout_for_guard = SheetLayout(
            questions_per_column=questions_per_column,
            num_columns=num_columns,
            options=options,
        )
        rejection = _check_bubble_coverage(
            bubble_dets, layout_for_guard, ambiguous_capture=multi_sheet
        )
        if rejection:
            return JSONResponse(status_code=422, content=rejection)

        # Extract USN
        req_prefix = str(uuid.uuid4())[:8]
        usn_value = assigned_usn
        if not usn_value and usn_dets:
            det = usn_dets[0]
            usn_value = extract_usn_from_roi(
                clean_img, det.x1, det.y1, det.x2, det.y2,
                debug_prefix=req_prefix if OMR_DEBUG_DUMP else ""
            )
            if usn_value and roster_list:
                usn_value = match_usn_against_roster(usn_value, roster_list)

        # Classify
        classifications = classify_all(clean_img, bubble_dets)

        layout = SheetLayout(
            questions_per_column=questions_per_column,
            num_columns=num_columns,
            options=options,
        )

        # ── Debug Save ──────────────────────────────────────────────────────
        if OMR_DEBUG_DUMP:
            debug_dir = Path(__file__).parent / "debug_output"
            debug_dir.mkdir(exist_ok=True)
            
            # Save preprocessed sheet
            cv2.imwrite(str(debug_dir / f"{req_prefix}_preprocessed_sheet.png"), clean_img)
            
            # Save a few sample bubble details to debug folder
            from core.classification import _extract_inner_region
            for idx, cr in enumerate(classifications[:15]):
                det = cr.detection
                x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                roi = clean_img[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_resized = cv2.resize(roi, (64, 64))
                    gray, mask = _extract_inner_region(roi_resized)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    masked_binary = cv2.bitwise_and(binary, mask)
                    
                    cv2.imwrite(str(debug_dir / f"{req_prefix}_bubble_{idx}_roi.png"), roi_resized)
                    cv2.imwrite(str(debug_dir / f"{req_prefix}_bubble_{idx}_binary.png"), binary)
                    cv2.imwrite(str(debug_dir / f"{req_prefix}_bubble_{idx}_mask.png"), mask)
                    cv2.imwrite(str(debug_dir / f"{req_prefix}_bubble_{idx}_masked.png"), masked_binary)
                    
                    # Write text metadata
                    with open(debug_dir / f"{req_prefix}_bubble_{idx}_meta.txt", "w") as f:
                        f.write(f"state: {cr.state.value}\n")
                        f.write(f"fill_ratio: {cr.fill_ratio:.4f}\n")
                        f.write(f"bbox: {x1}, {y1}, {x2}, {y2}\n")

        # Annotate ONLY valid detections
        usn_y2 = usn_dets[0].y2 if usn_dets else None
        grid = map_bubbles_to_grid(classifications, layout, usn_y2)
        valid_classifications = []
        for q_num, opts in grid.items():
            for opt_letter, cr in opts.items():
                valid_classifications.append(cr)
                
        usn_det = usn_dets[0] if usn_dets else None
        annotated = _annotate_image(clean_img, usn_det, valid_classifications)

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

        # Scoring report if answer key is set
        score_report_dict = None
        answer_key = get_answer_key_for_session(session_id, version) if session_id else None
        if answer_key:
            usn_y2 = usn_dets[0].y2 if usn_dets else None
            report = score_sheet(classifications, answer_key, layout, usn_y2=usn_y2)
            score_report_dict = {
                "total_questions": report.total_questions,
                "answered": report.answered,
                "correct": report.correct,
                "incorrect": report.incorrect,
                "unanswered": report.unanswered,
                "multiple_marked": report.multiple_marked,
                "ambiguous": report.ambiguous,
                "score_percent": report.score_percent,
                "per_question": [
                    {
                        "question_number": q.question_number,
                        "marked_options": q.marked_options,
                        "correct_option": q.correct_option,
                        "status": q.status.value,
                        "has_ambiguous": q.has_ambiguous,
                    }
                    for q in report.per_question
                ]
            }

        return {
            "original_image_b64": _img_to_base64(orig_img) if orig_img is not None else "",
            "preprocessed_image_b64": _img_to_base64(clean_img),
            "annotated_image_b64": _img_to_base64(annotated),
            "total_detections": len(detections),
            "bubble_detections": len(bubble_dets),
            "usn_detections": len(usn_dets),
            "usn": usn_value,
            "filled_count": filled,
            "empty_count": empty,
            "ambiguous_count": ambig,
            "needs_manual_review": ambig > 0,
            "processing_time_ms": int((t_end - t_start) * 1000),
            "image_resolution": image_res,
            "warp_status": warp_status,
            "is_warped": is_warped,
            "bubbles": [b.model_dump() for b in bubbles],
            "score_report": score_report_dict
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


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Serve the login HTML page."""
    html_path = _STATIC_DIR / "login.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Login page not found.")
    return html_path.read_text(encoding="utf-8")


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin HTML page."""
    html_path = _STATIC_DIR / "admin.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Admin page not found.")
    return html_path.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root redirects to /dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


# ── Live Webcam WebSocket ────────────────────────────────────────────────────

from fastapi import WebSocket, WebSocketDisconnect
import json
import base64

@app.websocket("/ws/evaluate")
async def websocket_evaluate(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected")
    last_valid_usn = None
    try:
        while True:
            # Receive frame data (base64 string inside json)
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            img_b64 = payload.get("image")
            session_id = payload.get("session_id", "default")
            
            if not img_b64:
                continue
                
            assigned_usn = payload.get("assigned_usn")
            
            # Decode base64 bytes
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]
            img_bytes = base64.b64decode(img_b64)
            
            # Preprocess and check alignment
            try:
                clean_img, aligned, multi_sheet = preprocess_image_detect(img_bytes)
            except Exception as e:
                print(f"[WebSocket Preprocess Error] {e}")
                continue
                
            if not aligned:
                response = {
                    "aligned": False,
                    "usn": "ALIGNING",
                    "filled_count": 0,
                    "empty_count": 0,
                    "ambiguous_count": 0,
                    "needs_manual_review": False,
                    "bubbles": [],
                    "annotated_image_b64": None,
                    "message": "Aligning OMR sheet..."
                }
                await websocket.send_text(json.dumps(response))
                continue
                
            # Localize
            detections = run_yolo_inference(clean_img)

            # Check for upside-down orientation using USN bounding box
            usn_dets = [d for d in detections if d.class_name == "usn"]
            if usn_dets:
                usn_det = usn_dets[0]
                h_img = clean_img.shape[0]
                usn_center_y = (usn_det.y1 + usn_det.y2) / 2
                if usn_center_y > h_img / 2:
                    clean_img = cv2.rotate(clean_img, cv2.ROTATE_180)
                    detections = run_yolo_inference(clean_img)
                    usn_dets = [d for d in detections if d.class_name == "usn"]
                    
            bubble_dets = [d for d in detections if d.class_name != "usn"]
            
            # ── Input Quality Guard (TC#2 / TC#3 / TC#19) ──────────────────
            layout = SheetLayout(questions_per_column=15, num_columns=2, options="ABCD")
            rejection = _check_bubble_coverage(
                bubble_dets, layout, ambiguous_capture=multi_sheet
            )
            if rejection:
                response = {
                    "aligned": False,
                    "usn": "ALIGNING",
                    "filled_count": 0,
                    "empty_count": 0,
                    "ambiguous_count": 0,
                    "needs_manual_review": False,
                    "bubbles": [],
                    "annotated_image_b64": None,
                    "message": rejection.get("detail", "Aligning OMR sheet...")
                }
                await websocket.send_text(json.dumps(response))
                continue
            
            # Extract USN
            usn_value = None
            if assigned_usn:
                usn_value = assigned_usn
                last_valid_usn = assigned_usn
            elif usn_dets:
                det = usn_dets[0]
                if last_valid_usn and last_valid_usn != "UNKNOWN":
                    usn_value = last_valid_usn
                else:
                    usn_value = extract_usn_from_roi(clean_img, det.x1, det.y1, det.x2, det.y2)
                    if usn_value and usn_value != "UNKNOWN":
                        last_valid_usn = usn_value
                
            # Classify
            classifications = classify_all(clean_img, bubble_dets)
            
            # Grade
            session_id = payload.get("session_id", "default")
            version = payload.get("version", "DEFAULT")
            answer_key = get_answer_key_for_session(session_id, version) if session_id else None
            usn_y2 = usn_dets[0].y2 if usn_dets else None
            layout = SheetLayout(questions_per_column=15, num_columns=2, options="ABCD")
            score_report = score_sheet(classifications, answer_key, layout, usn_y2=usn_y2)
            
            # Annotate clean image for visual feedback
            annotated_img = clean_img.copy()
            for cr in classifications:
                color = (0, 255, 0) if cr.state == BubbleState.FILLED else (0, 0, 255) if cr.state == BubbleState.AMBIGUOUS else (128, 128, 128)
                det = cr.detection
                cv2.rectangle(annotated_img, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), color, 2)
                cv2.putText(annotated_img, cr.state.value, (int(det.x1), int(det.y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
            # Draw USN bounding box
            if usn_dets:
                det = usn_dets[0]
                cv2.rectangle(annotated_img, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), (255, 0, 0), 2)
                if usn_value:
                    cv2.putText(annotated_img, f"USN: {usn_value}", (int(det.x1), int(det.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Encode annotated image back to base64 jpeg
            _, buffer = cv2.imencode(".jpg", annotated_img)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")
            
            # Count statuses
            filled = sum(1 for c in classifications if c.state == BubbleState.FILLED)
            empty = sum(1 for c in classifications if c.state == BubbleState.EMPTY)
            ambig = sum(1 for c in classifications if c.state == BubbleState.AMBIGUOUS)
            
            # Prepare response dict
            score_report_dict = None
            if score_report:
                score_report_dict = {
                    "total_questions": score_report.total_questions,
                    "correct": score_report.correct,
                    "incorrect": score_report.incorrect,
                    "unanswered": score_report.unanswered,
                    "multiple_marked": score_report.multiple_marked,
                    "score_percent": score_report.score_percent,
                    "per_question": [
                        {
                            "question_number": q.question_number,
                            "marked_options": q.marked_options,
                            "correct_option": q.correct_option,
                            "status": q.status
                        } for q in score_report.per_question
                    ]
                }

            h_clean, w_clean = clean_img.shape[:2]
            response = {
                "aligned": True,
                "usn": usn_value or "UNKNOWN",
                "filled_count": filled,
                "empty_count": empty,
                "ambiguous_count": ambig,
                "needs_manual_review": ambig > 0,
                "image_resolution": f"{w_clean}x{h_clean}",
                "warp_status": "WARPED_SUCCESS" if aligned else "UNWARPED_FALLBACK",
                "is_warped": bool(aligned),
                "score_report": score_report_dict,
                "bubbles": [
                    {
                        "index": idx,
                        "state": c.state.value,
                        "confidence": c.detection.confidence,
                        "fill_ratio": c.fill_ratio,
                        "needs_review": c.state == BubbleState.AMBIGUOUS
                    } for idx, c in enumerate(classifications)
                ],
                "annotated_image_b64": annotated_b64,
                "message": "Evaluation successful"
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected gracefully")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")