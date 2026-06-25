"""
test_pipeline.py
================
Unit + integration tests for the OMR backend AI pipeline.

Run:
  cd backend_ai
  python -m pytest tests/ -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test: preprocess module
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Tests for core/preprocess.py functions."""

    def test_decode_valid_jpeg(self):
        """A valid JPEG should decode to a 3-channel BGR numpy array."""
        import cv2
        from core.preprocess import preprocess_image

        # Create a synthetic 200×300 RGB image and encode as JPEG
        dummy_img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        success, jpeg_bytes = cv2.imencode(".jpg", dummy_img)
        assert success, "Failed to encode test image"

        result = preprocess_image(jpeg_bytes.tobytes())
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3       # H × W × C
        assert result.shape[2] == 3   # BGR channels

    def test_decode_invalid_bytes_raises(self):
        """Random non-image bytes should raise ValueError."""
        from core.preprocess import preprocess_image

        with pytest.raises(ValueError, match="Failed to decode"):
            preprocess_image(b"this is not an image")

    def test_bilateral_filter_preserves_shape(self):
        """Bilateral filter should not change image dimensions."""
        from core.preprocess import _bilateral_filter

        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        filtered = _bilateral_filter(img)
        assert filtered.shape == img.shape

    def test_perspective_warp_returns_image(self):
        """Even without detectable corners, warp should gracefully return."""
        from core.preprocess import _perspective_warp

        # Uniform gray image — no edges to detect
        img = np.full((400, 300, 3), 128, dtype=np.uint8)
        warped = _perspective_warp(img)
        assert isinstance(warped, np.ndarray)
        assert warped.ndim == 3


# ---------------------------------------------------------------------------
# Test: classification module
# ---------------------------------------------------------------------------


class TestClassification:
    """Tests for core/classification.py threshold-based classifier."""

    def test_solid_black_bubble_is_filled(self):
        """A fully dark bubble ROI should be classified as FILLED."""
        from core.classification import _classify_with_threshold, BubbleState
        from core.localization import BubbleDetection

        # Pure black 64×64 ROI
        roi = np.zeros((64, 64, 3), dtype=np.uint8)
        det = BubbleDetection(0, 0, 64, 64, 0.95, 0, "filled")
        result = _classify_with_threshold(roi, det)
        assert result.state == BubbleState.FILLED

    def test_solid_white_bubble_is_empty(self):
        """A fully white bubble ROI should be classified as EMPTY."""
        from core.classification import _classify_with_threshold, BubbleState
        from core.localization import BubbleDetection

        # Pure white 64×64 ROI
        roi = np.full((64, 64, 3), 255, dtype=np.uint8)
        det = BubbleDetection(0, 0, 64, 64, 0.90, 1, "unfilled")
        result = _classify_with_threshold(roi, det)
        assert result.state == BubbleState.EMPTY

    def test_classify_all_returns_list(self):
        """classify_all should return a list of ClassificationResult."""
        from core.classification import classify_all
        from core.localization import BubbleDetection

        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        detections = [
            BubbleDetection(10, 10, 30, 30, 0.9, 0, "filled"),
            BubbleDetection(50, 10, 70, 30, 0.8, 1, "unfilled"),
        ]
        results = classify_all(img, detections)
        assert len(results) == 2
        for r in results:
            assert hasattr(r, "state")
            assert hasattr(r, "fill_ratio")


# ---------------------------------------------------------------------------
# Test: localization module (NMS logic — does not need model weights)
# ---------------------------------------------------------------------------


class TestNMS:
    """Tests for the custom NMS in core/localization.py."""

    def test_nms_removes_overlapping_boxes(self):
        """Two nearly identical boxes on the same class should be deduplicated."""
        from core.localization import _custom_nms, BubbleDetection

        dets = [
            BubbleDetection(10, 10, 50, 50, 0.95, 0, "filled"),
            BubbleDetection(12, 12, 52, 52, 0.80, 0, "filled"),  # overlaps with above
        ]
        kept = _custom_nms(dets, iou_threshold=0.45)
        assert len(kept) == 1
        assert kept[0].confidence == 0.95  # higher confidence kept

    def test_nms_keeps_different_classes(self):
        """Overlapping boxes of different classes should both be kept."""
        from core.localization import _custom_nms, BubbleDetection

        dets = [
            BubbleDetection(10, 10, 50, 50, 0.95, 0, "filled"),
            BubbleDetection(12, 12, 52, 52, 0.90, 1, "unfilled"),
        ]
        kept = _custom_nms(dets, iou_threshold=0.45)
        assert len(kept) == 2

    def test_nms_keeps_non_overlapping(self):
        """Non-overlapping boxes of the same class should all be kept."""
        from core.localization import _custom_nms, BubbleDetection

        dets = [
            BubbleDetection(10, 10, 30, 30, 0.95, 0, "filled"),
            BubbleDetection(100, 100, 120, 120, 0.90, 0, "filled"),
        ]
        kept = _custom_nms(dets, iou_threshold=0.45)
        assert len(kept) == 2

    def test_iou_identical_boxes(self):
        """IoU of two identical boxes should be 1.0."""
        from core.localization import _iou, BubbleDetection

        a = BubbleDetection(10, 10, 50, 50, 0.9, 0, "filled")
        b = BubbleDetection(10, 10, 50, 50, 0.8, 0, "filled")
        assert abs(_iou(a, b) - 1.0) < 1e-6

    def test_iou_non_overlapping_boxes(self):
        """IoU of non-overlapping boxes should be 0.0."""
        from core.localization import _iou, BubbleDetection

        a = BubbleDetection(0, 0, 10, 10, 0.9, 0, "filled")
        b = BubbleDetection(100, 100, 110, 110, 0.8, 0, "filled")
        assert _iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# Test: scoring module
# ---------------------------------------------------------------------------


class TestScoring:
    """Tests for core/scoring.py spatial mapping and scoring engine."""

    def _make_cr(self, x1, y1, x2, y2, state_str):
        """Helper to create a ClassificationResult for testing."""
        from core.localization import BubbleDetection
        from core.classification import ClassificationResult, BubbleState

        det = BubbleDetection(x1, y1, x2, y2, 0.9, 0, "filled")
        state = BubbleState(state_str)
        fill = 0.8 if state == BubbleState.FILLED else 0.05
        return ClassificationResult(det, state, fill)

    def test_map_single_question_4_options(self):
        """4 bubbles in a horizontal row should map to Q1 with options A-D."""
        from core.scoring import map_bubbles_to_grid, SheetLayout

        layout = SheetLayout(questions_per_column=1, num_columns=1, options="ABCD")
        crs = [
            self._make_cr(10, 50, 30, 70, "empty"),    # A
            self._make_cr(40, 50, 60, 70, "filled"),    # B
            self._make_cr(70, 50, 90, 70, "empty"),     # C
            self._make_cr(100, 50, 120, 70, "empty"),   # D
        ]
        grid = map_bubbles_to_grid(crs, layout)
        assert 1 in grid
        assert len(grid[1]) == 4
        # B should be filled
        assert grid[1]["B"].state.value == "filled"

    def test_score_sheet_perfect(self):
        """All correct answers should yield 100% score."""
        from core.scoring import score_sheet, SheetLayout

        layout = SheetLayout(questions_per_column=2, num_columns=1, options="ABCD")
        answer_key = {1: "B", 2: "C"}

        # Q1: row at y=50, B is filled (x=40..60)
        # Q2: row at y=150, C is filled (x=70..90)
        crs = [
            # Q1 options
            self._make_cr(10, 50, 30, 70, "empty"),
            self._make_cr(40, 50, 60, 70, "filled"),
            self._make_cr(70, 50, 90, 70, "empty"),
            self._make_cr(100, 50, 120, 70, "empty"),
            # Q2 options
            self._make_cr(10, 150, 30, 170, "empty"),
            self._make_cr(40, 150, 60, 170, "empty"),
            self._make_cr(70, 150, 90, 170, "filled"),
            self._make_cr(100, 150, 120, 170, "empty"),
        ]

        report = score_sheet(crs, answer_key, layout)
        assert report.correct == 2
        assert report.score_percent == 100.0

    def test_score_sheet_unanswered(self):
        """A question with no filled bubbles should be unanswered."""
        from core.scoring import score_sheet, SheetLayout, AnswerStatus

        layout = SheetLayout(questions_per_column=1, num_columns=1, options="ABCD")
        answer_key = {1: "A"}

        crs = [
            self._make_cr(10, 50, 30, 70, "empty"),
            self._make_cr(40, 50, 60, 70, "empty"),
            self._make_cr(70, 50, 90, 70, "empty"),
            self._make_cr(100, 50, 120, 70, "empty"),
        ]

        report = score_sheet(crs, answer_key, layout)
        assert report.unanswered == 1
        assert report.per_question[0].status == AnswerStatus.UNANSWERED

    def test_map_tilted_bubbles(self):
        """Bubbles with slight vertical tilt/jitter within 15px should cluster into the correct rows."""
        from core.scoring import map_bubbles_to_grid, SheetLayout

        layout = SheetLayout(questions_per_column=2, num_columns=1, options="ABCD")

        # Row 1 (y average ~53.0) and Row 2 (y average ~103.0)
        crs = [
            # Row 1 bubbles: y has jitter [50, 55, 52, 54]
            self._make_cr(10, 50, 30, 70, "empty"),
            self._make_cr(40, 55, 60, 75, "filled"),
            self._make_cr(70, 52, 90, 72, "empty"),
            self._make_cr(100, 54, 120, 74, "empty"),
            # Row 2 bubbles: y has jitter [100, 105, 102, 104]
            self._make_cr(10, 100, 30, 120, "empty"),
            self._make_cr(40, 105, 60, 125, "empty"),
            self._make_cr(70, 102, 90, 122, "filled"),
            self._make_cr(100, 104, 120, 124, "empty"),
        ]

        grid = map_bubbles_to_grid(crs, layout)
        assert 1 in grid
        assert 2 in grid
        assert len(grid[1]) == 4
        assert len(grid[2]) == 4
        assert grid[1]["B"].state.value == "filled"
        assert grid[2]["C"].state.value == "filled"


# ---------------------------------------------------------------------------
# Test: FastAPI endpoints
# ---------------------------------------------------------------------------


class TestAPI:
    """Tests for FastAPI HTTP endpoints (no model weights required)."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_answer_key_round_trip(self, client):
        """Upload an answer key and retrieve it."""
        payload = {
            "session_id": "test_session_001",
            "answers": {"1": "A", "2": "C", "3": "B"},
        }
        resp = client.post("/answer-key", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True
        assert data["total_questions"] == 3

        # Retrieve
        resp2 = client.get("/answer-key/test_session_001")
        assert resp2.status_code == 200
        assert resp2.json()["total_questions"] == 3

    def test_answer_key_not_found(self, client):
        resp = client.get("/answer-key/nonexistent")
        assert resp.status_code == 404

    def test_evaluate_rejects_non_image(self, client):
        """Uploading a non-image file should return 400."""
        resp = client.post(
            "/evaluate",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400
