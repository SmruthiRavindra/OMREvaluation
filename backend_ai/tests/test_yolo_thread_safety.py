"""
test_yolo_thread_safety.py
--------------------------
Verifies thread-safety of concurrent YOLO inference with thread-local models:
1. test_concurrent_yolo_results_are_stable — checks result stability under concurrent load
2. test_concurrent_cold_start — checks concurrent first-time model loading across threads
"""
from __future__ import annotations

import sys, os, threading, types
import numpy as np
import cv2
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.localization import run_yolo_inference, _get_model


def _make_blank_image_bytes() -> bytes:
    img = np.ones((1100, 800, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestYOLOThreadSafety:

    def test_concurrent_yolo_results_are_stable(self):
        """
        Runs concurrent inference across multiple threads after pre-warming
        and asserts that all detection outputs match baseline.
        """
        content = _make_blank_image_bytes()
        img = preprocess_image(content)

        # Baseline single-threaded runs
        baseline_counts = [len(run_yolo_inference(img)) for _ in range(5)]
        expected = baseline_counts[0]

        def _worker_task(_):
            img_local = preprocess_image(content)
            return len(run_yolo_inference(img_local))

        all_counts = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_worker_task, i) for i in range(20)]
            for f in as_completed(futures):
                all_counts.append(f.result())

        mismatches = [c for c in all_counts if c != expected]
        assert not mismatches, f"Mismatches found: expected {expected}, got {mismatches}"

    def test_concurrent_cold_start(self):
        """
        Verifies that multiple worker threads calling _get_model() simultaneously
        receive a valid, thread-safe YOLO instance without exceptions or race conditions.
        """
        models_fetched = []
        errors = []

        def _cold_start_worker():
            try:
                model = _get_model()
                assert model is not None
                models_fetched.append(model)
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_cold_start_worker) for _ in range(8)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors during cold start: {errors}"
        assert len(models_fetched) == 8
        instance_ids = set(id(m) for m in models_fetched)
        assert len(instance_ids) == 1