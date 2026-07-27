import types, sys, os, numpy as np, cv2, pytest
sys.path.insert(0, os.path.join(os.path.dirname('/app/tests/'), '..'))
from main import _check_bubble_coverage, _MIN_BUBBLE_FRACTION, _MIN_ROW_FRACTION
from core.scoring import SheetLayout
from core.preprocess import _count_large_quads

LAYOUT = SheetLayout(questions_per_column=15, num_columns=2, options='ABCD')
EXPECTED_TOTAL = LAYOUT.total_bubbles
MIN_BUB = int(_MIN_BUBBLE_FRACTION * EXPECTED_TOTAL)

def _det(y1=0, y2=10, x1=0, x2=10):
    return types.SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, class_name='q1')

class TestTC2BlankDocument:
    def test_zero_detections_rejected(self):
        r = _check_bubble_coverage([], LAYOUT)
        assert r and r['reason'] == 'insufficient_detections'
    def test_below_threshold_rejected(self):
        dets = [_det(y1=i*5, y2=i*5+4) for i in range(5)]
        r = _check_bubble_coverage(dets, LAYOUT)
        assert r and r['reason'] == 'insufficient_detections'
    def test_at_threshold_not_tc2(self):
        dets = [_det(y1=i*5, y2=i*5+4) for i in range(MIN_BUB)]
        r = _check_bubble_coverage(dets, LAYOUT)
        if r: assert r['reason'] != 'insufficient_detections'

class TestTC3HalfPage:
    def _row_dets(self, rows, spacing=40):
        return [_det(y1=r*spacing, y2=r*spacing+10, x1=o*15, x2=o*15+12) for r in range(rows) for o in range(4)]
    def test_3_rows_rejected(self):
        dets = self._row_dets(3)
        while len(dets) < MIN_BUB: dets += self._row_dets(3)
        r = _check_bubble_coverage(dets, LAYOUT)
        assert r and r['reason'] == 'incomplete_sheet'
    def test_15_rows_not_tc3(self):
        dets = self._row_dets(15)
        r = _check_bubble_coverage(dets, LAYOUT)
        if r: assert r['reason'] != 'incomplete_sheet'

class TestTC19MultipleSheets:
    def test_ambiguous_flag_rejects(self):
        dets = [_det(y1=i*30, y2=i*30+10) for i in range(15)]
        r = _check_bubble_coverage(dets, LAYOUT, ambiguous_capture=True)
        assert r and r['reason'] == 'multiple_sheets_detected'
    def test_ambiguous_beats_tc2(self):
        r = _check_bubble_coverage([], LAYOUT, ambiguous_capture=True)
        assert r and r['reason'] == 'multiple_sheets_detected'
    def test_count_quads_single(self):
        img = np.zeros((800, 600), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (550, 750), 255, -1)
        assert _count_large_quads(img) == 1
    def test_count_quads_two(self):
        img = np.zeros((800, 1200), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (560, 780), 255, -1)
        cv2.rectangle(img, (640, 20), (1180, 780), 255, -1)
        assert _count_large_quads(img) == 2

class TestRegressionNormalSheet:
    def test_full_sheet_not_rejected(self):
        dets = []
        for cb in range(2):
            for row in range(15):
                for opt in range(4):
                    dets.append(_det(y1=row*50, y2=row*50+10, x1=cb*300+opt*15, x2=cb*300+opt*15+12))
        assert len(dets) == EXPECTED_TOTAL
        assert _check_bubble_coverage(dets, LAYOUT) is None