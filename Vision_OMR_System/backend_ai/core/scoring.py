"""
scoring.py
==========
Spatial mapping + answer-key comparison for OMR bubble grading.

Responsibilities:
  1. Map classified bubble detections to (question_number, option_letter)
     based on their spatial position in the warped image.
  2. Compare detected answers against an answer key.
  3. Produce a per-question score breakdown.

The spatial layout is configurable via `SheetLayout`.  The default assumes
a standard 30-question × 4-option (A/B/C/D) OMR sheet arranged in a single
column block.  Override with your own layout as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .classification import BubbleState, ClassificationResult


# ---------------------------------------------------------------------------
# Sheet layout configuration
# ---------------------------------------------------------------------------

@dataclass
class SheetLayout:
    """
    Describes the physical layout of bubbles on the OMR sheet.

    Attributes
    ----------
    questions_per_column : int
        Number of questions stacked vertically in one column block.
    num_columns : int
        Number of column blocks side-by-side on the sheet.
    options : str
        String of option letters, e.g. "ABCD" for a 4-option sheet.
    """
    questions_per_column: int = 30
    num_columns: int = 1
    options: str = "ABCD"

    @property
    def total_questions(self) -> int:
        return self.questions_per_column * self.num_columns

    @property
    def num_options(self) -> int:
        return len(self.options)

    @property
    def total_bubbles(self) -> int:
        return self.total_questions * self.num_options


# Default layout — override per exam if needed
DEFAULT_LAYOUT = SheetLayout()


# ---------------------------------------------------------------------------
# Mapped answer types
# ---------------------------------------------------------------------------

class AnswerStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNANSWERED = "unanswered"
    MULTIPLE_MARKED = "multiple_marked"
    AMBIGUOUS = "ambiguous"


@dataclass
class QuestionResult:
    """Result for a single question after comparing against the answer key."""
    question_number: int
    marked_options: List[str]       # e.g. ["B"] or ["A", "C"] if multi-marked
    correct_option: Optional[str]   # from answer key, None if no key provided
    status: AnswerStatus
    has_ambiguous: bool = False     # True if any bubble in this row was ambiguous


@dataclass
class ScoreReport:
    """Aggregate scoring output."""
    total_questions: int
    answered: int
    correct: int
    incorrect: int
    unanswered: int
    multiple_marked: int
    ambiguous: int
    score_percent: float
    per_question: List[QuestionResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spatial mapping
# ---------------------------------------------------------------------------

def map_bubbles_to_grid(
    classifications: List[ClassificationResult],
    layout: SheetLayout = DEFAULT_LAYOUT,
) -> Dict[int, Dict[str, ClassificationResult]]:
    """
    Assign each classified bubble to a (question, option) cell.

    Strategy:
      1. Sort all bubble detections by their centre-x to separate column
         blocks (if multiple columns).
      2. Within each column block, sort by centre-y to determine question
         row ordering.
      3. Within each question row, sort by centre-x to assign option
         letters (A, B, C, D …).

    Returns a nested dict:  { question_number: { "A": result, "B": result, … } }
    """
    if not classifications:
        return {}

    # Compute centre coordinates for each classification
    items = []
    for cr in classifications:
        cx = (cr.detection.x1 + cr.detection.x2) / 2
        cy = (cr.detection.y1 + cr.detection.y2) / 2
        items.append((cx, cy, cr))

    # ── Split into column blocks by x-coordinate ────────────────────────
    items.sort(key=lambda t: t[0])  # sort by cx

    col_blocks: List[List[Tuple[float, float, ClassificationResult]]] = []
    if layout.num_columns == 1:
        col_blocks = [items]
    else:
        # Divide into N roughly equal groups by x-position
        block_size = len(items) // layout.num_columns
        for i in range(layout.num_columns):
            start = i * block_size
            end = start + block_size if i < layout.num_columns - 1 else len(items)
            col_blocks.append(items[start:end])

    # ── Within each column, sort by y → split into question rows ────────
    grid: Dict[int, Dict[str, ClassificationResult]] = {}
    q_offset = 0

    for block in col_blocks:
        block.sort(key=lambda t: t[1])  # sort by cy (top to bottom)

        # Group into rows of `num_options` bubbles each
        num_opts = layout.num_options
        for row_idx in range(0, len(block), num_opts):
            row_items = block[row_idx : row_idx + num_opts]
            # Sort within row by x (left to right) → option order
            row_items.sort(key=lambda t: t[0])

            q_num = q_offset + (row_idx // num_opts) + 1
            grid[q_num] = {}
            for opt_idx, (_, _, cr) in enumerate(row_items):
                if opt_idx < len(layout.options):
                    option_letter = layout.options[opt_idx]
                    grid[q_num][option_letter] = cr

        q_offset += layout.questions_per_column

    return grid


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def score_sheet(
    classifications: List[ClassificationResult],
    answer_key: Optional[Dict[int, str]] = None,
    layout: SheetLayout = DEFAULT_LAYOUT,
) -> ScoreReport:
    """
    Grade a complete OMR sheet.

    Parameters
    ----------
    classifications : list of ClassificationResult
        Output from `classify_all()`.
    answer_key : dict, optional
        Mapping of question_number → correct option letter, e.g. {1: "A", 2: "C"}.
        If None, scoring reports marked options without correctness judgement.
    layout : SheetLayout
        Physical layout descriptor.

    Returns
    -------
    ScoreReport
    """
    grid = map_bubbles_to_grid(classifications, layout)

    per_question: List[QuestionResult] = []
    correct = incorrect = unanswered = multiple_marked = ambiguous = 0

    total_questions = layout.total_questions if answer_key else len(grid)

    for q_num in range(1, total_questions + 1):
        options = grid.get(q_num, {})
        correct_option = answer_key.get(q_num) if answer_key else None

        # Determine which options were filled
        marked: List[str] = []
        has_ambig = False
        for opt_letter, cr in options.items():
            if cr.state == BubbleState.FILLED:
                marked.append(opt_letter)
            elif cr.state == BubbleState.AMBIGUOUS:
                has_ambig = True

        # Determine status
        if has_ambig and not marked:
            status = AnswerStatus.AMBIGUOUS
            ambiguous += 1
        elif len(marked) == 0:
            status = AnswerStatus.UNANSWERED
            unanswered += 1
        elif len(marked) > 1:
            status = AnswerStatus.MULTIPLE_MARKED
            multiple_marked += 1
        elif answer_key is not None and correct_option is not None:
            if marked[0] == correct_option:
                status = AnswerStatus.CORRECT
                correct += 1
            else:
                status = AnswerStatus.INCORRECT
                incorrect += 1
        else:
            # No answer key — just record as answered
            status = AnswerStatus.CORRECT  # neutral when no key
            correct += 1

        per_question.append(
            QuestionResult(
                question_number=q_num,
                marked_options=marked,
                correct_option=correct_option,
                status=status,
                has_ambiguous=has_ambig,
            )
        )

    answered = correct + incorrect + multiple_marked
    score_pct = (correct / total_questions * 100) if total_questions > 0 else 0.0

    return ScoreReport(
        total_questions=total_questions,
        answered=answered,
        correct=correct,
        incorrect=incorrect,
        unanswered=unanswered,
        multiple_marked=multiple_marked,
        ambiguous=ambiguous,
        score_percent=round(score_pct, 2),
        per_question=per_question,
    )
