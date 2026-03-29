"""
Graders for the AI Interview Simulator.

All graders are fully deterministic (no LLM calls) and return scores
in [0.0, 1.0] with partial-credit signals.

Scoring dimensions (weighted sum → total_score):
  relevance       — keyword overlap with expected topics
  structure       — STAR / step-by-step / technical structure signals
  depth           — specificity: numbers, examples, multi-part answers
  professionalism — vocabulary quality, no red-flag patterns
"""
from __future__ import annotations

import re
from typing import List

from interview_env.models import (
    DifficultyLevel,
    InterviewReward,
    InterviewStage,
)
from interview_env.tasks import InterviewQuestion, TaskDefinition


# ──────────────────────────────────────────────
# Weights per scoring dimension
# ──────────────────────────────────────────────
WEIGHTS = {
    "relevance":       0.35,
    "structure":       0.25,
    "depth":           0.25,
    "professionalism": 0.15,
}

# STAR keywords for behavioral questions
STAR_KEYWORDS = {
    "situation":  ["situation", "context", "background", "at the time", "we were", "i was"],
    "task":       ["task", "responsible", "my role", "goal", "objective", "needed to"],
    "action":     ["action", "decided", "implemented", "built", "wrote", "changed",
                   "i created", "i refactored", "i designed", "i led", "i worked"],
    "result":     ["result", "outcome", "impact", "achieved", "reduced", "improved",
                   "increased", "saved", "shipped", "delivered", "learned"],
}

# Red-flag patterns that reduce professionalism score
RED_FLAG_PATTERNS = [
    r"\b(idk|dunno|whatever|kinda|gonna|wanna|ain't)\b",
    r"\b(hate|stupid|dumb|idiot|worst)\b",
    r"(\.{4,}|!{3,}|\?{3,})",       # excessive punctuation
    r"\b(never done|no idea|can't answer)\b",
]

# Technical depth signals: numbers, measurements, version references
DEPTH_PATTERNS = [
    r"\d+(\.\d+)?(%|ms|MB|GB|TB|KB|RPS|QPS|rpm|s\b)",        # measurements
    r"\b\d{2,}\b",                                               # standalone numbers
    r"\b(v\d+|\d{4})\b",                                        # versions / years
    r"\b(O\(n\)|O\(log n\)|O\(1\)|O\(n\^2\))\b",       # big-O
    r"\b(e\.g\.|for example|such as|specifically|namely)\b",
    r"\b(first|second|third|finally|additionally|however|therefore|furthermore)\b",
    r"\b(redis|postgres|postgresql|kafka|kubernetes|docker|aws|gcp|azure)\b",
    r"\b(because|since|therefore|as a result|which means)\b",
    r"\b(trade.?off|alternative|instead|rather than|compared to|versus)\b",
    r"\b(tested|measured|monitored|profiled|benchmarked|deployed|shipped)\b",
]


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower()


def _word_count(text: str) -> int:
    return len(text.split())


def _keyword_hits(text: str, keywords: List[str]) -> int:
    low = _normalise(text)
    return sum(1 for kw in keywords if kw.lower() in low)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ──────────────────────────────────────────────
# Dimension scorers
# ──────────────────────────────────────────────

def score_relevance(response: str, question: InterviewQuestion) -> float:
    """
    Measures keyword overlap with expected topics.
    Full score if ≥ 60 % of key_topics appear.
    Partial credit proportional to fraction found.
    """
    hits = _keyword_hits(response, question.key_topics)
    fraction = hits / len(question.key_topics) if question.key_topics else 0.0
    # Scale so that hitting 60% gives ~1.0
    return _clamp(fraction / 0.60)


def score_structure(response: str, stage: InterviewStage) -> float:
    """
    For behavioral questions: check STAR coverage.
    For technical questions: check step-by-step structure.
    """
    low = _normalise(response)

    if stage in (InterviewStage.BEHAVIORAL, InterviewStage.SITUATIONAL, InterviewStage.INTRO):
        # STAR method scoring
        star_scores = []
        for component, keywords in STAR_KEYWORDS.items():
            hit = any(kw in low for kw in keywords)
            star_scores.append(1.0 if hit else 0.0)
        return _clamp(sum(star_scores) / len(star_scores))

    else:
        # Technical: reward enumeration and logical connectors
        tech_signals = [
            r"\b(first|firstly|step 1|to start)\b",
            r"\b(then|next|after that|subsequently)\b",
            r"\b(finally|lastly|in summary|overall)\b",
            r"\b(because|since|therefore|as a result|this means)\b",
            r"\b(trade.?off|alternatively|however|but|whereas)\b",
        ]
        hits = sum(1 for pat in tech_signals if re.search(pat, low))
        return _clamp(hits / len(tech_signals) / 0.60)


def score_depth(response: str, ideal_word_count: int) -> float:
    """
    Rewards:
      - Appropriate length (Goldilocks: 70 %–150 % of ideal)
      - Presence of specificity signals (numbers, examples, etc.)
    """
    wc = _word_count(response)
    # Length score
    ratio = wc / ideal_word_count if ideal_word_count > 0 else 0.0
    if ratio < 0.30:
        length_score = ratio / 0.30 * 0.3         # very short → max 0.3
    elif ratio <= 1.50:
        length_score = 0.3 + (ratio - 0.30) / 1.20 * 0.7  # grows to 1.0
    else:
        length_score = max(0.5, 1.0 - (ratio - 1.50) * 0.4)   # too long → penalise gently

    # Specificity signals
    spec_hits = sum(
        1 for pat in DEPTH_PATTERNS if re.search(pat, response, re.IGNORECASE)
    )
    spec_score = _clamp(spec_hits / len(DEPTH_PATTERNS) / 0.50)

    return _clamp(0.5 * _clamp(length_score) + 0.5 * spec_score)


def score_professionalism(response: str) -> float:
    """
    Starts at 1.0, deducts for red-flag patterns.
    Also rewards use of professional vocabulary.
    """
    low = _normalise(response)
    score = 1.0

    for pat in RED_FLAG_PATTERNS:
        if re.search(pat, low):
            score -= 0.20

    # Professional vocabulary bonus signals
    pro_words = [
        "stakeholder", "prioritise", "collaborate", "architecture",
        "scalable", "maintainable", "trade-off", "ownership",
        "impact", "metrics", "iterate", "align",
    ]
    pro_hits = _keyword_hits(response, pro_words)
    pro_bonus = min(0.10, pro_hits * 0.02)   # up to +0.10
    score += pro_bonus

    return _clamp(score)


# ──────────────────────────────────────────────
# Main grader entry point
# ──────────────────────────────────────────────

def grade_response(
    response: str,
    question: InterviewQuestion,
) -> InterviewReward:
    """
    Grade a single candidate response against a question.
    Returns a fully populated InterviewReward.
    """
    rel   = score_relevance(response, question)
    struct = score_structure(response, question.stage)
    depth = score_depth(response, question.ideal_length)
    prof  = score_professionalism(response)

    total = (
        WEIGHTS["relevance"]       * rel   +
        WEIGHTS["structure"]       * struct +
        WEIGHTS["depth"]           * depth +
        WEIGHTS["professionalism"] * prof
    )
    total = _clamp(total)

    # Build human-readable feedback
    parts = []
    if rel < 0.40:
        parts.append("Your answer missed several key topics expected for this question.")
    elif rel < 0.70:
        parts.append("Answer is partially relevant but could go deeper on key themes.")
    else:
        parts.append("Good relevance to the question.")

    if struct < 0.40:
        parts.append(
            "Structure needs work — use STAR (Situation/Task/Action/Result) for "
            "behavioral questions, or step-by-step reasoning for technical ones."
        )
    elif struct < 0.70:
        parts.append("Reasonable structure; making it more explicit would help.")
    else:
        parts.append("Well-structured response.")

    if depth < 0.40:
        parts.append("Be more specific — add numbers, real examples, or measurable outcomes.")
    elif depth < 0.70:
        parts.append("Good depth; adding concrete metrics would strengthen it.")
    else:
        parts.append("Strong specificity and depth.")

    if prof < 0.70:
        parts.append("Watch your language — aim for professional vocabulary and tone.")

    feedback = " ".join(parts) + f" (Score: {total:.2f})"

    return InterviewReward(
        total_score=round(total, 4),
        relevance=round(rel, 4),
        structure=round(struct, 4),
        depth=round(depth, 4),
        professionalism=round(prof, 4),
        feedback=feedback,
    )


def grade_episode(
    responses: List[str],
    task: TaskDefinition,
) -> dict:
    """
    Grade all responses for a complete episode.
    Returns aggregate stats and per-question breakdowns.
    """
    assert len(responses) == len(task.questions), (
        f"Expected {len(task.questions)} responses, got {len(responses)}"
    )

    per_q = []
    weighted_sum = 0.0
    total_weight = 0.0

    for resp, q in zip(responses, task.questions):
        reward = grade_response(resp, q)
        per_q.append(
            {
                "question":     q.text,
                "score":        reward.total_score,
                "relevance":    reward.relevance,
                "structure":    reward.structure,
                "depth":        reward.depth,
                "professionalism": reward.professionalism,
                "feedback":     reward.feedback,
            }
        )
        weighted_sum  += reward.total_score * q.weight
        total_weight  += q.weight

    final_score = _clamp(weighted_sum / total_weight) if total_weight > 0 else 0.0

    return {
        "task_id":       task.task_id,
        "difficulty":    task.difficulty,
        "final_score":   round(final_score, 4),
        "per_question":  per_q,
        "pass_threshold": 0.60,
        "passed":        final_score >= 0.60,
    }
