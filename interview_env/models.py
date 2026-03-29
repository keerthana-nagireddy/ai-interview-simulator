"""
Typed Pydantic models for the AI Interview Simulator OpenEnv environment.
Defines Observation, Action, Reward, and State structures.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class InterviewStage(str, Enum):
    INTRO       = "intro"
    BEHAVIORAL  = "behavioral"
    TECHNICAL   = "technical"
    SITUATIONAL = "situational"
    CLOSING     = "closing"
    COMPLETE    = "complete"


class DifficultyLevel(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ──────────────────────────────────────────────
# Core OpenEnv Models
# ──────────────────────────────────────────────

class QAPair(BaseModel):
    """A single question + the candidate's answer."""
    question:      str
    answer:        str
    stage:         InterviewStage
    question_score: Optional[float] = None   # filled after grading


class InterviewObservation(BaseModel):
    """
    Observation returned by reset() and step().
    Contains everything the agent needs to formulate its next response.
    """
    task_id:           str
    job_title:         str
    company_context:   str
    difficulty:        DifficultyLevel

    current_question:  str
    question_number:   int                    # 1-indexed
    total_questions:   int
    stage:             InterviewStage

    previous_qa:       List[QAPair] = Field(default_factory=list)
    cumulative_score:  float = 0.0
    time_elapsed:      float = 0.0            # seconds since reset
    interview_complete: bool = False


class InterviewAction(BaseModel):
    """
    Action submitted by the agent — the candidate's spoken response.
    """
    response: str = Field(
        ...,
        min_length=1,
        description="The candidate's answer to the current interview question.",
    )


class InterviewReward(BaseModel):
    """
    Detailed reward breakdown for a single step.
    All sub-scores are in [0.0, 1.0]; total_score is their weighted sum.
    """
    total_score:      float   # 0.0 – 1.0, primary signal
    relevance:        float   # Does the answer address the question?
    structure:        float   # Organised response (STAR / step-by-step)?
    depth:            float   # Specificity, examples, numbers
    professionalism:  float   # Tone, vocabulary, no red flags

    feedback:         str     # Human-readable grader feedback


class InterviewState(BaseModel):
    """
    Full internal state — returned by state().
    Useful for debugging, checkpointing, and OpenEnv validators.
    """
    task_id:           str
    difficulty:        DifficultyLevel
    job_title:         str
    company_context:   str

    questions:         List[str]             # All questions in order
    stages:            List[InterviewStage]  # Stage per question
    current_index:     int                   # Which question we're on

    qa_history:        List[QAPair]
    per_question_scores: List[float]

    episode_done:      bool
    total_steps:       int
    start_time:        float
    cumulative_reward: float


class StepResult(BaseModel):
    """Return type of env.step()."""
    observation: InterviewObservation
    reward:      float                    # scalar for RL loop
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)
