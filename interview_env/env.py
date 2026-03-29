"""
AI Interview Simulator — OpenEnv Environment
============================================

Implements the full OpenEnv interface:
  reset()  → InterviewObservation
  step()   → StepResult(observation, reward, done, info)
  state()  → InterviewState

The environment simulates a real job interview: the agent receives one
question at a time and must provide a text response. After every answer
a deterministic grader scores it (0.0–1.0) and issues partial reward.
The episode ends when all questions have been answered.

Usage:
    env = InterviewEnv(task_id="junior_behavioral")
    obs = env.reset()
    while True:
        action = InterviewAction(response="My answer …")
        result = env.step(action)
        if result.done:
            break
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from interview_env.graders import grade_response
from interview_env.models import (
    DifficultyLevel,
    InterviewAction,
    InterviewObservation,
    InterviewStage,
    InterviewState,
    QAPair,
    StepResult,
)
from interview_env.tasks import TASK_REGISTRY, TaskDefinition, get_task


class InterviewEnv:
    """
    OpenEnv-compliant AI Interview Simulator environment.

    Parameters
    ----------
    task_id : str
        One of: "junior_behavioral", "mid_technical", "senior_system_design"
    max_response_length : int
        Characters — responses exceeding this are truncated (soft limit).
    penalty_per_truncation : float
        Reward deducted when the agent's response is truncated.
    """

    VALID_TASKS = list(TASK_REGISTRY.keys())
    DEFAULT_TASK = "junior_behavioral"

    def __init__(
        self,
        task_id: str = DEFAULT_TASK,
        max_response_length: int = 2000,
        penalty_per_truncation: float = 0.05,
    ) -> None:
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Choose from: {self.VALID_TASKS}"
            )
        self._task_id              = task_id
        self._max_response_length  = max_response_length
        self._penalty_per_truncation = penalty_per_truncation

        # Internal state (initialised by reset)
        self._task: TaskDefinition        = get_task(task_id)
        self._current_index: int          = 0
        self._qa_history: list[QAPair]    = []
        self._per_q_scores: list[float]   = []
        self._done: bool                  = False
        self._total_steps: int            = 0
        self._start_time: float           = time.time()
        self._cumulative_reward: float    = 0.0

    # ──────────────────────────────────────────
    # OpenEnv Interface
    # ──────────────────────────────────────────

    def reset(self) -> InterviewObservation:
        """
        Reset the environment to the beginning of the interview.
        Returns the first question as an observation.
        """
        self._task            = get_task(self._task_id)
        self._current_index   = 0
        self._qa_history      = []
        self._per_q_scores    = []
        self._done            = False
        self._total_steps     = 0
        self._start_time      = time.time()
        self._cumulative_reward = 0.0

        return self._build_observation()

    def step(self, action: InterviewAction) -> StepResult:
        """
        Submit the agent's answer to the current interview question.

        Returns
        -------
        StepResult with:
          observation  — next question (or completion message if done)
          reward       — per-step score in [–0.05, 1.0]
          done         — True when all questions are answered
          info         — detailed grader breakdown
        """
        if self._done:
            # No-op after episode ends
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"warning": "Episode already complete. Call reset()."},
            )

        self._total_steps += 1
        current_q = self._task.questions[self._current_index]

        # Soft-truncate response and penalise
        response    = action.response
        truncated   = False
        reward_adj  = 0.0
        if len(response) > self._max_response_length:
            response  = response[: self._max_response_length]
            truncated = True
            reward_adj = -self._penalty_per_truncation

        # Grade the response
        reward_obj = grade_response(response, current_q)
        step_reward = reward_obj.total_score + reward_adj
        step_reward = max(0.0, min(1.0, step_reward))

        # Record in history
        qa = QAPair(
            question        = current_q.text,
            answer          = response,
            stage           = current_q.stage,
            question_score  = reward_obj.total_score,
        )
        self._qa_history.append(qa)
        self._per_q_scores.append(reward_obj.total_score)
        self._cumulative_reward += step_reward

        # Advance to next question
        self._current_index += 1
        if self._current_index >= len(self._task.questions):
            self._done = True

        obs  = self._build_observation()
        info: Dict[str, Any] = {
            "question_answered": current_q.text,
            "grader_breakdown": {
                "relevance":       reward_obj.relevance,
                "structure":       reward_obj.structure,
                "depth":           reward_obj.depth,
                "professionalism": reward_obj.professionalism,
                "total_score":     reward_obj.total_score,
            },
            "feedback":       reward_obj.feedback,
            "truncated":      truncated,
            "question_number": self._current_index,   # 1-indexed (just answered)
            "total_questions": len(self._task.questions),
        }

        if self._done:
            final_score = self._final_score()
            info["episode_summary"] = {
                "final_score":       final_score,
                "per_question_scores": self._per_q_scores,
                "passed":            final_score >= 0.60,
                "pass_threshold":    0.60,
            }

        return StepResult(
            observation = obs,
            reward      = round(step_reward, 4),
            done        = self._done,
            info        = info,
        )

    def state(self) -> InterviewState:
        """
        Return a serialisable snapshot of the full internal state.
        Useful for debugging, checkpointing, and OpenEnv validation.
        """
        return InterviewState(
            task_id          = self._task_id,
            difficulty       = self._task.difficulty,
            job_title        = self._task.job_title,
            company_context  = self._task.company_context,
            questions        = [q.text for q in self._task.questions],
            stages           = [q.stage for q in self._task.questions],
            current_index    = self._current_index,
            qa_history       = self._qa_history,
            per_question_scores = self._per_q_scores,
            episode_done     = self._done,
            total_steps      = self._total_steps,
            start_time       = self._start_time,
            cumulative_reward = self._cumulative_reward,
        )

    # ──────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def available_tasks(self) -> list[str]:
        return self.VALID_TASKS

    @property
    def action_space_description(self) -> str:
        return (
            "InterviewAction(response: str) — "
            "A free-text string (1–2000 chars) representing the candidate's spoken answer."
        )

    @property
    def observation_space_description(self) -> str:
        return (
            "InterviewObservation with fields: "
            "task_id, job_title, company_context, difficulty, "
            "current_question, question_number, total_questions, stage, "
            "previous_qa (list of QAPair), cumulative_score, time_elapsed, interview_complete."
        )

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _build_observation(self) -> InterviewObservation:
        if self._done or self._current_index >= len(self._task.questions):
            current_question   = (
                "Thank you for completing the interview. "
                "We will be in touch shortly."
            )
            stage              = InterviewStage.COMPLETE
            question_number    = len(self._task.questions)
            interview_complete = True
        else:
            q                  = self._task.questions[self._current_index]
            current_question   = q.text
            stage              = q.stage
            question_number    = self._current_index + 1
            interview_complete = False

        return InterviewObservation(
            task_id            = self._task_id,
            job_title          = self._task.job_title,
            company_context    = self._task.company_context,
            difficulty         = self._task.difficulty,
            current_question   = current_question,
            question_number    = question_number,
            total_questions    = len(self._task.questions),
            stage              = stage,
            previous_qa        = list(self._qa_history),
            cumulative_score   = round(self._cumulative_reward, 4),
            time_elapsed       = round(time.time() - self._start_time, 2),
            interview_complete = interview_complete,
        )

    def _final_score(self) -> float:
        """Weighted average over all per-question scores."""
        if not self._per_q_scores:
            return 0.0
        questions = self._task.questions
        weighted  = sum(
            s * q.weight
            for s, q in zip(self._per_q_scores, questions)
        )
        total_w = sum(q.weight for q in questions)
        return round(weighted / total_w, 4) if total_w > 0 else 0.0
