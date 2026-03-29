"""
validate.py — OpenEnv Spec Compliance Validator
================================================

Run with:   python validate.py
Checks:
  1. All Pydantic models have correct fields and types
  2. reset() returns a valid InterviewObservation
  3. step() returns a valid StepResult with reward in range
  4. state() returns a valid InterviewState
  5. All 3 tasks complete full episodes
  6. Reward is always in [-0.05, 1.0]
  7. done=True only after final question
  8. Partial rewards are non-trivial (not all same score)
  9. Episode boundary: step() after done returns done=True
 10. openenv.yaml parses correctly and has required fields
"""
from __future__ import annotations

import sys
import yaml
from pathlib import Path
from typing import List

from interview_env import (
    InterviewEnv,
    InterviewAction,
    InterviewObservation,
    InterviewState,
    StepResult,
    TASK_REGISTRY,
)
from interview_env.models import InterviewReward

PASS = "  ✅"
FAIL = "  ❌"
results: List[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"{status}  {name}" + (f"  ({detail})" if detail else ""))


def run_checks():
    print("\n══════════════════════════════════════════════")
    print("  OpenEnv Validator — AI Interview Simulator")
    print("══════════════════════════════════════════════\n")

    # ── 1. Model field existence ───────────────────────────────────────────
    print("── Model structure ──")
    obs_fields = InterviewObservation.model_fields.keys()
    check("InterviewObservation has current_question",   "current_question"   in obs_fields)
    check("InterviewObservation has task_id",            "task_id"            in obs_fields)
    check("InterviewObservation has stage",              "stage"              in obs_fields)
    check("InterviewObservation has difficulty",         "difficulty"         in obs_fields)
    check("InterviewObservation has interview_complete", "interview_complete" in obs_fields)

    reward_fields = InterviewReward.model_fields.keys()
    check("InterviewReward has total_score",      "total_score"      in reward_fields)
    check("InterviewReward has relevance",        "relevance"        in reward_fields)
    check("InterviewReward has structure",        "structure"        in reward_fields)
    check("InterviewReward has depth",            "depth"            in reward_fields)
    check("InterviewReward has professionalism",  "professionalism"  in reward_fields)
    check("InterviewReward has feedback",         "feedback"         in reward_fields)

    state_fields = InterviewState.model_fields.keys()
    check("InterviewState has qa_history",     "qa_history"     in state_fields)
    check("InterviewState has episode_done",   "episode_done"   in state_fields)
    check("InterviewState has cumulative_reward", "cumulative_reward" in state_fields)

    step_fields = StepResult.model_fields.keys()
    check("StepResult has observation", "observation" in step_fields)
    check("StepResult has reward",      "reward"      in step_fields)
    check("StepResult has done",        "done"        in step_fields)
    check("StepResult has info",        "info"        in step_fields)
    print()

    # ── 2. reset() contract ───────────────────────────────────────────────
    print("── reset() contract ──")
    env = InterviewEnv(task_id="junior_behavioral")
    obs = env.reset()
    check("reset() returns InterviewObservation",        isinstance(obs, InterviewObservation))
    check("reset() sets interview_complete=False",       obs.interview_complete is False)
    check("reset() question_number starts at 1",         obs.question_number == 1)
    check("reset() cumulative_score starts at 0.0",      obs.cumulative_score == 0.0)
    check("reset() previous_qa is empty",                obs.previous_qa == [])
    check("reset() current_question is non-empty string",isinstance(obs.current_question, str) and len(obs.current_question) > 5)

    # Double-reset should restart cleanly
    obs2 = env.reset()
    check("double reset() restarts cleanly",             obs2.question_number == 1 and obs2.cumulative_score == 0.0)
    print()

    # ── 3. step() contract ───────────────────────────────────────────────
    print("── step() contract ──")
    env = InterviewEnv(task_id="junior_behavioral")
    env.reset()
    action = InterviewAction(response="I once worked on a project where I fixed a bug.")
    result = env.step(action)

    check("step() returns StepResult",          isinstance(result, StepResult))
    check("step() reward in [-0.05, 1.0]",      -0.05 <= result.reward <= 1.0, f"got {result.reward}")
    check("step() done is bool",                 isinstance(result.done, bool))
    check("step() observation is InterviewObservation", isinstance(result.observation, InterviewObservation))
    check("step() info is dict",                 isinstance(result.info, dict))
    check("step() info has grader_breakdown",    "grader_breakdown" in result.info)
    check("step() info has feedback",            "feedback" in result.info)
    check("step() info grader has 4 dimensions", all(
        k in result.info["grader_breakdown"]
        for k in ["relevance", "structure", "depth", "professionalism"]
    ))
    print()

    # ── 4. state() contract ──────────────────────────────────────────────
    print("── state() contract ──")
    state = env.state()
    check("state() returns InterviewState",      isinstance(state, InterviewState))
    check("state() is JSON-serialisable",        bool(state.model_dump()))
    check("state() qa_history has 1 entry",      len(state.qa_history) == 1)
    check("state() episode_done is False mid-episode", state.episode_done is False)
    print()

    # ── 5. Full episode for all 3 tasks ──────────────────────────────────
    print("── Full episode per task ──")
    ANSWERS = {
        "easy":   (
            "In that situation I was on a team building a new feature. "
            "My task was to implement the backend API endpoint. "
            "I took action by writing clean code with tests and documentation. "
            "The result was a successful delivery appreciated by stakeholders."
        ),
        "medium": (
            "First, I would analyse the architecture and identify bottlenecks. "
            "The trade-off between REST and GraphQL is flexibility vs simplicity. "
            "I used Redis for caching and PostgreSQL with proper indexing to improve "
            "query performance. The result reduced latency by 40ms on the p99."
        ),
        "hard":   (
            "For this distributed system design, first I would implement a Redis-based "
            "sliding window rate limiter with Lua scripts for atomicity. "
            "The architecture uses sharded Redis cluster with replication for fault tolerance. "
            "Trade-off: sliding window uses more memory but provides accurate rate limiting. "
            "Fallback: local token bucket when Redis is partitioned. "
            "Result: handles 100k RPS with sub-millisecond overhead."
        ),
    }

    for task_id, task in TASK_REGISTRY.items():
        env = InterviewEnv(task_id=task_id)
        obs = env.reset()
        rewards = []
        steps = 0

        while not obs.interview_complete:
            answer  = ANSWERS[task.difficulty.value]
            result  = env.step(InterviewAction(response=answer))
            rewards.append(result.reward)
            steps  += 1
            obs     = result.observation

        final_state = env.state()
        episode_score = sum(rewards) / len(rewards) if rewards else 0.0

        check(
            f"[{task_id}] completes in {task.total_questions} steps",
            steps == task.total_questions,
            f"got {steps}",
        )
        check(
            f"[{task_id}] episode_done=True after last step",
            final_state.episode_done,
        )
        check(
            f"[{task_id}] all rewards in range",
            all(-0.05 <= r <= 1.0 for r in rewards),
            f"rewards={[round(r,2) for r in rewards]}",
        )
        check(
            f"[{task_id}] partial rewards (not all identical)",
            len(set(round(r, 3) for r in rewards)) > 1 or steps == 1,
            f"score={episode_score:.3f}",
        )
        check(
            f"[{task_id}] qa_history length matches steps",
            len(final_state.qa_history) == steps,
        )
    print()

    # ── 6. Episode boundary ───────────────────────────────────────────────
    print("── Episode boundary ──")
    env = InterviewEnv(task_id="junior_behavioral")
    env.reset()
    for _ in range(10):  # exhaust all questions (4 + extra)
        r = env.step(InterviewAction(response="test answer here for boundary check"))
    check("step() after done still returns done=True", r.done is True)
    check("step() after done reward is 0.0",           r.reward == 0.0)
    print()

    # ── 7. Reward shaping — quality sensitivity ───────────────────────────
    print("── Reward shaping quality ──")
    from interview_env.graders import grade_response
    from interview_env.tasks import TASK_JUNIOR_BEHAVIORAL
    q = TASK_JUNIOR_BEHAVIORAL.questions[1]  # debug question

    bad   = grade_response("idk", q).total_score
    med   = grade_response(
        "I debugged a problem once by checking the logs and fixed it.", q
    ).total_score
    good  = grade_response(
        "In a situation where our API was returning 500 errors, my task was to diagnose "
        "the root cause. I took action by reproducing the bug locally, adding debug logging, "
        "and isolating it to a race condition in the database connection pool. "
        "I fixed it by implementing proper connection timeouts. The result was zero errors "
        "in production and I learned to always add integration tests for concurrency.", q
    ).total_score

    check("Bad answer scores lower than medium",  bad  < med,  f"{bad:.3f} < {med:.3f}")
    check("Medium answer scores lower than good", med  < good, f"{med:.3f} < {good:.3f}")
    check("Bad answer score < 0.35",              bad  < 0.35, f"got {bad:.3f}")
    check("Good answer score > 0.65",             good > 0.65, f"got {good:.3f}")
    print()

    # ── 8. openenv.yaml ───────────────────────────────────────────────────
    print("── openenv.yaml ──")
    yaml_path = Path("openenv.yaml")
    check("openenv.yaml exists", yaml_path.exists())
    if yaml_path.exists():
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        check("yaml has 'name'",    "name"    in meta)
        check("yaml has 'version'", "version" in meta)
        check("yaml has 'tasks'",   "tasks"   in meta)
        check("yaml has 3+ tasks",  len(meta.get("tasks", [])) >= 3)
        check("yaml has reward spec", "reward" in meta)
        check("yaml has interface",   "interface" in meta)
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    failed = [name for name, ok, _ in results if not ok]

    print("══════════════════════════════════════════════")
    print(f"  Result: {passed}/{total} checks passed")
    if failed:
        print(f"\n  FAILED:")
        for name in failed:
            print(f"    ❌  {name}")
    else:
        print("  🎉 All checks passed! Environment is OpenEnv compliant.")
    print("══════════════════════════════════════════════\n")

    return len(failed) == 0


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
