"""
Task definitions for the AI Interview Simulator.

Three tasks with increasing difficulty:
  Task 1 — junior_behavioral   (EASY)
  Task 2 — mid_technical       (MEDIUM)
  Task 3 — senior_system_design (HARD)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from interview_env.models import DifficultyLevel, InterviewStage


@dataclass
class InterviewQuestion:
    text:           str
    stage:          InterviewStage
    key_topics:     List[str]          # expected themes / keywords
    ideal_length:   int = 150          # target word count (approximate)
    weight:         float = 1.0        # relative importance within the task


@dataclass
class TaskDefinition:
    task_id:         str
    name:            str
    difficulty:      DifficultyLevel
    job_title:       str
    company_context: str
    questions:       List[InterviewQuestion]
    description:     str

    @property
    def total_questions(self) -> int:
        return len(self.questions)


# ──────────────────────────────────────────────────────────────────────────────
# TASK 1 — Junior Behavioral Interview  (EASY)
# ──────────────────────────────────────────────────────────────────────────────
TASK_JUNIOR_BEHAVIORAL = TaskDefinition(
    task_id    = "junior_behavioral",
    name       = "Junior Software Engineer — Behavioral Interview",
    difficulty = DifficultyLevel.EASY,
    job_title  = "Junior Software Engineer",
    company_context = (
        "TechStart Inc. is a Series-A startup building developer tools. "
        "The team is 20 engineers, culture is collaborative, and they value "
        "learning, ownership, and clear communication."
    ),
    description = (
        "Entry-level behavioral interview with 4 questions. "
        "Answers are evaluated on the STAR framework (Situation, Task, "
        "Action, Result), relevance, and professionalism. "
        "Good answers are 100–200 words with concrete examples."
    ),
    questions = [
        InterviewQuestion(
            text = (
                "Tell me about yourself and why you're interested in "
                "this Junior Software Engineer role at TechStart."
            ),
            stage      = InterviewStage.INTRO,
            key_topics = [
                "background", "education", "experience", "motivation",
                "interest", "skills", "goal", "passionate", "learn",
            ],
            ideal_length = 120,
            weight       = 0.8,
        ),
        InterviewQuestion(
            text = (
                "Describe a time when you had to debug a challenging problem. "
                "What was the situation, what did you try, and what did you learn?"
            ),
            stage      = InterviewStage.BEHAVIORAL,
            key_topics = [
                "debug", "problem", "error", "solution", "fixed", "learned",
                "log", "test", "stack trace", "isolate", "reproduce",
                "situation", "result", "action",
            ],
            ideal_length = 180,
            weight       = 1.2,
        ),
        InterviewQuestion(
            text = (
                "Tell me about a project you're proud of. "
                "What did you build, what was your role, and what impact did it have?"
            ),
            stage      = InterviewStage.BEHAVIORAL,
            key_topics = [
                "project", "built", "developed", "designed", "team",
                "role", "impact", "users", "result", "outcome",
                "feature", "deployed", "shipped",
            ],
            ideal_length = 180,
            weight       = 1.0,
        ),
        InterviewQuestion(
            text = (
                "How do you handle feedback on your code during a code review? "
                "Give a specific example."
            ),
            stage      = InterviewStage.BEHAVIORAL,
            key_topics = [
                "feedback", "code review", "comment", "improve", "change",
                "open", "learn", "collaborate", "understand", "appreciate",
                "review", "PR", "pull request",
            ],
            ideal_length = 150,
            weight       = 1.0,
        ),
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# TASK 2 — Mid-Level Full-Stack Technical Interview  (MEDIUM)
# ──────────────────────────────────────────────────────────────────────────────
TASK_MID_TECHNICAL = TaskDefinition(
    task_id    = "mid_technical",
    name       = "Mid-Level Full-Stack Engineer — Technical Interview",
    difficulty = DifficultyLevel.MEDIUM,
    job_title  = "Mid-Level Full-Stack Engineer",
    company_context = (
        "FinFlow Ltd. builds B2B fintech SaaS used by 500+ companies. "
        "Stack: React, Node.js, PostgreSQL, AWS. "
        "The team ships weekly and values reliability, code quality, and velocity."
    ),
    description = (
        "5-question interview mixing technical knowledge and behavioral depth. "
        "Technical questions expect accurate, specific answers with trade-offs. "
        "Behavioral questions require the STAR method with measurable results. "
        "Good answers are 150–250 words with concrete examples and numbers."
    ),
    questions = [
        InterviewQuestion(
            text = (
                "Explain the difference between REST and GraphQL APIs. "
                "When would you choose one over the other?"
            ),
            stage      = InterviewStage.TECHNICAL,
            key_topics = [
                "rest", "graphql", "endpoint", "query", "schema",
                "over-fetching", "under-fetching", "flexibility",
                "mutation", "http", "stateless", "client", "tradeoff",
            ],
            ideal_length = 200,
            weight       = 1.0,
        ),
        InterviewQuestion(
            text = (
                "How does the JavaScript event loop work? "
                "Explain the call stack, task queue, and microtask queue."
            ),
            stage      = InterviewStage.TECHNICAL,
            key_topics = [
                "event loop", "call stack", "task queue", "microtask",
                "asynchronous", "promise", "setTimeout", "blocking",
                "non-blocking", "single thread", "callback", "execution",
            ],
            ideal_length = 200,
            weight       = 1.2,
        ),
        InterviewQuestion(
            text = (
                "Describe your approach to database query optimisation. "
                "Walk me through how you'd diagnose and fix a slow SQL query."
            ),
            stage      = InterviewStage.TECHNICAL,
            key_topics = [
                "index", "explain", "query plan", "N+1", "join",
                "slow query log", "cache", "pagination", "analyse",
                "execution plan", "select", "table scan", "optimise",
            ],
            ideal_length = 220,
            weight       = 1.2,
        ),
        InterviewQuestion(
            text = (
                "Tell me about a time you had to meet a tight deadline "
                "without sacrificing quality. What trade-offs did you make?"
            ),
            stage      = InterviewStage.BEHAVIORAL,
            key_topics = [
                "deadline", "priority", "scope", "tradeoff", "quality",
                "technical debt", "stakeholder", "communicate", "plan",
                "shipped", "delivered", "result",
            ],
            ideal_length = 180,
            weight       = 1.0,
        ),
        InterviewQuestion(
            text = (
                "How do you ensure your front-end React components remain "
                "performant as the application scales?"
            ),
            stage      = InterviewStage.TECHNICAL,
            key_topics = [
                "memo", "useMemo", "useCallback", "virtualisation",
                "lazy load", "code split", "re-render", "profiler",
                "bundle size", "lighthouse", "key", "state", "performance",
            ],
            ideal_length = 200,
            weight       = 0.9,
        ),
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# TASK 3 — Senior Engineer System Design + Leadership  (HARD)
# ──────────────────────────────────────────────────────────────────────────────
TASK_SENIOR_SYSTEM_DESIGN = TaskDefinition(
    task_id    = "senior_system_design",
    name       = "Senior Software Engineer — System Design & Leadership",
    difficulty = DifficultyLevel.HARD,
    job_title  = "Senior Software Engineer",
    company_context = (
        "ScaleUp Corp is a Series-C e-commerce platform processing 50 million "
        "orders per year. Stack: microservices, Kubernetes, Kafka, "
        "PostgreSQL, Redis, AWS. "
        "Engineers are expected to own systems end-to-end, mentor juniors, "
        "and drive architectural decisions."
    ),
    description = (
        "6-question interview covering system design, distributed systems, "
        "leadership, and conflict resolution. "
        "System design answers must include scalability, fault tolerance, "
        "and trade-offs. Leadership answers must show influence and impact. "
        "Good answers are 200–350 words with specifics, numbers, and alternatives considered."
    ),
    questions = [
        InterviewQuestion(
            text = (
                "Design a distributed rate limiter that can handle "
                "100,000 requests per second across multiple servers. "
                "Walk me through your architecture, storage choice, and failure modes."
            ),
            stage      = InterviewStage.TECHNICAL,
            key_topics = [
                "redis", "token bucket", "sliding window", "fixed window",
                "distributed", "race condition", "atomic", "lua script",
                "consistency", "availability", "latency", "failure",
                "fallback", "partition", "replication", "cache",
            ],
            ideal_length = 300,
            weight       = 1.5,
        ),
        InterviewQuestion(
            text = (
                "How would you design the order processing system for "
                "an e-commerce platform that must guarantee no orders are lost, "
                "even if services go down? Focus on event-driven design."
            ),
            stage      = InterviewStage.TECHNICAL,
            key_topics = [
                "kafka", "message queue", "idempotent", "outbox pattern",
                "saga", "event sourcing", "retry", "dead letter queue",
                "exactly once", "at least once", "transaction", "durability",
                "consumer group", "offset", "ack",
            ],
            ideal_length = 300,
            weight       = 1.5,
        ),
        InterviewQuestion(
            text = (
                "Tell me about the most technically complex system you've "
                "designed or owned. What were the biggest risks, and how did you "
                "mitigate them in production?"
            ),
            stage      = InterviewStage.BEHAVIORAL,
            key_topics = [
                "designed", "owned", "risk", "mitigation", "production",
                "monitoring", "alert", "rollback", "canary", "feature flag",
                "incident", "postmortem", "scale", "bottleneck",
            ],
            ideal_length = 280,
            weight       = 1.3,
        ),
        InterviewQuestion(
            text = (
                "Describe a situation where you disagreed with your tech lead "
                "or manager on a technical decision. How did you handle it, "
                "and what was the outcome?"
            ),
            stage      = InterviewStage.SITUATIONAL,
            key_topics = [
                "disagree", "decision", "data", "evidence", "compromise",
                "respected", "escalate", "influence", "RFC", "proposal",
                "outcome", "learned", "relationship", "align",
            ],
            ideal_length = 220,
            weight       = 1.1,
        ),
        InterviewQuestion(
            text = (
                "How do you approach mentoring junior engineers? "
                "Give a concrete example of how you helped someone grow significantly."
            ),
            stage      = InterviewStage.BEHAVIORAL,
            key_topics = [
                "mentor", "junior", "growth", "feedback", "pair programming",
                "review", "goal", "1:1", "autonomy", "confidence",
                "promoted", "improved", "impact", "coaching",
            ],
            ideal_length = 220,
            weight       = 1.0,
        ),
        InterviewQuestion(
            text = (
                "ScaleUp's checkout latency has increased by 300ms over the last "
                "month after a Kubernetes cluster upgrade. Walk me through exactly "
                "how you would diagnose and resolve this in production."
            ),
            stage      = InterviewStage.TECHNICAL,
            key_topics = [
                "apm", "tracing", "jaeger", "datadog", "prometheus",
                "metrics", "latency", "p99", "kubernetes", "resource limit",
                "cpu", "memory", "network", "service mesh", "ingress",
                "rollback", "profiling", "flamegraph", "baseline",
            ],
            ideal_length = 300,
            weight       = 1.5,
        ),
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────
TASK_REGISTRY: dict[str, TaskDefinition] = {
    TASK_JUNIOR_BEHAVIORAL.task_id:    TASK_JUNIOR_BEHAVIORAL,
    TASK_MID_TECHNICAL.task_id:        TASK_MID_TECHNICAL,
    TASK_SENIOR_SYSTEM_DESIGN.task_id: TASK_SENIOR_SYSTEM_DESIGN,
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]
