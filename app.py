"""
app.py — Hugging Face Spaces entry point for the AI Interview Simulator.

Exposes:
  1. A FastAPI REST API that implements the OpenEnv HTTP spec
     (POST /reset, POST /step, GET /state, GET /tasks, GET /health)
  2. A Gradio UI so humans and evaluators can interact with the environment
     directly in the browser.

Both run on the same process via gr.mount_gradio_app.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from interview_env import InterviewEnv, InterviewAction, TASK_REGISTRY
from interview_env.models import InterviewObservation, InterviewState, StepResult


# ──────────────────────────────────────────────
# Global environment registry (one per session)
# In production you'd use proper session management.
# For hackathon / HF Space a single shared env is fine.
# ──────────────────────────────────────────────

_envs: Dict[str, InterviewEnv] = {}


def _get_or_create_env(task_id: str) -> InterviewEnv:
    if task_id not in _envs:
        _envs[task_id] = InterviewEnv(task_id=task_id)
    return _envs[task_id]


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

api = FastAPI(
    title="AI Interview Simulator — OpenEnv",
    description=(
        "An OpenEnv-compliant environment that simulates real job interviews. "
        "Agents learn to answer interview questions and receive structured feedback."
    ),
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetRequest(BaseModel):
    task_id: str = "junior_behavioral"


class StepRequest(BaseModel):
    task_id:  str
    response: str


@api.get("/health")
def health():
    return {"status": "ok", "environment": "ai-interview-simulator"}


@api.get("/tasks")
def list_tasks():
    return {
        tid: {
            "name":        t.name if hasattr(t, "name") else tid,
            "difficulty":  TASK_REGISTRY[tid].difficulty,
            "questions":   TASK_REGISTRY[tid].total_questions,
            "description": TASK_REGISTRY[tid].description,
        }
        for tid, t in TASK_REGISTRY.items()
    }


@api.post("/reset")
def reset_env(req: ResetRequest):
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")
    env = _get_or_create_env(req.task_id)
    obs = env.reset()
    return obs.model_dump()


@api.post("/step")
def step_env(req: StepRequest):
    if req.task_id not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for task '{req.task_id}'. Call /reset first.",
        )
    env = _envs[req.task_id]
    action = InterviewAction(response=req.response)
    result = env.step(action)
    return result.model_dump()


@api.get("/state")
def get_state(task_id: str = "junior_behavioral"):
    if task_id not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for task '{task_id}'. Call /reset first.",
        )
    return _envs[task_id].state().model_dump()


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

def _format_history(history: List) -> str:
    if not history:
        return ""
    lines = []
    for h in history:
        lines.append(f"**Q:** {h[0]}")
        lines.append(f"**A:** {h[1]}")
        lines.append("---")
    return "\n".join(lines)


def gradio_reset(task_id: str):
    env = _get_or_create_env(task_id)
    obs = env.reset()
    question = obs.current_question
    status   = (
        f"**Interview started!**\n\n"
        f"**Role:** {obs.job_title}\n\n"
        f"**Company:** {obs.company_context}\n\n"
        f"**Difficulty:** {obs.difficulty}\n\n"
        f"**Questions:** {obs.total_questions}"
    )
    return question, [], status, ""


def gradio_step(task_id: str, response: str, history: List, current_q: str):
    if not response.strip():
        return current_q, history, "⚠️ Please type a response before submitting.", ""

    env = _get_or_create_env(task_id)
    action = InterviewAction(response=response)
    result = env.step(action)

    history = history + [[current_q, response]]

    if result.done:
        summary = result.info.get("episode_summary", {})
        final   = summary.get("final_score", 0.0)
        passed  = "✅ PASSED" if summary.get("passed") else "❌ Did not pass"
        status  = (
            f"**Interview Complete!** {passed}\n\n"
            f"**Final Score:** {final:.0%}\n\n"
            f"**Feedback:** {result.info.get('feedback', '')}"
        )
        return result.observation.current_question, history, status, ""
    else:
        feedback = result.info.get("feedback", "")
        score    = result.info.get("grader_breakdown", {}).get("total_score", 0.0)
        status   = (
            f"**Question {result.info['question_number'] - 1} Score:** {score:.0%}\n\n"
            f"**Feedback:** {feedback}\n\n"
            f"**Progress:** {result.info['question_number'] - 1}/{result.info['total_questions']} answered"
        )
        return result.observation.current_question, history, status, ""


def build_gradio_ui():
    task_choices = list(TASK_REGISTRY.keys())

    with gr.Blocks(
        title="🎤 AI Interview Simulator",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎤 AI Interview Simulator — OpenEnv
            Practice real job interviews with structured AI feedback.
            Select a difficulty, start the interview, and answer each question.
            """
        )

        with gr.Row():
            task_dd = gr.Dropdown(
                choices=task_choices,
                value=task_choices[0],
                label="Interview Task",
            )
            reset_btn = gr.Button("🔄 Start / Restart Interview", variant="primary")

        status_md = gr.Markdown("Click **Start** to begin the interview.")

        question_box = gr.Textbox(
            label="📋 Current Question",
            interactive=False,
            lines=3,
        )

        response_box = gr.Textbox(
            label="✍️ Your Answer",
            placeholder="Type your answer here…",
            lines=5,
        )

        submit_btn = gr.Button("📤 Submit Answer", variant="secondary")

        chat_hist = gr.Chatbot(label="Interview Transcript", height=400)

        # Wiring
        reset_btn.click(
            fn=gradio_reset,
            inputs=[task_dd],
            outputs=[question_box, chat_hist, status_md, response_box],
        )

        submit_btn.click(
            fn=gradio_step,
            inputs=[task_dd, response_box, chat_hist, question_box],
            outputs=[question_box, chat_hist, status_md, response_box],
        )

    return demo


# ──────────────────────────────────────────────
# Mount Gradio on FastAPI
# ──────────────────────────────────────────────

demo = build_gradio_ui()
app  = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
