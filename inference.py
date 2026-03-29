import argparse
import os
import sys
import time
from typing import List, Tuple

from openai import OpenAI

from interview_env import InterviewEnv, InterviewAction, TASK_REGISTRY
from interview_env.models import InterviewObservation


# =========================
# CONFIG
# =========================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

TEMPERATURE  = 0.3
MAX_TOKENS   = 500
MAX_RETRIES  = 2


# =========================
# STRONG SYSTEM PROMPT 🔥
# =========================

SYSTEM_PROMPT = """
You are a top-tier senior software engineer in a real interview.

STRICT RULES:

1. STRUCTURE:
   - Behavioral → STAR (Situation, Task, Action, Result)
   - Technical → Step-by-step explanation
   - System Design → Follow EXACT format below

2. SYSTEM DESIGN FORMAT (MANDATORY):
   - Requirements (functional + non-functional)
   - High-level architecture
   - Components (API, DB, cache, queue)
   - Data flow
   - Scaling strategy
   - Failure handling (retries, monitoring)
   - Trade-offs (consistency vs availability)

3. ALWAYS INCLUDE:
   - Real tools (Redis, Kafka, AWS, Kubernetes)
   - Metrics (latency, scale, throughput)
   - Trade-offs

4. BE:
   - Clear
   - Structured
   - Professional
   - Specific

Do NOT repeat the question.
"""


# =========================
# PROMPT BUILDER
# =========================

def build_user_prompt(obs: InterviewObservation) -> str:
    return f"""
Role: {obs.job_title}
Company: {obs.company_context}
Stage: {obs.stage.value}

Question:
{obs.current_question}

Answer clearly with proper structure.
"""


# =========================
# LLM CALL
# =========================

def call_llm(client: OpenAI, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"⚠ Retry {attempt+1}: {e}")
            time.sleep(2)

    return "I would approach this problem systematically using best practices."


# =========================
# SELF-IMPROVEMENT LOOP 🔥
# =========================

def improve_answer(client: OpenAI, question: str, answer: str) -> str:
    improve_prompt = f"""
Improve this interview answer.

Question:
{question}

Answer:
{answer}

Make it:
- More structured
- Add tools (Redis, Kafka, etc.)
- Add metrics
- Add trade-offs
- More professional

Return only improved answer.
"""
    return call_llm(client, improve_prompt)


# =========================
# RUN TASK
# =========================

def run_task(client: OpenAI, task_id: str) -> Tuple[float, List[float]]:

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    env = InterviewEnv(task_id=task_id)
    obs = env.reset()

    scores = []
    step = 0

    while not obs.interview_complete:
        step += 1

        print(f"\nQ{step}: {obs.current_question[:80]}...")

        # Step 1: Generate answer
        answer = call_llm(client, build_user_prompt(obs))

        # Step 2: Improve answer
        improved = improve_answer(client, obs.current_question, answer)

        # Step 3: Send to env
        action = InterviewAction(response=improved)
        result = env.step(action)

        score = result.info.get("grader_breakdown", {}).get("total_score", 0)
        scores.append(score)

        print(f"Score: {score:.2f}")

        obs = result.observation

        if result.done:
            final = result.info.get("episode_summary", {}).get("final_score", 0)
            passed = "PASSED ✅" if final >= 0.60 else "FAILED ❌"

            print(f"\n── Episode complete: {passed} ──")
            print(f"Final Score: {final:.4f}")

            return final, scores

    return 0.0, scores


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = list(TASK_REGISTRY.keys()) if args.task == "all" else [args.task]

    results = {}

    for task_id in tasks:
        final, scores = run_task(client, task_id)
        results[task_id] = final

    # =========================
    # SUMMARY
    # =========================

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for t, score in results.items():
        status = "PASS" if score >= 0.60 else "FAIL"
        print(f"{t:<30} {score:.4f}  {status}")

    avg = sum(results.values()) / len(results)
    print(f"\nAverage Score: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
