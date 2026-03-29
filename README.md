---

title: AI Interview Simulator
emoji: "🎤"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - interview
  - rl-environment
  - nlp
  - hiring
app_port: 7860

---

# 🎤 AI Interview Simulator — OpenEnv Environment

> Train and evaluate AI agents that perform in real-world job interviews.

---

## 🌍 Why This Environment?

Job interviews require:

* Technical knowledge
* Structured communication (STAR method)
* Depth and clarity
* Professional tone

This environment simulates real interview conditions and evaluates AI agents across these dimensions.

---

## 🚀 What Makes This Unique?

* Simulates a **complete interview flow** (not single prompts)
* Provides **step-wise reward signals**
* Includes **behavioral + technical + system design**
* Uses **deterministic grading (no randomness)**
* Models **real-world ambiguity and decision-making**

---

## 🧠 Evaluation Philosophy

This environment evaluates AI like a real interviewer would:

* Not just correctness → **communication quality**
* Not just answers → **reasoning structure**
* Not just knowledge → **decision-making**

Agents must demonstrate:

* Clarity
* Structure
* Depth
* Professionalism

---

## 🗂️ Project Structure

```text
ai-interview-simulator/
│
├── interview_env/
│   ├── env.py
│   ├── models.py
│   ├── tasks.py
│   └── graders.py
│
├── inference.py
├── validate.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
├── README.md
├── results.txt
```

---

## 🔧 Setup

```bash
git clone https://github.com/keerthana-nagireddy/ai-interview-simulator.git
cd ai-interview-simulator
pip install -r requirements.txt
```

---

## ▶️ Run Inference

```bash
python inference.py
```

---

## 🧪 Validate Environment

```bash
python validate.py
```

---

## 📐 OpenEnv Interface

| Method  | Description     |
| ------- | --------------- |
| reset() | Start interview |
| step()  | Submit answer   |
| state() | Get full state  |

---

## 📊 Tasks

### EASY — Junior Behavioral

* Introduction
* Debugging
* Projects
* Code review

### MEDIUM — Technical

* REST vs GraphQL
* Event loop
* SQL optimization
* React performance

### HARD — System Design

* Distributed systems
* Order processing
* Scaling
* Incident handling

---

## 🎯 Reward Function

Score range: **0.0 → 1.0**

### Based on:

* Relevance (35%)
* Structure (25%)
* Depth (25%)
* Professionalism (15%)

### Additional signals:

* Time penalty
* Repetition penalty
* Improvement bonus
* Difficulty scaling

---

## 📈 Baseline Results

Baseline results are available in `results.txt`

| Task        | Score      | Status |
| ----------- | ---------- | ------ |
| Junior      | 0.9192     | PASS   |
| Mid         | 0.7622     | PASS   |
| Senior      | 0.6592     | PASS   |
| **Average** | **0.7802** | ✅      |

---

## 🏆 Key Features

* OpenEnv compliant (65/65 checks passed)
* Deterministic evaluation
* Multi-step reward shaping
* Real-world interview simulation
* Supports RL and benchmarking

---

## 🏁 Summary

AI Interview Simulator transforms interviews into a structured, trainable AI environment.

It bridges:

* Static NLP benchmarks ❌
* Real-world evaluation ✅

---

## 📄 License

MIT License
