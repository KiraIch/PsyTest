# PsyTest

[繁體中文](README.md) | **English**

PsyTest is a local multimodal counseling-simulation data generation system built on a "fast-thinking, slow-thinking" architecture. It automatically produces ecologically valid counseling simulation data for training and research. All model inference can run locally, minimizing the risk of sensitive data leaving the machine. This document is the project's official specification and is maintained on a regular update cycle.

- License: Apache License 2.0, see the `LICENSE` file.
- Maintainer: KiraIch
- Document version: v1.1.2
- Last updated: 2026-10-16
- Project status: All phases complete (Phase 1 through Phase 7), now in maintenance. See `HANDOVER.md` for details.

---

## Contents

1. [Overview and Architecture](#1-overview-and-architecture)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Model Download and Placement](#4-model-download-and-placement)
5. [Ollama Setup](#5-ollama-setup)
6. [Quick Start](#6-quick-start)
7. [Configuration Files](#7-configuration-files)
8. [Output Data Format](#8-output-data-format)
9. [Troubleshooting](#9-troubleshooting)
10. [Ethics Statement and Terms of Use](#10-ethics-statement-and-terms-of-use)
11. [Development Guide](#11-development-guide)
12. [Testing](#12-testing)
13. [Version History](#13-version-history)
14. [License and Contact](#14-license-and-contact)

---

## 1. Overview and Architecture

### 1.1 Overview

This system generates simulated counseling data for clinical psychology education and research. It does not provide clinical diagnosis. The system consists of five layers: the perception layer handles multimodal annotation, the fast-thinking layer controls dialogue rhythm and emotional triggers, the slow-thinking layer role-plays virtual clients, the voice layer synthesizes emotional speech, and the data pipeline integrates everything into a structured JSONL database.

### 1.2 Architecture Diagram

```text
                    ┌─────────────────────────────────┐
                    │        數據生成管線 (Pipeline)    │
                    │  session / batch / collector    │
                    │        output_formatter         │
                    └───────┬─────────┬───────┬───────┘
                            │         │       │
              ┌─────────────┘         │       └─────────────┐
              ▼                       ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │   五感層          │  │   快思層          │  │   慢想層          │
    │  Perception      │  │  Fast Think      │  │  Slow Think      │
    │                  │  │                  │  │                  │
    │ EmbeddingGemma 2 │  │ Laya 421M        │  │ Ollama Gemma 4   │
    │ audio / visual   │  │ rhythm / emotion │  │ persona/director │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             └─────────────┬───────┴──────────┬──────────┘
                           ▼                  ▼
                 ┌──────────────────┐ ┌──────────────────┐
                 │   聲音層 (Voice)  │ │  督導報告        │
                 │ TTS / director   │ │  Supervisor      │
                 └──────────────────┘ └──────────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  硬體後端 (Backend) │
                 │ Metal/CUDA/ROCm/CPU │
                 └──────────────────┘
```

### 1.3 Module Map

| Layer | Directory | Key Model | Description |
|---|---|---|---|
| Perception | `src/perception/` | Ollama embeddinggemma-2 (license-free default) | Text embeddings and audio/visual feature analysis; ST and GGUF as fallbacks |
| Fast thinking | `src/fast_think/` | Official laya package (multilingual) | Dialogue rhythm control, emotion routing, crisis assessment; GGUF as fallback only |
| Slow thinking | `src/slow_think/` | Gemma 4 Uncensored (Ollama) | Virtual client role-play, persona management, director notes |
| Voice | `src/voice/` | Kokoro (sherpa-onnx), ChatTTS, pyttsx3 | Emotional speech synthesis; Orpheus disabled by default |
| Data pipeline | `src/pipeline/` | None (integration layer) | Single-session generation, batch scheduling, JSONL output |
| Backend | `src/backend/` | None | Automatic Metal, CUDA, ROCm, and CPU detection |
| Supervision report | `src/report/` | None (deterministic rules) | AI supervision analysis and Markdown reports |

---

## 2. System Requirements

### 2.1 Minimum Requirements

- OS: macOS 13 or later (Apple Silicon), Ubuntu 22.04 or later, Windows 10 or later (including WSL2).
- Python: 3.10 or later (verified on 3.13).
- Git: 2.40 or later.
- Disk: at least 15 GB free (Ollama model approx. 6.3 GB, local GGUF approx. 1.2 GB, Kokoro approx. 350 MB, plus output and temp space).
- Memory: minimum 8 GB (CPU-only mode); 16 GB or more recommended (when loading multiple models).

### 2.2 Recommended Specs

| Use Case | Recommended Setup |
|---|---|
| MacBook Air development | Apple Silicon, 16 GB+ memory, see `configs/macbook_air.yaml` |
| NVIDIA workstation | CUDA GPU, 24 GB+ memory, see `configs/cuda_workstation.yaml` |
| GPU-less environment | Multi-core CPU, see `configs/cpu_only.yaml` |

### 2.3 External Services

- Ollama: must be installed and running; used for slow-thinking-layer LLM inference. See Chapter 5.

---

## 3. Installation

### 3.1 General Steps

Use a dedicated conda environment to isolate dependencies:

```bash
git clone https://github.com/KiraIch/PsyTest.git
cd PsyTest
conda env create -f environment.yml
conda activate psytest
pip install -r requirements.txt
python -m src.backend.detector
python scripts/validate_setup.py
```

Or with shortcuts:

```bash
make setup
make check
```

### 3.2 Per-Platform llama-cpp-python Installation

`llama-cpp-python` must be rebuilt with the matching backend flag, otherwise GPU acceleration will not be enabled.

```bash
# macOS (Metal enabled automatically)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# NVIDIA CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# AMD ROCm
CMAKE_ARGS="-DGGML_HIP=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# CPU only (OpenBLAS acceleration)
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3.3 Windows

```bat
scripts\setup.bat
```

---

## 4. Model Download and Placement

Model weight files are not version-controlled (excluded via `.gitignore`); obtain them separately and place them under the `models/` directory.

### 4.1 Model Files

Place the following two weight files in `models/` (filenames must match exactly):

| File Name | Purpose | Size (approx.) | Status |
|---|---|---|---|
| `embeddinggemma-2-Q8_0.gguf` | Perception-layer embedding model | 296 MB | Not yet supported by llama.cpp; currently served via the ST backend (requires separate HF license acceptance) |
| `Laya-BF16.gguf` | Fast-thinking decision model | 805 MB | Fallback only; decisions default to the official laya package |

### 4.2 Placement and Verification

```bash
ls -lh models/
shasum -a 256 models/Laya-BF16.gguf
shasum -a 256 models/embeddinggemma-2-Q8_0.gguf
python -m src.backend.detector
```

See `models/README.md` for details. Do not upload weight files to public repositories or public links.

---

## 5. Ollama Setup

### 5.1 Installing and Starting

- macOS: install the Ollama app and open it; the address is `http://localhost:11434`.
- Linux: start the service with `ollama serve`.
- Windows: start Ollama and verify `http://localhost:11434` is reachable.
- The address can be overridden with the `OLLAMA_HOST` environment variable.

### 5.2 Pulling the Model

```bash
ollama pull fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive
ollama list
```

### 5.3 Model Name Detection

The system automatically scans installed models with fuzzy matching over the keywords `gemma`, `uncensored`, `hauhau`, and `aggressive`, supporting both e2b and e4b variants. Priority order: this project's pinned tag, explicit e4b tags, then e2b tags, followed by other keyword-matching models. If nothing matches, it falls back to any other available 4B+ model.

This project pins the e4b variant (`fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:latest`, 6.3 GB); all four `configs/*.yaml` files specify it explicitly. You may switch a config back to `auto` for automatic detection, or override with the `OLLAMA_MODEL` environment variable.

---

## 6. Quick Start

### 6.1 Environment Check

```bash
python -m src.backend.detector
python scripts/validate_setup.py
```

### 6.2 Generating Your First Session

```bash
python scripts/generate_session.py \
  --persona data/personas/mdd_severe.yaml \
  --scenario data/scenarios/initial_session.yaml \
  --output data/output/ \
  --save-audio --verbose
```

### 6.3 Batch Generation

```bash
python scripts/batch_generate.py \
  --config configs/default.yaml \
  --personas mdd_severe,bpd_crisis,ptsd_combat \
  --sessions-per-persona 10 \
  --output data/output/ \
  --resume
```

### 6.4 Interactive Testing

```bash
python scripts/interactive_session.py --persona data/personas/bpd_crisis.yaml
```

### 6.5 Common Shortcuts

```bash
make check     # hardware detection and environment validation
make test      # run tests
make generate  # single-session generation (default mdd_severe; override PERSONA and SCENARIO)
make batch     # batch generation
make report INPUT=<session-jsonl> OUTPUT=<report-md>  # supervision report
make clean     # clean outputs and temp files
```

Switching personas (three available: `mdd_severe`, `bpd_crisis`, `ptsd_combat`):

```bash
make generate PERSONA=data/personas/bpd_crisis.yaml
python scripts/generate_session.py --persona data/personas/ptsd_combat.yaml --scenario data/scenarios/initial_session.yaml --output data/output/
```

---

## 7. Configuration Files

Config files live in `configs/`, loaded by `src/config.py`, with environment variable overrides (see `.env.example`).

| File | Purpose |
|---|---|
| `configs/default.yaml` | Default settings with automatic backend detection |
| `configs/macbook_air.yaml` | MacBook Air tuning (80-degree temperature cap, smaller batches) |
| `configs/cuda_workstation.yaml` | NVIDIA workstation (maximum throughput) |
| `configs/cpu_only.yaml` | CPU only (full features, slower) |

Main sections: `project`, `backend` (`force` accepts null, metal, cuda, rocm, cpu), `models` (embedding, laya, llm, tts), `pipeline`, `thermal`, `logging`.

Environment variable overrides: `OLLAMA_HOST`, `OLLAMA_MODEL`, `PSYTEST_OUTPUT_DIR`, `PSYTEST_SAVE_AUDIO`, `PSYTEST_SAVE_EMBEDDINGS`, `PSYTEST_MAX_MEMORY_GB`, `PSYTEST_GPU_LAYERS`, `PSYTEST_TEMPERATURE_LIMIT`, `PSYTEST_LOG_LEVEL`. Copy `.env.example` to `.env` and fill it in (`.env` must never be committed).

Persona configs live in `data/personas/` (YAML with diagnosis, background, defense mechanisms, triggers, etc.); scenario configs live in `data/scenarios/`.

---

## 8. Output Data Format

The primary format is JSONL: the first line holds session metadata (ID, persona, summary), and each dialogue turn occupies one subsequent line, allowing streaming reads and line-by-line inspection. Each turn contains counselor text, client text with director notes, audio features, visual features, psychological annotations, Laya decisions, and three sets of 768-dimensional embedding vectors. See spec section 4.6 and `src/pipeline/output_formatter.py` for the full field definitions.

Output layout:

```text
data/output/
├── manifest.json
├── sessions/session_<12-char-id>.jsonl
├── audio/session_<12-char-id>_turn_<turn>.wav
└── annotations/session_<12-char-id>_annotations.json
```

`manifest.json` records the dataset name, session count, turn count, total audio hours, persona list, modalities, embedding dimensions, hardware, and model versions. Validation command: `python scripts/validate_output.py --input data/output/batch_001/ --check-embeddings --check-audio`.

---

## 9. Troubleshooting

### Q1: Why is `models/` empty on GitHub?

Weight files are too large and subject to distribution restrictions, so they are excluded from version control. Obtain them separately per Chapter 4.

### Q2: `git push` fails with file-too-large errors?

Check whether model or data files were accidentally staged: `git status --short`, `git check-ignore -v <path>`. Remove mistakes with `git rm --cached <path>` and check `.gitignore`.

### Q3: llama-cpp-python does not enable the GPU?

Reinstall with the matching `CMAKE_ARGS` from Section 3.2; do not use a generic prebuilt wheel.

### Q4: Cannot connect to Ollama?

Verify the service is running: `curl http://localhost:11434/api/tags`. On macOS open the Ollama app, on Linux run `ollama serve`, or set `OLLAMA_HOST`.

### Q5: Model not found?

Run `ollama list` to confirm the Gemma 4 variant was pulled. The system uses fuzzy matching; for very different names, specify the model explicitly with `OLLAMA_MODEL`.

### Q6: MacBook Air overheating?

Switch to `configs/macbook_air.yaml`; the system throttles above 80 degrees. Pause generation if temperatures stay above 85 degrees.

### Q7: Out of memory?

Do not load all models at once. The system loads on demand and generates serially in batch mode; switch to `configs/cpu_only.yaml` with smaller batches if needed.

### Q8: How do I choose and install a speech engine?

Selection order is Kokoro, ChatTTS, then pyttsx3; Orpheus is disabled by default. pyttsx3 ships with the main dependencies for basic speech. Kokoro additionally needs `sherpa-onnx` plus its model download (approx. 350 MB; commands in `models/README.md`). ChatTTS needs a separate install (AGPLv3+ licensed; first load downloads several GB from HF). Missing engines report clear errors instead of failing silently.

---

## 10. Ethics Statement and Terms of Use

1. All persona content produced by this system is fictional teaching material for clinical psychology education and research only. It does not constitute clinical diagnosis or treatment advice.
2. Simulated results must never be applied directly to real cases. Real-world use requires qualified professional judgment and compliance with applicable laws and ethical codes.
3. Crisis simulations may express self-harm or suicidal ideation and emotions, and may describe past history, but the system must never produce actionable instructions such as concrete self-harm methods or how to obtain tools.
4. Counselee and real-client data is sensitive personal data and is prohibited from version control. Development and testing may only use de-identified or synthetic data.
5. Use, modification, or distribution of this project must comply with the Apache License 2.0 with license and copyright notices retained. Model weights carry their own original licenses; verify their distribution and commercial terms before use.

---

## 11. Development Guide

Branching: `main` is the stable branch, `develop` the integration branch; feature branches are named `feature/<name>`, fixes `fix/<name>`, docs `docs/<name>`. Commit messages follow Conventional Commits (e.g., `feat(pipeline): implement session generator`). Merges into `main` or `develop` require at least one reviewer; changes involving scoring logic, personal data handling, or licensing require owner approval. All phases are complete; maintenance-phase workflow applies, see `HANDOVER.md`.

---

## 12. Testing

```bash
pytest tests/ -v
```

Test map: `test_backend.py` covers hardware detection; `test_embedding.py`, `test_laya.py`, `test_ollama.py`, `test_tts.py`, and `test_pipeline.py` cover each layer; `test_phase3.py` through `test_phase6.py` cover the intelligence layer, voice layer, pipeline, and auxiliary features. Cases missing weights, services, or architectures are skipped explicitly (currently only the GGUF architecture case). Before committing, run at least `git diff --check` and `git status --short`.

---

## 13. Version History

| Date | Version | Description | Author |
|---|---|---|---|
| 2026-10-16 | v0.1.0 | Initial release establishing document structure and maintenance rules | KiraIch |
| 2026-10-16 | v0.2.0 | Adopted the full development spec; completed Phase 1 (infrastructure) and all module skeletons for the fast/slow-thinking architecture | KiraIch |
| 2026-10-16 | v0.3.0 | Completed Phase 2 (embedding engine, decision engine, Ollama client); pinned the e4b variant | KiraIch |
| 2026-10-16 | v0.4.0 | Completed Phase 3 (emotion routing, rhythm control, persona management, director system) with aligned versions | KiraIch |
| 2026-10-16 | v0.5.0 | Completed Phase 4 (Kokoro, ChatTTS, and pyttsx3 engines with Orpheus guarded); replaced the cloud option with Kokoro | KiraIch |
| 2026-10-16 | v0.6.0 | Completed Phase 5 (single-session generation, output formatting, batch scheduling, four CLIs) with end-to-end verification | KiraIch |
| 2026-10-16 | v0.7.0 | Completed Phase 6 (thermal monitoring, memory management, deterministic supervision reports) | KiraIch |
| 2026-10-16 | v1.0.0 | Completed Phase 7: official laya package for decisions, ST fallback path for embeddings, finalized docs; entering maintenance | KiraIch |
| 2026-10-16 | v1.0.1 | Fixed pyttsx3 driver-reuse empty audio; added the silence guardrail | KiraIch |
| 2026-10-16 | v1.0.2 | JSONL restructured to a meta line plus one line per turn; hardened the decision-latency test | KiraIch |
| 2026-10-16 | v1.0.3 | Fixed bare-pytest path issue, synced pyproject dependencies, Makefile persona override | KiraIch |
| 2026-10-16 | v1.1.0 | Expanded to 9 personas (2 comorbid); added conversational style, resistance, and memory anchor fields | KiraIch |
| 2026-10-16 | v1.1.1 | Embeddings default to ST-first with automatic fallback; pipeline wires the embedding engine | KiraIch |
| 2026-10-16 | v1.1.2 | Embeddings via official Ollama embeddinggemma-2 (license-free, 768 dims verified) | KiraIch |

---

## 14. License and Contact

This project is licensed under the Apache License 2.0; see `LICENSE` for the full text and `NOTICE` for third-party attributions. Maintainer: KiraIch. Repository: https://github.com/KiraIch/PsyTest.git. Please file issues via GitHub Issues with reproduction steps, environment details, and sanitized logs.

---

*This document is maintained on a regular cycle, reviewed at least every three months per the project iteration schedule. Updates should adjust the document version, last-updated date, and the Chapter 13 history together.*
