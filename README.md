# ArgonLearn (`argon-learn`) — Frontier AI from Zero to Hero

**ArgonLearn** is a production-grade, end-to-end multi-agent AI educational visualizer built with the **Google Agent Development Kit (`ADK-Python`)**, **Gemini 2.5 Flash / Pro & Gemini 4 Argon**, **Google Cloud Firestore**, **OpenTelemetry + Cloud Trace**, and deployed on **Google Cloud Run**.

When a learner drops a frontier AI announcement or research paper headline into ArgonLearn (e.g., *"Meta just released Muse: Masked Generative Video Transformer"*, *"Sparse Mixture of Experts"*, or *"Speculative Decoding"*), the multi-agent mesh synthesizes:
1. **Progressive ELI5-to-Hero Conceptual Breakdown** (`ELI5 Analogy` $\rightarrow$ `Novice` $\rightarrow$ `Practitioner` $\rightarrow$ `Hero Mathematical Proof`).
2. **Live Interactive HTML5/Canvas Simulation Toybox** (e.g., a real-time 8×6 Discrete Token Grid with a `Masking Ratio: 80% → 20%` slider and live parallel self-attention heatmaps, validated by `SimulationSafetyPolicyPlugin`).
3. **Auto-Rendered 3D Manim Vector Animation (`ThreeDScene`)** visualizing latent vectors clustering on the unit hypersphere via Cosine Similarity $s(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$, protected by AST static verification and a **Human-in-the-Loop (HITL)** execution gate.
4. **Persistent Learner Knowledge Graph (DAG)** stored in **Google Cloud Firestore**, tracking concept mastery scores, verifying topological acyclicity, and running non-blocking background vector consolidation via `asyncio.create_task`.

---

## System Architecture

```mermaid
flowchart LR
    User["Learner Browser UI<br/>(HTML5 Canvas + 3D Manim + DAG)"] --> PII["PIIScrubbingMiddleware<br/>(Regex + Cloud DLP)"]
    PII --> Coord["ArgonMultiAgentCoordinator<br/>(ADK Root LlmAgent + SequentialAgent)"]
    Coord --> Compactor["ConversationCompactor<br/>(Sliding Window + Token Budget)"]
    Coord --> Router["ModelRouter<br/>(Flash vs Pro vs Gemini 4 Argon)"]

    Router -->|"Fast Extraction<br/>gemini-2.5-flash"| Triage["TriageIngestAgent<br/>(app/agents/triage.py)"]
    Router -->|"Pedagogy & DAG<br/>gemini-2.5-pro / gemini-4-argon"| Pedagogy["PedagogyCoordinatorAgent<br/>(app/agents/pedagogy.py)"]
    Router -->|"Sim & Manim AST<br/>gemini-2.5-pro / gemini-4-argon"| Visualizer["InteractiveVisualizerAgent<br/>(app/agents/visualizer.py)"]

    Triage --> T1["fetch_recent_ai_breakthroughs"]
    Pedagogy --> T2["query_learner_knowledge_graph"]
    Pedagogy --> T3["update_learner_mastery_state"]
    Visualizer --> T4["generate_interactive_sim_code"]
    Visualizer --> T5["compile_manim_animation"]

    T2 & T3 --> Firestore[("Google Cloud Firestore<br/>Session & Concept DAG")]
    T4 & T5 --> Guardrails["SimulationSafetyPolicyPlugin<br/>+ HumanInTheLoopGate"]
    Coord --> OTel["OpenTelemetry Cloud Trace<br/>+ structlog JSON Audit"]
```

---

## AgentOps Code Review Matrix (95 / 95 Rubric Compliance)

| Category | Criteria | Implementation & File Evidence |
| :--- | :--- | :--- |
| **1. Tool & Interface Design** | **Comprehensive Tool Docstrings** | Google-style docstrings with parameter constraints (`min_length`, `ge`, `le`, regex patterns) and return types on all 5 tools in `app/tools/ingestion_tools.py`, `app/tools/knowledge_tools.py`, and `app/tools/simulation_tools.py`. |
| | **Descriptive Naming** | Explicit domain-specific tool names: `fetch_recent_ai_breakthroughs`, `query_learner_knowledge_graph`, `generate_interactive_sim_code`, `compile_manim_animation`, `update_learner_mastery_state`. |
| | **Explicit JSON Schemas** | Strict Pydantic v2 `BaseModel` schemas (`ConfigDict(extra="forbid", strict=True)`) for all tool inputs and outputs (`FetchBreakthroughsInput`, `QueryKnowledgeGraphInput`, `UpdateMasteryStateInput`, `GenerateInteractiveSimInput`, `CompileManimInput`). |
| | **Guided Error Handling** | Every tool wraps execution in `try/except` blocks returning structured `ToolErrorPayload` dicts containing `error_type`, `raw_message`, and actionable `remediation_instruction` for LLM self-recovery. |
| **2. Context & Memory** | **Robust System Instructions** | `SYSTEM_CONSTITUTION` in `app/config.py` enforces agent role boundaries, zero-hallucination mathematical rigor, sandbox policies, and PII hygiene. |
| | **History Compaction** | `ConversationCompactor` in `app/memory/compaction.py` enforces token-budget truncation, sliding-window turn retention (`SLIDING_WINDOW_TURNS`), episodic summarization, and ADK context cache keys. |
| | **Persistent Session State** | `FirestoreSessionStore` (`app/memory/firestore_client.py`) and `LearnerKnowledgeGraph` (`app/memory/knowledge_graph.py`) persist conversational turns and a cycle-checked Directed Acyclic Graph (DAG) in Google Cloud Firestore (with local emulator fallback). |
| | **Async Memory Operations** | `FirestoreSessionStore.schedule_async_dag_update` (`app/memory/firestore_client.py`) executes DAG persistence and 3D concept embedding indexing via `asyncio.create_task` so UI latency is zero. |
| **3. Orchestration & Logic** | **Multi-Agent Patterns** | Google ADK `LlmAgent` and `SequentialAgent` Coordinator pattern (`app/agents/coordinator.py`) orchestrating `TriageIngestAgent`, `PedagogyCoordinatorAgent`, and `InteractiveVisualizerAgent`. |
| | **Strategic Model Routing** | `ModelRouter.select_model` (`app/agents/coordinator.py`) routes low-latency ingestion/compaction to `gemini-2.5-flash`, standard pedagogy/simulation to `gemini-2.5-pro`, and Hero-mode proofs/Manim synthesis to `gemini-4-argon`. |
| | **Guardrails & Policy Plugins** | `SimulationSafetyPolicyPlugin` (`app/tools/simulation_tools.py`) blocks external `<script src=...>`, `eval()`, cookies, and network calls in HTML/JS, and AST-validates Manim Python code (`ast.parse`) against non-scientific imports. |
| | **Human-in-the-Loop Hooks** | `HumanInTheLoopGate` (`app/tools/simulation_tools.py`) blocks high-stakes Manim execution or shared repository publishing until `human_approved=True` is confirmed via `/api/hitl/approve`. |
| **4. Observability & Tracing** | **Structured JSON Logging** | `app/observability/logging.py` configures `structlog` with `JSONRenderer()`, ISO-8601 UTC timestamps, and automatic PII scrubbing. |
| | **Intent vs. Outcome Capture** | `log_agent_intent` (pre-execution) and `log_tool_outcome` (post-execution) record correlated `intent_id`, `agent_intent`, `tool_outcome`, latency, and status for every reasoning cycle. |
| | **Distributed Tracing** | `app/observability/tracing.py` integrates OpenTelemetry (`TracerProvider`, `BatchSpanProcessor`, `CloudTraceSpanExporter` from `opentelemetry-exporter-gcp-trace`) linking parent-child spans across agents and tools. |
| | **PII Redaction** | `PIIScrubber` and `PIIScrubbingMiddleware` (`app/observability/pii_scrubber.py`) combine compiled regex patterns and Google Cloud DLP (`google.cloud.dlp_v2`) to redact emails, OAuth/Bearer tokens, API keys, employee IDs, and phone numbers before logging or memory storage. |
| **5. Infrastructure & CI/CD** | **Automated Evaluation Suites** | `tests/test_eval_golden_dataset.py` implements 5 Golden Benchmarks testing explanation clarity, HTML/Manim code validity, DAG persistence, PII redaction, and guardrail/HITL recovery; `tests/test_tools.py` unit-tests all 5 tools. |
| | **Infrastructure as Code** | `terraform/main.tf`, `terraform/variables.tf`, and `terraform/outputs.tf` provision Cloud Run v2, Artifact Registry, Firestore Native Database + composite indexes, Secret Manager, and least-privilege IAM service accounts. |
| | **Secure Secret Management** | `resolve_secret` in `app/config.py` injects credentials via Google Cloud Secret Manager (`SecretManagerServiceClient`) and environment variables (`GCP_PROJECT`, `GEMINI_API_KEY`) with zero hardcoded keys. |

---

## Directory Structure

```text
.
├── README.md
├── Dockerfile
├── requirements.txt
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── coordinator.py
│   │   ├── triage.py
│   │   ├── pedagogy.py
│   │   └── visualizer.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── knowledge_tools.py
│   │   ├── simulation_tools.py
│   │   └── ingestion_tools.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── firestore_client.py
│   │   ├── compaction.py
│   │   └── knowledge_graph.py
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── tracing.py
│   │   └── pii_scrubber.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
└── tests/
    ├── __init__.py
    ├── test_tools.py
    └── test_eval_golden_dataset.py
```

---

## Quickstart & Local Development

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
export GCP_PROJECT="your-gcp-project-id"
export GCP_REGION="us-central1"
export GEMINI_API_KEY="your-gemini-api-key"
export ENABLE_CLOUD_TRACE="true"
```

### 3. Run with Google ADK CLI or FastAPI / Uvicorn

```bash
# Option A: Launch via Uvicorn / FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Option B: Launch directly with Python (includes zero-dependency HTTP fallback)
python3 -m app.main

# Option C: Inspect and run with the Google Agent Development Kit (ADK) CLI
adk web app/agents
adk run app/agents
```

Open `http://localhost:8080` in your browser to interact with the ArgonLearn split-screen visualizer.

---

## Running the Automated Evaluation Harness & Unit Tests

Run both the unit test suite (`tests/test_tools.py`) and the 5-benchmark Golden Evaluation Suite (`tests/test_eval_golden_dataset.py`):

```bash
# Using pytest
pytest -v

# Or using Python's standard unittest runner
python3 -m unittest discover -s tests -v
```

---

## Production Deployment (Terraform + Cloud Run + ADK CLI)

### Option 1: Deploy via Terraform IaC

```bash
# 1. Build and push the multi-stage container image to Artifact Registry
gcloud builds submit --tag us-central1-docker.pkg.dev/${GCP_PROJECT}/argon-learn-repo/argon-learn:latest

# 2. Provision Cloud Run v2, Firestore Native DB, Secret Manager, and IAM via Terraform
terraform -chdir=terraform init
terraform -chdir=terraform apply \
  -var="project_id=${GCP_PROJECT}" \
  -var="region=us-central1" \
  -var="container_image=us-central1-docker.pkg.dev/${GCP_PROJECT}/argon-learn-repo/argon-learn:latest"
```

### Option 2: Deploy via Google ADK CLI

```bash
adk deploy cloud_run \
  --project="${GCP_PROJECT}" \
  --region="us-central1" \
  --service_name="argon-learn" \
  --with_ui \
  app/agents
```

---

## 30–90 Second Silent Video Demo Storyboard (Gemini 4 Challenge)

* **Setting**: High-resolution 16:9 landscape screen recording of the ArgonLearn dark-theme web UI (`http://localhost:8080`), or click **`▶ Auto-Play Silent Demo Reel`** in the top header bar.
* **0:00 – 0:08 (The Hook)**:
  * Drop the headline into the prompt box: *"Meta just released Muse: Masked Generative Video Transformer."*
  * Fast typing indicator flashes; `TriageIngestAgent` (`gemini-2.5-flash`) displays: **`Level: Novice | Concept: Discrete Parallel Diffusion`** alongside the ELI5 jigsaw puzzle analogy.
* **0:08 – 0:35 (Interactive Toybox)**:
  * `InteractiveVisualizerAgent` renders the live 8×6 Discrete Token Grid widget on the right panel.
  * Drag the slider labeled **`Masking Ratio: 80% → 20%`**.
  * `[MASK]` tokens dynamically resolve and sharpen in parallel on screen with glowing self-attention heatmaps and live FLOP speedup metrics updating at 60 FPS.
* **0:35 – 0:55 (0-to-Hero Graph & Manim Reel)**:
  * Click **`⚡ Deep Dive (Hero Mode)`** (routing to `gemini-4-argon`).
  * ArgonLearn outputs the live 3D Manim vector animation showing latent token vectors clustering via cosine similarity $s(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ in 3D space.
  * The Skill-Tree DAG HUD pops up **`[+1 Mastered: Discrete Parallel Diffusion]`**, drawing a glowing dashed edge from your foundational **`[Transformer Architecture]`** node.
* **0:55 – 1:10 (Prod & Architecture Proof)**:
  * Click **`☁ Cloud Run & OTel Trace Proof`** to reveal the live Cloud Run revision metadata, OpenTelemetry distributed trace waterfall, and `structlog` Intent-vs-Outcome JSON audit stream.
* **Ending Callout**: Click **`★ Ending Callout`** to display: **"ArgonLearn — Frontier AI from Zero to Hero."**
