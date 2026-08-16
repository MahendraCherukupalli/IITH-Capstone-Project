# CareFlow 🏥 — Enterprise Multi-Agent Care Coordination Assistant

CareFlow is a stateful, multi-agent AI assistant designed to automate complex clinical care coordination tasks, including insurance copay lookups, prior authorization requirement searches, specialist referral status tracking, and clinical safety triage with Human-in-the-Loop (HITL) approval checkpoints.

Built on an enterprise-grade architecture, CareFlow combines hybrid vector search (Qdrant + BM25 + CrossEncoder reranking), Model Context Protocol (MCP) Electronic Health Record (EHR) data servers, safety guardrails, OpenTelemetry tracing via Langfuse Cloud, and an interactive real-time dashboard.

---

## 🌟 Core Capabilities

- **💳 Specialist Copay & Plan Verification**: Instant queries against patient insurance plan rules (e.g., `MERIDIAN-GOLD`).
- **📋 Prior Authorization Matrix Search**: Automated retrieval over policy documents to verify preauth requirements for procedures (e.g., MRI of brain).
- **🩺 Active Referral Tracking**: Real-time status checks on specialist referrals (e.g., Cardiology, Dr. Aris Thorne).
- **🛡️ Clinical Safety & Guardrails**: NeMo-inspired input/output validation, prompt injection defense, and PII protection.
- **⏸️ Human-in-the-Loop (HITL) Checkpoints**: Automatic workflow pauses for emergent symptoms (e.g., acute chest pain) requiring Care Coordinator manual approval.
- **📊 Real-Time Observability & Telemetry**: Live OpenTelemetry trace spans, multi-model cost aggregations, P95 observation latency charts, and Qdrant corpus insights.

---

## 🛠️ Core Frameworks & Technologies

| Framework / Technology | Layer / Purpose | Key Role in CareFlow |
| :--- | :--- | :--- |
| **LangGraph** | Multi-Agent Orchestration | Manages stateful agent nodes, conditional execution edges, and state interruption checkpoints (`interrupt()`). |
| **Langfuse** | LLM Engineering & Telemetry | Collects OpenTelemetry trace spans, computes daily model/token costs, and calculates P95 observation latency metrics. |
| **Qdrant** | Vector Search Engine | Indexes clinical policy PDF chunks using dense embeddings (`all-MiniLM-L6-v2`) with cosine similarity. |
| **LiteLLM** | Unified LLM Proxy | Seamlessly bridges Google Gemini 2.5 Flash, OpenAI GPT-4o, and local Ollama models with fallback resilience. |
| **Model Context Protocol (MCP)** | Electronic Health Record (EHR) Data Standard | Exposes standardized Electronic Health Record (EHR) tools (`get_patient_profile`, `get_coverage_details`, `get_referral_status`). |
| **FastAPI & Uvicorn** | Backend Server | Asynchronous Python REST API serving chat workflows, patient profile switching, and live telemetry feeds. |
| **Rank-BM25 & CrossEncoder** | Hybrid Search & Reranking | Combines sparse BM25Okapi search with `ms-marco-MiniLM-L-6-v2` re-ranking over 14 clinical policy PDFs. |
| **DeepEval** | Evaluation & Benchmarking | Scores Faithfulness, Answer Relevancy, Hallucination, and Safety across golden evaluation test sets. |

---

## 🤖 Multi-Agent Architecture & Tool Specs

CareFlow uses a specialized team of autonomous agents operating on a shared state (`CareFlowState`) with end-to-end security guardrails, hybrid RAG retrieval, Model Context Protocol (MCP) servers, and Langfuse OpenTelemetry tracking.

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          🏥 CAREFLOW COMPLETE END-TO-END ARCHITECTURE (DATA ➔ UI)                              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                │
│  [ STEP 1: RAW DATA SOURCES ]                                                                                  │
│    📄 14 Clinical Policy PDF Documents                  💾 Patient EHR JSON Database                          │
│       (copay-schedule.pdf, preauth-matrix.pdf, etc.)           (Demographics, Vitals, Allergies, Referrals)      │
│                         │                                                    │                                 │
│                         ▼                                                    ▼                                 │
│  [ STEP 2: INGESTION, INDEXING & MCP PROTOCOL ]                                                                │
│    🔪 PDF Chunker & Markdown Normalizer                🔌 Model Context Protocol (MCP) Server                  │
│    🧠 Qdrant Dense Vector Indexer (MiniLM-L6)                 Exposes EHR Data Tools:                              │
│    🔤 Rank-BM25 Keyword Indexer                              - get_patient_profile()                              │
│    📦 Qdrant Collection: careflow_corpus                     - get_coverage_details()                             │
│                         │                                        - get_referral_status()                               │
│                         │                                                    │                                 │
│ ────────────────────────┼────────────────────────────────────────────────────┼──────────────────────────────── │
│                         │                                                    │                                 │
│  [ STEP 3: USER REQUEST INGRESS & SAFETY GUARDRAILS ]                                                          │
│    👤 Patient Query Input ──► 🔒 Input Guardrail Filter (Prompt Injection Check & Emergency Flag)             │
│                                              │                                                                 │
│                                              ▼                                                                 │
│  [ STEP 4: STATEFUL LANGGRAPH MULTI-AGENT WORKFLOW ENGINE ]                                                    │
│                             ┌──────────────────────────────────┐                                               │
│                             │ 🎯 Intake & Triage Agent Node    │                                               │
│                             └────────────────┬─────────────────┘                                               │
│                                              │                                                                 │
│            ┌─────────────────────────────────┼─────────────────────────────────┐                               │
│            │ Copay / Preauth Search          │ Referral Inquiries              │ Emergent Symptoms             │
│            ▼                                 ▼                                 ▼                               │
│  ┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐             │
│  │ 💳 Insurance Specialist   │     │ 🩺 Referral Specialist    │     │ ⏸️ Human-in-the-Loop      │             │
│  │    Agent Node             │     │    Agent Node             │     │    (HITL Checkpoint)      │             │
│  └─────────────┬─────────────┘     └─────────────┬─────────────┘     └─────────────┬─────────────┘             │
│                │                                 │                                 │ Coordinator               │
│                ├─────────────────────────────────┴─────────────────────────────────┤ Approved                  │
│                │ Queries RAG Policy & MCP EHR Tools                                │                           │
│                ▼                                                                   │                           │
│  ┌───────────────────────────┐                                                     │                           │
│  │ 🔍 Hybrid Policy RAG      │                                                     │                           │
│  │    Dense Vector + BM25    │                                                     │                           │
│  │    Reciprocal Rank Fusion │                                                     │                           │
│  │    CrossEncoder Reranker  │                                                     │                           │
│  └─────────────┬─────────────┘                                                     │                           │
│                │ Document Chunks + PDF Citations                                   │                           │
│                ▼                                                                   │                           │
│  ┌───────────────────────────┐                                                     │                           │
│  │ 🕵️ Safety Critic Node     │                                                     │                           │
│  │    (Medical Audit)        │                                                     │                           │
│  └─────────────┬─────────────┘                                                     │                           │
│                │ Safety Approved                                                   │                           │
│                └─────────────────────────────────┬─────────────────────────────────┘                           │
│                                                  │                                                             │
│                                                  ▼                                                             │
│  [ STEP 5: LLM PROXY & OPENTELEMETRY TELEMETRY LAYER ]                                                         │
│    🔌 LiteLLM Proxy Layer ──► Google Gemini 2.5 Flash / OpenAI GPT-4o / Local Ollama                          │
│    📡 Langfuse Cloud ───────► OpenTelemetry Trace Spans, Daily Model Costs, P95 Observation Latency Analytics │
│                                                  │                                                             │
│                                                  ▼                                                             │
│  [ STEP 6: FASTAPI REST SERVER & FRONTEND WEB WORKSPACE ]                                                       │
│    ⚡ FastAPI Backend Server (/api/chat, /api/patients/{ref}, /api/hitl/resume, /api/telemetry/langfuse/p95)   │
│                                                  │                                                             │
│      ┌───────────────────────────────────────────┴───────────────────────────────────────────┐                 │
│      ▼                                                                                       ▼                 │
│  🖥️ VIEW 1: CARE COORDINATOR WORKSPACE                                 📈 VIEW 2: OBSERVABILITY DASHBOARD       │
│    ├── 👤 Patient Chat Widget (Formatted Markdown & PDF Links)           ├── 📊 Total Count Traces Bar Chart   │
│    ├── 🩺 Patient EHR Hub (Live Patient Switching)                       ├── ⚙️ Multi-Model LLM Cost Chart     │
│    ├── 📜 Workflow Trace Stream Console                                 ├── 📈 P95 Latency Observation Chart  │
│    └── ⚠️ HITL Clinical Safety Review Banner                             ├── 🔍 Qdrant Corpus Insights Card    │
│                                                                          └── 📜 OpenTelemetry Log Stream       │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 👥 Agents Detail

1. **Intake & Triage Agent** (`src/agents/intake_agent.py`)
   - **Role**: Analyzes initial patient query intent, checks for clinical emergency flags, and routes execution to the appropriate domain specialist.
   - **Output**: Sets `target_agent` (`insurance_agent`, `referral_agent`, or `flag_for_human`).

2. **Insurance Specialist Agent** (`src/agents/insurance_agent.py`)
   - **Role**: Handles copay inquiries, coverage rules, annual deductible progress, and prior authorization requirements.
   - **Tools Used**: `retrieve_policy_documents`, `get_coverage_details`, `get_patient_profile`.

3. **Referral Specialist Agent** (`src/agents/referral_agent.py`)
   - **Role**: Tracks active specialist referral requests, in-network physician assignments, and approval expiration dates.
   - **Tools Used**: `get_referral_status`, `get_patient_profile`.

4. **Safety Critic Agent** (`src/agents/safety_reviewer.py`)
   - **Role**: Conducts a final audit on draft agent responses before patient delivery. Checks for medical accuracy, hallucination prevention, emergent symptom safety, and PII leakage.

5. **Human-in-the-Loop (HITL) Checkpoint** (`src/tools/flag_human.py`)
   - **Role**: Pauses graph execution when emergent symptoms (e.g., chest pain, severe dyspnea) are detected. Requires a Care Coordinator to manually approve or escalate in the workspace.

---

### 🔧 Tools & Capabilities Detail

| Tool Function | File Path | Description |
| :--- | :--- | :--- |
| `get_patient_profile(patient_id)` | `src/tools/ehr_tools.py` | Retrieves patient demographics, active plan, clinical vitals, known allergies, and active diagnoses. |
| `get_coverage_details(plan_name)` | `src/tools/ehr_tools.py` | Fetches specialist copay rates, annual deductible progress, and prior authorization rules. |
| `get_referral_status(patient_id)` | `src/tools/ehr_tools.py` | Returns active referral status, target specialty clinic, attending physician, and approval flags. |
| `retrieve_policy_documents(query)` | `src/rag/retriever.py` | Executes dense Qdrant vector search + sparse BM25 keyword search with Reciprocal Rank Fusion & CrossEncoder reranking over 14 clinical policy PDFs. |
| `flag_for_human_review(reason)` | `src/tools/flag_human.py` | Triggers a state interruption (`interrupt()`) in LangGraph for clinical coordinator intervention. |

---

## 📁 Directory Structure

```
careflow/
├── data/
│   ├── corpus/
│   │   ├── pdf/              # 14 Clinical Policy PDF Documents
│   │   └── markdown/         # Extracted corpus markdown chunks
│   └── ehr_database.json     # Mock EHR Patient Records (Sarah Jenkins, etc.)
├── scripts/
│   ├── evaluate_golden_set.py# DeepEval Golden Set Evaluation
│   ├── evaluate_m6_team.py   # Multi-Agent Workflow Benchmark
│   ├── evaluate_m7_resilience.py # Resilience & Fallback Test Suite
│   └── evaluate_m8_e2e.py    # End-to-End System Evaluation
├── src/
│   ├── agents/               # Autonomous Agent Implementations
│   │   ├── insurance_agent.py
│   │   ├── intake_agent.py
│   │   ├── referral_agent.py
│   │   └── safety_reviewer.py
│   ├── api/                  # FastAPI Application & Frontend Assets
│   │   ├── main.py           # REST Endpoints & Static Server
│   │   └── static/           # HTML, CSS, JavaScript Dashboard
│   ├── guardrails/           # Input/Output Guardrail Validators
│   ├── mcp/                  # MCP EHR Server Implementation
│   ├── rag/                  # Qdrant Vector Store, BM25, & Reranker
│   ├── tools/                # Agent Tools (EHR, HITL)
│   ├── utils/                # Observability & Resilience Utilities
│   └── workflow/             # LangGraph State Graph & Nodes
├── requirements.txt          # Python Package Dependencies
└── README.md                 # System Documentation
```

---

## 🚀 Prerequisites & Installation

### 1. Requirements
- **Python**: `3.10` or higher
- **Qdrant**: Local instance (`http://localhost:6333`) or Qdrant Cloud cluster
- **Langfuse Account**: Cloud (`https://us.cloud.langfuse.com`) or self-hosted keypair

### 2. Installation Setup
```bash
# Clone repository & navigate to directory
cd careflow

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration (`.env`)
Copy the provided `.env.example` template to create your local `.env` file:
```bash
cp .env.example .env
```

Fill in your active credentials in `.env`:
```env
# LLM Providers
GEMINI_API_KEY=your_google_gemini_key
OPENAI_API_KEY=your_openai_key

# Qdrant Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION_CORPUS=careflow_corpus

# Langfuse Observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

---

## 💻 Running the Application

### 1. Launch FastAPI Backend & Workspace Web Dashboard
```bash
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the Application
Open your web browser and navigate to:
- **Care Coordinator Workspace & Observability Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Running Benchmarks & Evaluation Suites

CareFlow includes automated DeepEval benchmark scripts to evaluate faithfulness, relevancy, hallucination, and system resilience:

```bash
# Run End-to-End DeepEval Evaluation Benchmark (M8)
python3 scripts/evaluate_m8_e2e.py

# Run Multi-Agent Team Benchmark (M6)
python3 scripts/evaluate_m6_team.py

# Run Resilience & Circuit Breaker Evaluation (M7)
python3 scripts/evaluate_m7_resilience.py

# Run Golden Set Evaluation
python3 scripts/evaluate_golden_set.py
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `POST /chat` | `POST` | Executes care coordination query through the multi-agent graph. |
| `GET /api/patient/{patient_ref}` | `GET` | Fetches active patient profile, vitals, coverage, and referral details. |
| `GET /api/patients/search?q={query}` | `GET` | Autocomplete patient search across EHR records. |
| `POST /resume` | `POST` | Resumes a paused workflow state following coordinator approval or escalation. |
| `GET /api/telemetry/langfuse/p95` | `GET` | Fetches real-time OpenTelemetry trace metrics, LLM generation costs, P95 observation latencies, and Qdrant stats. |

---

