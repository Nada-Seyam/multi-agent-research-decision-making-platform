# Decision Discussion Platform

A full-stack multi-agent discussion and analytics application. Users frame a
decision question — by uploading a structured objective file or entering one
manually — and specialized AI personas research, deliberate, and arrive at an
evidence-aware recommendation across multiple rounds. The system computes
real-time analytics (opinion trajectory, agreement, influence, sentiment, and
an interaction graph) as the discussion progresses and presents a structured
conclusion when it completes.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Services](#services)
  - [Discussion Producer API](#discussion-producer-api)
  - [Analytics Consumer API](#analytics-consumer-api)
  - [Week 2 RAG Service](#week-2-rag-service)
  - [Frontend (React SPA)](#frontend-react-spa)
- [Unified Launcher](#unified-launcher)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Start the Backend](#start-the-backend)
  - [Start the Frontend](#start-the-frontend)
- [Unified API Reference](#unified-api-reference)
  - [Health Check](#health-check)
  - [Create a Discussion (JSON)](#create-a-discussion-json)
  - [Create a Discussion (File Upload)](#create-a-discussion-file-upload)
  - [Get Discussion Snapshot](#get-discussion-snapshot)
  - [Get Discussion Analysis](#get-discussion-analysis)
  - [Analytics Consumer Endpoints](#analytics-consumer-endpoints)
- [Frontend Views](#frontend-views)
  - [Setup Page](#setup-page)
  - [Discussion View](#discussion-view)
  - [Analytics Dashboard](#analytics-dashboard)
  - [Conclusion View](#conclusion-view)
- [Discussion Engine](#discussion-engine)
  - [Persona Selection](#persona-selection)
  - [Agent Graph](#agent-graph)
  - [Round Execution](#round-execution)
  - [Decision Synthesis](#decision-synthesis)
  - [Conclusion Generation](#conclusion-generation)
- [Analytics Pipeline](#analytics-pipeline)
  - [Event-Driven Architecture](#event-driven-architecture)
  - [Round-Level Analytics](#round-level-analytics)
  - [Final Analytics](#final-analytics)
  - [Streaming Artifact Store](#streaming-artifact-store)
- [Data Flow](#data-flow)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The platform consists of four cooperating services:

```
                         ┌──────────────────────────┐
                         │    React SPA (Vite)       │
                         │    http://127.0.0.1:5173  │
                         └────────────┬─────────────┘
                                      │  Vite proxy /api
                                      ▼
                         ┌──────────────────────────┐
                         │  Discussion Producer API  │
                         │  http://127.0.0.1:8001    │
                         │                           │
                         │  POST /api/v1/discussions  │
                         │  GET  /api/v1/discussions  │
                         │  GET  .../analysis (proxy) │
                         │  GET  /health              │
                         └──┬───────────────┬────────┘
                            │               │
       lifecycle events     │               │  health probe + RAG queries
       (HTTP + idempotency) │               │
                            ▼               ▼
        ┌───────────────────────┐   ┌──────────────────┐
        │ Analytics Consumer API│   │ Week 2 RAG Svc   │
        │ http://127.0.0.1:8002 │   │ http://127.0.0.1 │
        │                       │   │ :8000             │
        │ POST .../events       │   │                   │
        │ GET  .../analysis     │   │ Evidence retrieval │
        │ GET  /health          │   │ Persona config     │
        └───────────────────────┘   └──────────────────┘
```

The frontend communicates exclusively with the producer API on port 8001.
The Vite dev server proxies `/api` requests to the producer, so no CORS
configuration is needed in the browser during local development. The producer
handles discussion creation, snapshot persistence, and proxies analytics
requests to the consumer — keeping the frontend on a single API origin.

---

## Services

### Discussion Producer API

**Port:** 8001 (configurable via `DISCUSSION_API_PORT`)

The producer is a FastAPI application located at
`services/discussion-engine/discussion/api.py`. It
is the primary backend the frontend talks to and provides:

- **Discussion creation** — accepts a JSON payload or a multipart file upload,
  validates the input, loads the objective through the Week 2 adapter,
  selects participants, builds the agent graph, and runs the multi-round
  discussion as a background task.
- **Snapshot polling** — persists a JSON snapshot to disk after every lifecycle
  event (queued, preparing, running, completed, failed). The frontend polls
  `GET /api/v1/discussions/{id}` at 1.5-second intervals to pick up new
  rounds as they complete.
- **Analysis proxy** — forwards `GET /api/v1/discussions/{id}/analysis` to
  the analytics consumer, so the frontend never needs to know the consumer's
  address.
- **Event delivery** — emits lifecycle events (`discussion.started`,
  `round.completed`, `discussion.completed`, `discussion.failed`) to the
  configured event endpoint via HTTP POST with an `Idempotency-Key` header.
  When the analytics consumer is co-located, events go to its
  `/api/v1/discussion-events` endpoint. A built-in mock consumer catches
  events when no external endpoint is configured.
- **RAG lifecycle management** — starts the Week 2 retrieval API
  automatically during FastAPI lifespan startup and shuts it down on exit.
- **CORS** — allows `http://127.0.0.1:5173` and `http://localhost:5173` by
  default (configurable via `DISCUSSION_CORS_ORIGINS`).

### Analytics Consumer API

**Port:** 8002 (configurable via `ANALYTICS_API_PORT`)

The consumer is a FastAPI application located at
`services/analytics/src/analytics/api.py`. It:

- **Ingests lifecycle events** — receives producer events at
  `POST /api/v1/discussion-events`, validates them idempotently (duplicate
  events with the same `Idempotency-Key` and identical content return 200;
  conflicting content returns 409), and triggers analytics computation.
- **Computes round-level analytics** — on every `round.completed` event, the
  consumer loads the embedded discussion snapshot, runs opinion change,
  agreement, and sentiment analysis for that round, and appends the result
  to the durable analytics artifact.
- **Computes final analytics** — on the `discussion.completed` event, the
  consumer runs the full analysis suite including influence (which requires
  the complete discussion history), and writes the final artifact.
- **Serves analytics** — exposes `GET /api/v1/discussions/{id}/analysis` to
  return the current analytics artifact (which grows incrementally as rounds
  complete).
- **Optional bearer-token authentication** — if `ANALYTICS_EVENT_TOKEN` is
  set, all incoming events must carry a matching `Authorization: Bearer`
  header.

### Week 2 RAG Service

**Port:** 8000

The Week 2 retrieval API is not a standalone service in this repository — it
is started automatically by the producer API through the `week2_adapter`
module, which calls into the Week 2 codebase at runtime. The adapter:

- Puts the Week 2 `src/` package on `sys.path`.
- Calls `start_local_rag()` during FastAPI startup to launch the RAG
  subprocess.
- Calls `stop_local_rag()` during shutdown to terminate it.
- Proxies persona loading, objective parsing, research planning, evidence
  retrieval, and LLM calls through thin wrapper functions.

The producer's health endpoint checks RAG readiness by probing
`GET http://127.0.0.1:8000/health` and reports `"degraded"` if the RAG
service is unreachable.

### Frontend (React SPA)

**Port:** 5173

A single-page React application built with Vite and TypeScript, located in
`frontend/`. The app uses vanilla CSS (injected as a JS template literal),
[Lucide React](https://lucide.dev/) icons, and custom SVG visualizations —
no charting library is needed. Three Google Fonts are loaded at runtime:
IBM Plex Sans, IBM Plex Mono, and DM Serif Display.

The frontend is entirely stateless and API-driven. It does not store data
locally. All discussion and analytics data comes from polling the producer
API.

---

## Unified Launcher

The `main.py` file at the repository root starts both backend services with
a single command:

```python
python main.py
```

The launcher:

1. Adds the consumer (`services/analytics/src`) and producer
   (`services/discussion-engine`) source
   directories to `sys.path`.
2. Configures the producer's `DISCUSSION_EVENT_ENDPOINT` environment variable
   to point at the co-located consumer
   (`http://127.0.0.1:{consumer_port}/api/v1/discussion-events`).
3. Starts the analytics consumer as a background daemon thread and waits up
   to 15 seconds for it to become ready.
4. Starts the discussion producer on the main thread (blocking).
5. On `Ctrl+C`, shuts down the producer first, then gracefully stops the
   consumer thread.

If `ANALYTICS_EVENT_TOKEN` is set, the launcher also copies it into
`DISCUSSION_EVENT_TOKEN` so the producer's event deliveries carry the correct
bearer token.

---

## Getting Started

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- An **API key for an OpenAI-compatible LLM endpoint** (`API_KEY`, `LLM_BASE_URL`,
  `LLM_MODEL`) configured in the Week 2 `.env` file (`services/discussion-engine/week2/src/.env`)
- The **Week 2 codebase** present at the path specified by
  `DISCUSSION_WEEK2_ROOT` (default: `services/discussion-engine/week2`)

### Environment Variables

The system reads configuration from environment variables and `.env` files
in each sub-project.

#### Discussion Producer (`DISCUSSION_` prefix)

| Variable | Default | Description |
|---|---|---|
| `DISCUSSION_API_HOST` | `127.0.0.1` | Host the producer binds to |
| `DISCUSSION_API_PORT` | `8001` | Port the producer listens on |
| `DISCUSSION_NUM_PARTICIPANTS` | `4` | Agents seated per discussion (3–6) |
| `DISCUSSION_PARTICIPANT_SELECTION_MODE` | `random` | Seeded `random` draw or an explicit `custom` roster |
| `DISCUSSION_SELECTED_PARTICIPANT_IDS` | `[]` | JSON list of 3–6 unique ids used in custom mode |
| `DISCUSSION_NUM_ROUNDS` | `4` | Default round count (3–6) |
| `DISCUSSION_RANDOM_SEEDS` | *(required unless using the legacy single seed)* | JSON list of candidate seeds; one is selected randomly for each discussion |
| `DISCUSSION_RANDOM_SEED` | *(none)* | Backward-compatible single seed used when no seed pool is configured |
| `DISCUSSION_WEEK2_ROOT` | `week2` | Path to the Week 2 project root |
| `DISCUSSION_OUTPUT_DIR` | `outputs/discussions` | Where snapshots are written |
| `DISCUSSION_LOGS_DIR` | `logs/discussions` | Where detailed logs are written |
| `DISCUSSION_MAX_INITIAL_QUERIES` | `4` | Max research queries during setup |
| `DISCUSSION_MAX_EVIDENCE_CHUNKS_PER_QUERY` | `4` | Evidence chunks per query |
| `DISCUSSION_MAX_EVIDENCE_CHARS_PER_CHUNK` | `6000` | Max characters per chunk |
| `DISCUSSION_MAX_TOTAL_EVIDENCE_CHUNKS` | `50` | Global evidence cap |
| `DISCUSSION_MAX_PERSONA_EVIDENCE_CHARS` | `24000` | Prompt-only dossier cap; full evidence remains in artifacts |
| `DISCUSSION_MAX_PARALLEL_AGENTS` | `4` | Concurrent agent model calls per round (`1` disables concurrency) |
| `DISCUSSION_MAX_CONCURRENT_DISCUSSIONS` | `2` | Discussion jobs allowed to run concurrently |
| `DISCUSSION_MAX_QUEUED_DISCUSSIONS` | `8` | Additional accepted jobs before the API returns `429` |
| `DISCUSSION_DYNAMIC_CHALLENGER_ROUTING` | `true` | Route messages to the strongest challenger |
| `DISCUSSION_UTILITY_CONSISTENCY_TOLERANCE` | `0.25` | Tolerance for utility score consistency |
| `DISCUSSION_PERSIST_ARTIFACTS` | `true` | Write snapshots and logs to disk |
| `DISCUSSION_EVENT_ENDPOINT` | *(auto-configured)* | URL for lifecycle event delivery |
| `DISCUSSION_EVENT_TOKEN` | *(none)* | Bearer token for event authentication |
| `DISCUSSION_EVENT_DELIVERY_REQUIRED` | `false` | Fail the discussion if event delivery fails |
| `DISCUSSION_EVENT_TIMEOUT_SECONDS` | `10` | HTTP timeout per delivery attempt |
| `DISCUSSION_EVENT_MAX_ATTEMPTS` | `5` | Max delivery retries |
| `DISCUSSION_EVENT_RETRY_INITIAL_SECONDS` | `1` | Initial retry backoff |
| `DISCUSSION_EVENT_RETRY_MAX_SECONDS` | `30` | Maximum retry backoff |
| `DISCUSSION_CORS_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Allowed CORS origins |

#### Analytics Consumer (`ANALYTICS_` prefix)

| Variable | Default | Description |
|---|---|---|
| `ANALYTICS_API_HOST` | `127.0.0.1` | Host the consumer binds to |
| `ANALYTICS_API_PORT` | `8002` | Port the consumer listens on |
| `ANALYTICS_OUTPUT_DIR` | `outputs/analytics` | Where analytics artifacts are written |
| `ANALYTICS_LOGS_DIR` | `logs/analytics` | Where analytics logs are written |
| `ANALYTICS_EVENT_TOKEN` | *(none)* | Bearer token required for incoming events |
| `ANALYTICS_NLI_DEVICE` | `auto` | NLI inference device (`auto`, `cpu`, or a Torch device) |
| `ANALYTICS_NLI_BATCH_SIZE` | `32` | Candidate-hypothesis pairs per model batch |
| `ANALYTICS_NLI_CACHE_SIZE` | `8192` | Maximum cached NLI pair scores |

#### Unified Launcher

| Variable | Default | Description |
|---|---|---|
| `ANALYTICS_CONSUMER_URL` | *(auto)* | Override the consumer event endpoint URL |
| `ANALYTICS_API_URL` | *(auto)* | Override the analytics query base URL |

### Start the Backend

From the repository root:

```powershell
# Install Python dependencies
pip install -r requirements.txt

# Start both APIs (producer + consumer + RAG)
python main.py
```

Once started, the following endpoints are available:

| Service | URL |
|---|---|
| RAG service | http://127.0.0.1:8000 |
| Producer Swagger docs | http://127.0.0.1:8001/docs |
| Consumer Swagger docs | http://127.0.0.1:8002/docs |

### Start the Frontend

In a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173** in the browser. The Vite dev server proxies
`/api` to the producer on port 8001 automatically.

### Docker Deployment

The multi-service Python backend (Producer, Consumer, and RAG services) is containerized to run as a single process tree inside one container. This architecture is designed for easy deployment to platforms like Hugging Face Spaces (Docker SDK).

**1. Build the Docker Image**

From the repository root, build the image using the provided
Dockerfile:

```powershell
docker build -t decision-platform-backend .
```

The build downloads immutable revisions of `BAAI/bge-m3`,
`cross-encoder/ms-marco-MiniLM-L-6-v2`, and the analytics model
`cross-encoder/nli-deberta-v3-base`, then saves the runtime artifacts under
`/opt/models`. This makes the image substantially larger and the first build
slower, but a newly created container does not download models during startup.
Docker can reuse this layer while the model build arguments remain unchanged.
Runtime Hugging Face access is disabled so an incomplete image fails clearly
instead of silently downloading a model.

**2. Run the Container**

Run the container, mapping the producer's port (8001) to your host and passing your environment variables (including your LLM `API_KEY`) via a `.env` file:

```powershell
docker run -p 8001:8001 --env-file .env decision-platform-backend
```

**Inside the container:**
The `main.py` launcher starts all three services automatically:
- The **Analytics Consumer API** runs as a background thread on port 8002.
- The **Discussion Producer API** runs on the main thread on port 8001.
- The **Week 2 RAG Service** is spawned as a subprocess on port 8000.

The container environment variables (`DISCUSSION_API_HOST` and `ANALYTICS_API_HOST`) are set to `0.0.0.0` during the build so that the producer is reachable from the outside. The RAG service binds to `127.0.0.1` by default, which is perfect for internal communication within the same container.

The producer does not wait for the RAG model and BM25 index to finish warming.
During that interval, `GET /health` reports `"status": "degraded"` and attempts
to start a discussion return HTTP 503 with `Retry-After: 10`. Once RAG is
ready, discussion creation works normally. `RAG_READY_TIMEOUT_SECONDS` only
controls synchronous Week 2 command-line runs; it does not terminate the web
producer.

The Docker image also builds the React frontend and serves it from `/` through
the producer on port 8001. The API remains available below `/api`, with health
and interactive API documentation at `/health` and `/docs` respectively.

---

## Unified API Reference

All endpoints below are served by the **producer** at `http://127.0.0.1:8001`
unless otherwise noted. The frontend only communicates with the producer.

### Health Check

```
GET /health
```

**Response (200):**

```json
{
  "status": "ok",
  "producer": "ready",
  "rag": {
    "ready": true,
    "url": "http://127.0.0.1:8000",
    "details": { "status": "ok" }
  },
  "mock_consumer": {
    "ready": true,
    "received_events": 0
  }
}
```

The `status` field is `"ok"` when the RAG service is healthy and `"degraded"`
when it is unreachable. The producer itself always starts regardless of RAG
availability.

### Create a Discussion (JSON)

```
POST /api/v1/discussions
Content-Type: application/json
```

**Request body:**

```json
{
  "topic": "Choosing a memory architecture",
  "objective": "Choose the primary architecture for the agent system.",
  "constraints": "Keep latency low\nPreserve auditability",
  "rounds": 4
}
```

| Field | Type | Constraints | Description |
|---|---|---|---|
| `topic` | string | 3–160 chars, required | The discussion subject |
| `objective` | string | 10–4000 chars, required | What the agents must decide |
| `constraints` | string | 0–4000 chars, optional | One constraint per line |
| `rounds` | integer | 3–6, default 4 | Number of discussion rounds |

**Response (202 Accepted):**

```json
{
  "discussion_id": "disc-a1b2c3d4",
  "status": "queued",
  "discussion_url": "/api/v1/discussions/disc-a1b2c3d4",
  "analysis_url": "/api/v1/discussions/disc-a1b2c3d4/analysis"
}
```

The discussion runs as a background task. Poll the `discussion_url` to track
progress.

### Create a Discussion (File Upload)

```
POST /api/v1/discussions/upload
Content-Type: multipart/form-data
```

| Field | Type | Constraints | Description |
|---|---|---|---|
| `objective_file` | file | `.txt` or `.md`, max 1 MB, UTF-8 | Structured objective document |
| `rounds` | integer | 3–6, default 4 | Number of discussion rounds |

The file should contain a structured objective with sections for the goal,
constraints, discussion topics, evidence requirements, and final goal.
This is the recommended input method and the primary option in the frontend.

**Response:** Same `202 Accepted` format as the JSON endpoint.

### Get Discussion Snapshot

```
GET /api/v1/discussions/{discussion_id}
```

**Response (200):**

```json
{
  "schema_version": "1.0",
  "discussion_id": "disc-a1b2c3d4",
  "status": "running",
  "progress": {
    "current_round": 2,
    "completed_rounds": 1,
    "total_rounds": 4
  },
  "started_at": "2026-09-22T16:00:00Z",
  "updated_at": "2026-09-22T16:05:32Z",
  "completed_at": null,
  "failure": null,
  "warnings": [],
  "objective": { "goal": "...", "final_goal": "..." },
  "participants": [
    "capability_maximizer",
    "practical_architect",
    "safety_governance_advocate",
    "cost_sustainability_analyst"
  ],
  "graph": {
    "agents": ["..."],
    "edges": [["agent_a", "agent_b"], "..."],
    "directed": true
  },
  "configuration": { "..." : "..." },
  "rounds": [
    {
      "round": 1,
      "opinions": [
        {
          "persona_id": "capability_maximizer",
          "round_number": 1,
          "opinion": "...",
          "position": "...",
          "confidence": 0.85,
          "changed_mind": false
        }
      ],
      "messages": [
        {
          "round_number": 1,
          "sequence": 0,
          "sender": "capability_maximizer",
          "recipients": ["practical_architect"],
          "content": "..."
        }
      ]
    }
  ],
  "opinions": {
    "capability_maximizer": {
      "rounds": ["..."]
    }
  },
  "decision_summary": null,
  "conclusion": null
}
```

The snapshot grows incrementally. Each completed round adds entries to
`rounds[]` and `opinions{}`. When the discussion completes, `decision_summary`
and `conclusion` are populated.

**Status progression:** `queued` → `preparing` → `running` → `completed` (or `failed`).

### Get Discussion Analysis

```
GET /api/v1/discussions/{discussion_id}/analysis
```

This endpoint is **proxied by the producer** to the analytics consumer at
`http://127.0.0.1:8002/api/v1/discussions/{id}/analysis`. The frontend never
needs the consumer's address.

**Response (200):**

```json
{
  "discussion_id": "disc-a1b2c3d4",
  "status": "completed",
  "round_analyses": [
    {
      "round": 1,
      "stances": {
        "capability_maximizer": {
          "agent_id": "capability_maximizer",
          "round": 1,
          "candidate": "Transformer-XL",
          "stance": 0.85,
          "change": 0.0,
          "candidate_changed": false
        }
      },
      "agreement": {
        "round": 1,
        "score": 0.42,
        "disagreement_score": 0.58,
        "num_agents_considered": 4
      },
      "sentiment": ["..."]
    }
  ],
  "final_analysis": {
    "opinion_change": {
      "capability_maximizer": ["...per-round stance records..."]
    },
    "agreement": ["...per-round agreement records..."],
    "influence": {
      "capability_maximizer": {
        "agent_id": "capability_maximizer",
        "score": 0.81,
        "num_data_points": 12,
        "reliability": "high",
        "num_recipients": 3
      }
    },
    "sentiment": ["...per-message sentiment records..."],
    "issues": []
  },
  "failure": null
}
```

The analytics artifact starts with `round_analyses[]` populated as each round
completes. `final_analysis` (which includes influence) is only available after
`discussion.completed`.

**Error responses:**

| Status | Meaning |
|---|---|
| 404 | No analytics artifact exists for this discussion |
| 503 | Analytics consumer is unreachable |

### Analytics Consumer Endpoints

These are exposed by the consumer at `http://127.0.0.1:8002` and are not
typically called by the frontend directly.

```
GET  /health
POST /api/v1/discussion-events
GET  /api/v1/discussions/{discussion_id}/analysis
GET  /api/v1/discussions/{discussion_id}/analysis/rounds/{round_number}
```

The `POST /api/v1/discussion-events` endpoint requires an
`Idempotency-Key` header matching the event's `event_id`. If
`ANALYTICS_EVENT_TOKEN` is set, an `Authorization: Bearer <token>` header is
also required.

---

## Frontend Views

### Setup Page

The landing page where the user frames the decision. It has two input
methods:

1. **File upload (recommended)** — upload a structured `.txt` or `.md`
   objective file. The file is parsed by the Week 2 adapter to extract the
   objective, constraints, and discussion topics.
2. **Manual entry** — fill in the topic (3–160 chars), objective
   (10–4000 chars), constraints (one per line), and select the number of
   rounds (3–6).

A sidebar displays the six available personas with their roles, focus areas,
and priority glyphs. A circular topology preview animates through each
persona with a self-cycling highlight. Four personas are randomly seated
for each discussion.

### Discussion View

After a discussion starts, the frontend polls the producer every 1.5 seconds.
The discussion view displays:

- **Agent graph** — participants are arranged as profile cards on a ring.
  Animated packets travel along directed edges as messages are replayed.
  The currently speaking agent is elevated; recipients pulse on arrival.
  A central hub shows the sender → recipients for the current message.
- **Message transcript sidebar** — a chronological chat panel grouped by
  round, with each message attributed to its sender (color-coded avatar)
  and showing recipient tags. Messages can be expanded to read the full
  content.
- **Status indicators** — a progress bar, status pill (queued / preparing /
  running / completed / failed), round counter, and participant count.

The graph layout is responsive with three card variants: `full` (desktop),
`lite` (tablet), and `mini` (phone). A `prefers-reduced-motion` media query
disables all animations.

### Analytics Dashboard

The analytics tab presents five categories of analysis in a responsive
grid layout:

1. **Agreement gauge** — a donut gauge showing the latest pairwise candidate
   alignment as a 0–100% score, color-coded green (≥70%), amber (40–70%),
   or red (<40%).
2. **Position changes** — a count of committed candidate switches across all
   agents, with pills showing the round and new candidate for each switch.
3. **Sentiment gauge** — net tone normalized onto a -100 to +100 scale,
   displayed as a donut gauge with a positive/negative/mixed label.
4. **Signal wheel** — a donut chart breaking down all evaluated sentiment
   signals into supportive, neutral, and opposing, with absolute counts and
   proportional fill bars.
5. **Agreement by round** — a combined bar-and-line chart showing pairwise
   alignment for every completed round, with animated bars rising on load.
6. **Influence** — a ranked list of agents ordered by influence score, each
   with a color-coded avatar and a proportional horizontal bar.
7. **Interaction graph** — an animated cyclic graph showing one-way directed
   flows between agents, with per-edge interaction strength. Click an agent
   node to focus on its connections. A self-cycling highlight rotates through
   each hop when no node is focused.
8. **Opinion trajectory** — a grid board showing each agent's stance
   (aligned / neutral / alternative / unresolved) per round, the candidate
   they support, and a "Changed" flag when a candidate switch occurred.
   A legend at the bottom explains the color coding.

### Conclusion View

Displayed automatically when the discussion completes. Shows:

- **Conclusion article** — the full written conclusion generated by the
  LLM, rendered with semantic sections (Recommendations, Technical Claims
  and Assumptions, Unresolved Conditions), inline bold/code formatting,
  and bullet lists. A consensus badge (positive / partial / negative)
  is shown alongside.
- **Decision signal card** — the consensus status with a support ratio bar
  and count.
- **Support list** — agents that explicitly support the preferred candidate.
- **Preserved dissent** — agents with dissenting positions and their
  reasoning.
- **Unresolved uncertainties** — open questions the discussion did not
  resolve.
- **Strongest alternative** — the runner-up candidate, if one was identified.

---

## Discussion Engine

### Persona Selection

The system supports six predefined personas, each with a distinct role and
weighted priorities across six criteria (capability, latency, reliability,
auditability, operational simplicity, cost efficiency):

| Code | Name | Role |
|---|---|---|
| CM | Capability Maximizer | Principal Research Architect |
| PA | Practical Systems Architect | Staff Systems Engineer |
| SG | Safety and Governance Advocate | Head of AI Safety |
| PU | Product and User Advocate | Head of Product |
| CS | Cost and Sustainability Analyst | Finance & Sustainability Lead |
| SB | Strategic Consensus Builder | VP of Strategy |

Persona definitions are loaded from the Week 2 `Personas.yaml` configuration.
For each new discussion, one seed is selected from `DISCUSSION_RANDOM_SEEDS`.
That selected seed then drives both participant sampling and graph construction,
so an individual run remains reproducible while Docker runs can use different
participant groups. `DISCUSSION_RANDOM_SEED` remains available as a single-seed
fallback.

### Agent Graph

A directed graph is built from the selected participants. Each agent is a
node; edges define the communication topology (who sends messages to whom).
The graph is validated for strong connectivity — every agent must be
reachable from every other agent. The resulting graph structure (agents and
edges) is embedded in the discussion snapshot and used by the frontend to
render the interaction topology.

When `DISCUSSION_DYNAMIC_CHALLENGER_ROUTING` is enabled, message routing
within each round prioritizes sending arguments to the agent most likely to
disagree, creating productive tension.

### Round Execution

Each round proceeds as follows:

1. **Execute round** — for each participant (in graph order), the Week 2
   LLM generates an opinion and a message based on the participant's persona,
   the shared research dossier, and the full history of opinions from
   previous rounds. Message recipients are determined by the graph topology.
2. **Commit round** — opinions are committed to the history, and any new
   candidate proposals are registered in the `proposed_alternatives` map.
3. **Persist & publish** — the updated snapshot is written to disk and a
   `round.completed` event is published to the event sink.

### Decision Synthesis

After all rounds complete, `synthesize_decision()` deterministically
analyzes the final round's opinions to produce a `DecisionSummary`
containing:

- The preferred candidate (plurality vote)
- Consensus status (full consensus, majority, split, etc.)
- Supporting personas
- Dissenting positions with reasoning
- The strongest alternative candidate
- Aggregate utility scores
- Unresolved uncertainties
- Switch conditions (what would change each dissenter's mind)

### Conclusion Generation

The final round's opinions are fed to the LLM to generate a structured
natural-language conclusion. If the conclusion fails a quality check
(e.g., too short, missing required sections), a `ConclusionQualityWarning`
is raised and a fallback conclusion is used. Quality warnings are preserved
in the snapshot's `warnings[]` array and displayed in the frontend.

---

## Analytics Pipeline

### Event-Driven Architecture

The analytics pipeline is fully event-driven rather than polling-based.
The producer emits four types of lifecycle events:

| Event Type | When | Analytics Action |
|---|---|---|
| `discussion.started` | Background task begins | Artifact created |
| `round.completed` | Each round finishes | Round-level analysis runs |
| `discussion.completed` | All rounds finish + conclusion | Full analysis runs (incl. influence) |
| `discussion.failed` | Unrecoverable error | Artifact marked failed |

Each event includes the full discussion snapshot and a unique `event_id`
used as the `Idempotency-Key`. The consumer processes each event exactly
once; duplicate deliveries return 200 without re-running analytics.

### Round-Level Analytics

Computed after each `round.completed` event:

- **Opinion change (stance tracking)** — for each agent, tracks the
  preferred candidate, a numeric stance score, and whether the candidate
  changed compared to the previous round.
- **Agreement** — pairwise candidate alignment across all agents. Produces
  a score (0–1) and disagreement score per round.
- **Sentiment** — evaluates each message for sentiment toward candidates,
  producing per-evaluation scores and labels (positive / negative / neutral).

### Final Analytics

Computed after the `discussion.completed` event (in addition to per-round
metrics):

- **Influence** — per-agent influence score based on how effectively each
  agent's arguments caused others to shift position. Includes reliability
  ratings and recipient counts. This metric requires the complete discussion
  history and is therefore deferred until completion.
- **Full opinion change trajectory** — the complete per-agent stance
  history across all rounds.
- **Aggregate agreement** — all per-round agreement records.
- **Aggregate sentiment** — all per-message sentiment records.

The analytics engine runs the four analysis categories in dependency order:
opinion change → agreement → influence → sentiment.

### Streaming Artifact Store

Analytics results are written to a single durable JSON artifact per
discussion, stored at `outputs/analytics/{discussion_id}/artifact.json`.
The artifact grows incrementally:

- After round 1: contains `round_analyses[0]`
- After round 2: contains `round_analyses[0..1]`
- After completion: contains `round_analyses[0..n]` + `final_analysis`
- After failure: `status` is `"failed"` and `failure` is populated

Each event received is also stored individually for auditability. A
thread-safe lock ensures concurrent events for the same discussion are
serialized.

---

## Data Flow

A complete discussion follows this sequence:

```
User submits objective
        │
        ▼
Producer validates input
        │
        ▼
Week 2 adapter loads objective + personas
        │
        ▼
3–6 participants selected (seeded random or custom roster)
        │
        ▼
Agent graph built (directed, strongly connected)
        │
        ▼
Snapshot persisted with status "queued"
        │
        ▼
Background task starts
        │
        ├──► Event: discussion.started
        │         │
        │         ▼
        │    Consumer creates analytics artifact
        │
        ▼
Shared research phase (RAG queries → evidence pool → dossier)
        │
        ▼
┌─── Round 1 ──────────────────────────────────────┐
│  For each agent (in graph order):                 │
│    LLM generates opinion + message                │
│    Message routed to graph-designated recipient   │
│  Opinions committed to history                    │
│  Snapshot updated, persisted                      │
│  Event: round.completed                           │
│       │                                           │
│       ▼                                           │
│  Consumer runs round analysis                     │
│  (stances, agreement, sentiment)                  │
│  Analytics artifact updated                       │
└───────────────────────────────────────────────────┘
        │
        ▼  (repeat for rounds 2..n)
        │
        ▼
Conclusion generated by LLM
        │
        ▼
Decision summary synthesized deterministically
        │
        ▼
Snapshot finalized with status "completed"
        │
        ├──► Event: discussion.completed
        │         │
        │         ▼
        │    Consumer runs full analysis
        │    (incl. influence)
        │    Final analytics artifact written
        │
        ▼
Frontend detects completion, switches to Conclusion view
```

---

## Project Structure

```
multi-agent-decision-platform/
├── main.py                               # Unified launcher (starts producer + consumer)
├── requirements.txt                      # Python dependencies for all services
├── conftest.py                           # Makes modules importable for pytest
├── Dockerfile                            # Multi-stage build (frontend, backend, cached models)
├── scripts/                              # Model-caching scripts used by the Dockerfile
├── tests/                                # Tests for the unified launcher
├── frontend/                             # React SPA (Vite + TypeScript)
│   ├── src/
│   │   ├── analytics/
│   │   ├── AgentGraph.tsx
│   │   ├── api.ts                        # Typed API client (fetch wrapper)
│   │   ├── App.tsx
│   │   ├── ConclusionView.tsx
│   │   ├── DiscussionHistory.tsx
│   │   ├── DiscussionReplay.tsx
│   │   ├── DossierPreparation.tsx
│   │   ├── layout.tsx
│   │   ├── main.tsx                      # React root mount
│   │   ├── personas.tsx
│   │   ├── styles.ts
│   │   ├── theme.ts
│   │   └── utils.ts
│   ├── index.html                        # HTML entry point
│   ├── package-lock.json
│   ├── package.json                      # npm dependencies and scripts
│   ├── tsconfig.app.json
│   ├── tsconfig.json                     # TypeScript configuration
│   ├── tsconfig.node.json
│   └── vite.config.ts                    # Vite config with /api proxy to :8001
├── services/
│   ├── analytics/                        # Analytics consumer API
│   │   ├── src/
│   │   │   └── analytics/
│   │   │       ├── agreement.py          # Pairwise agreement computation
│   │   │       ├── api.py                # FastAPI consumer application
│   │   │       ├── data_loader.py        # Discussion snapshot parser
│   │   │       ├── engine.py             # Unified analysis entry point
│   │   │       ├── influence.py          # Per-agent influence scoring
│   │   │       ├── models.py             # Analytics data models
│   │   │       ├── opinion_change.py     # Stance tracking and change detection
│   │   │       ├── sentiment.py          # Message sentiment analysis
│   │   │       ├── settings.py           # Environment-backed settings
│   │   │       ├── streaming_events.py   # Producer event parsing
│   │   │       ├── streaming_service.py  # Event-driven analytics orchestration
│   │   │       ├── streaming_store.py    # Durable artifact store
│   │   │       └── utils.py              # Shared utilities
│   │   ├── tests/                        # Analytics test suite
│   │   ├── .env.example                  # Consumer environment template
│   │   ├── conftest.py                   # Analytics test configuration
│   │   ├── consumer_api.py
│   │   ├── README.md
│   │   └── requirements.txt              # Analytics-specific dependencies
│   └── discussion-engine/                # Discussion engine / producer API
│       ├── discussion/
│       │   ├── api.py                    # FastAPI producer application
│       │   ├── artifacts.py              # File-based artifact store
│       │   ├── candidate_identity.py     # Candidate name normalization
│       │   ├── conclusion.py             # LLM conclusion generation
│       │   ├── config.py                 # Pydantic settings (DISCUSSION_ prefix)
│       │   ├── decision.py               # Deterministic decision synthesis
│       │   ├── delivery.py               # Delivery result types
│       │   ├── event_sinks.py            # HTTP event delivery with retry
│       │   ├── events.py                 # Lifecycle event types and construction
│       │   ├── graph.py                  # Directed agent graph with connectivity
│       │   ├── identifiers.py            # Discussion ID generation
│       │   ├── lifecycle.py              # Snapshot persistence + event publishing
│       │   ├── models.py                 # Discussion data models and serialization
│       │   ├── opinion_parser.py         # Structured opinion extraction
│       │   ├── orchestrator.py           # High-level discussion lifecycle
│       │   ├── participants.py           # Seeded participant selection
│       │   ├── prompting.py              # LLM prompt construction
│       │   ├── research.py               # Shared research phase
│       │   ├── round_execution.py        # Per-round agent execution
│       │   ├── runner.py                 # Convenience entry point
│       │   ├── structured_output.py      # Output parsing utilities
│       │   └── week2_adapter.py          # Integration bridge to Week 2
│       ├── schemas/                      # JSON schema definitions
│       │   ├── discussion-event-v1.schema.json
│       │   └── discussion-snapshot-v1.schema.json
│       ├── tests/                        # Discussion-engine test suite
│       ├── week2/                        # Week 2 codebase (personas, LLM, RAG)
│       │   ├── src/
│       │   ├── tests/
│       │   └── Tools/
│       ├── .env.example                  # Producer environment template
│       ├── pytest.ini
│       └── README.md
└── docs/                                 # Documentation
    └── platform-reference.md             # This document
```

---

## Testing

### Backend Tests

From the repository root:

```powershell
pytest tests/ -v
```

The `test_main.py` file tests:

- `_configure_local_connection` — verifies the producer's event endpoint is
  correctly wired to the co-located consumer.
- Token sharing — verifies `ANALYTICS_EVENT_TOKEN` is propagated to
  `DISCUSSION_EVENT_TOKEN`.
- Port validation — verifies that out-of-range port values (e.g., 70000)
  are rejected with a clear error.

### Frontend Build Verification

```powershell
cd frontend
npm run build
```

The TypeScript compiler (`tsc -b`) checks for type errors before Vite
produces the production bundle.

---

## Troubleshooting

### RAG service fails to start

**Symptom:** The producer health endpoint reports `"status": "degraded"` and
`"rag": { "ready": false }`.

**Cause:** The Week 2 codebase is not found at `DISCUSSION_WEEK2_ROOT`, or
the Week 2 `.env` is missing required variables (e.g., `API_KEY`).

**Solution:** Verify the `DISCUSSION_WEEK2_ROOT` environment variable points
to the correct directory. Ensure it contains a `src/agents/` package. Check
the Week 2 `.env` file for required API keys.

### "discussion.started" event delivery fails

**Symptom:** The discussion fails immediately after starting with an
`EventDeliveryError`.

**Cause:** The analytics consumer is not running, or the
`DISCUSSION_EVENT_ENDPOINT` is pointing to an unreachable address. When
`DISCUSSION_EVENT_DELIVERY_REQUIRED=true`, the producer treats delivery
failure as a fatal error.

**Solution:** Ensure the consumer is running on the expected port. If running
both services through `main.py`, the launcher handles this automatically.
If running services independently, set `DISCUSSION_EVENT_ENDPOINT` to the
consumer's event URL.

### Frontend shows "Could not load the discussion"

**Symptom:** The discussion tab shows an error after starting a discussion.

**Cause:** The producer API is not running, the Vite proxy is misconfigured,
or the discussion snapshot file is corrupted.

**Solution:** Verify the producer is running at `http://127.0.0.1:8001`.
Check `frontend/vite.config.ts` to confirm the proxy target matches the
producer's address. Try accessing the discussion URL directly in the
browser (e.g., `http://127.0.0.1:8001/api/v1/discussions/{id}`).

### Analytics tab shows "Analytics begin after round one"

**Symptom:** The analytics tab remains empty even after rounds have completed.

**Cause:** The analytics consumer did not receive or process the
`round.completed` event. This can happen if the consumer crashed, if event
delivery silently failed (when `DISCUSSION_EVENT_DELIVERY_REQUIRED=false`),
or if the consumer's artifact store path is different from what the producer
expects.

**Solution:** Check the consumer's terminal output for errors. Verify both
services use consistent `ANALYTICS_OUTPUT_DIR` settings. If using the
unified launcher (`main.py`), event routing is configured automatically.

### Port conflicts

**Symptom:** `OSError: [Errno 48] Address already in use` or similar.

**Cause:** Another process is using port 8000, 8001, or 8002.

**Solution:** Change the conflicting port via environment variables
(`DISCUSSION_API_PORT`, `ANALYTICS_API_PORT`) or stop the conflicting
process.

### "Week 2 Personas.yaml contains no personas"

**Symptom:** A `Week2IntegrationError` is raised during discussion creation.

**Cause:** The `Personas.yaml` file in the Week 2 directory is missing,
empty, or malformed.

**Solution:** Verify that
`{DISCUSSION_WEEK2_ROOT}/src/agents/Personas.yaml` exists and contains
valid persona definitions with at least two entries.

### Idempotency-Key conflict (409)

**Symptom:** The consumer rejects an event with HTTP 409.

**Cause:** The same `event_id` was submitted with different content. This
should not happen in normal operation — it indicates a state corruption
or a bug in event construction.

**Solution:** Check the producer logs for the original event payload.
If necessary, clear the consumer's event store at
`outputs/analytics/{discussion_id}/events/` and re-run.

### Frontend build fails with TypeScript errors

**Symptom:** `npm run build` reports type errors.

**Solution:** Run `npm install` to ensure all type definitions are present.
Check that `@types/react` and `@types/react-dom` versions match the
installed React version in `package.json`.
