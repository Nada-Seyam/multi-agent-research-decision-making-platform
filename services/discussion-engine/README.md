# Week 3 Agent Collaboration

Week 3 is a multi-agent discussion runner built on top of the Week 2 research framework. It gives several independent personas the same structured objective and consolidated research dossier, lets them exchange opinions over multiple rounds, and records the complete discussion for inspection.

The system is designed for decisions where several alternatives are technically viable and where the correct answer depends on explicit priorities. It does not assume that one architecture is universally best.

The central design principle is separation of responsibilities:

- The planner chooses a bounded set of broad, decision-level research queries.
- The evidence pool retrieves and qualifies context for those queries.
- The consolidator builds a neutral, traceable dossier.
- Personas interpret the dossier according to their own priorities.
- The discussion orchestrator routes opinions, enforces round barriers, and records state.
- The orchestrator produces a deterministic final decision summary from recorded utilities while preserving dissent and unresolved conditions.

## Contents

- [What the system does](#what-the-system-does)
- [High-level architecture](#high-level-architecture)
- [Repository layout](#repository-layout)
- [Execution lifecycle](#execution-lifecycle)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running a discussion](#running-a-discussion)
- [Local FastAPI](#local-fastapi)
- [Discussion history and replay](#discussion-history-and-replay)
- [Objective files](#objective-files)
- [Personas](#personas)
- [Research pipeline](#research-pipeline)
- [Discussion protocol](#discussion-protocol)
- [Research boundary](#research-boundary)
- [Outputs and logs](#outputs-and-logs)
- [Data model](#data-model)
- [Evidence discipline](#evidence-discipline)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Development guide](#development-guide)
- [Known limitations](#known-limitations)

## What the system does

Given an objective file, the runner:

1. Loads and validates the objective.
2. Loads the available personas from Week 2.
3. Selects a reproducible subset of personas.
4. Builds a connected directed discussion graph.
5. Plans a bounded initial research cycle.
6. Retrieves evidence for the plan's queries.
7. Consolidates accepted evidence into a dossier.
8. Runs each selected persona for each discussion round.
9. Makes previous-round opinions visible to connected neighbors.
10. Records missing evidence as structured gaps and critical uncertainties without expanding the fixed dossier.
11. Validates structured opinion metadata, including architecture changes and switch conditions.
12. Writes a concise result JSON and separate detailed logs.

The result is a record of reasoning and disagreement. It is not a guarantee that the final architecture is objectively optimal.



### Week 3 and Week 2 responsibilities

Week 3 owns discussion orchestration. Week 2 owns the reusable research and persona framework.

Week 3 calls Week 2 through [discussion/week2_adapter.py](discussion/week2_adapter.py). This adapter is the only module that handles Week 2's import-path layout. The rest of Week 3 uses the adapter rather than importing Week 2 internals directly.

The main Week 2 functions used by Week 3 are:

- `load_objective(...)`
- `get_personas()`
- `plan_research(...)`
- `build_evidence_pool(...)`
- `consolidate_evidence(...)`
- `run_for_persona(...)`

## Repository layout

```text
discussion-engine/
├── README.md                         # This document
├── .env.example                      # Week 3 configuration template
├── .env                              # Local Week 3 settings; do not commit secrets
│
├── discussion/                       # Week 3 orchestration package
│   ├── __init__.py
│   ├── __main__.py                   # CLI: python -m discussion
│   ├── api.py                        # Upload API and local mock consumer
│   ├── artifacts.py                  # Atomic snapshots and event journal storage
│   ├── conclusion.py                 # Final-opinion-only LLM synthesis
│   ├── config.py                     # DiscussionSettings
│   ├── delivery.py                   # Replay of pending consumer deliveries
│   ├── event_sinks.py                # Consumer event delivery adapters
│   ├── events.py                     # Versioned discussion event envelopes
│   ├── graph.py                      # AgentGraph and graph builders
│   ├── identifiers.py                # Run and objective identifiers
│   ├── lifecycle.py                  # Snapshot persistence and event publication
│   ├── models.py                     # Discussion and output data models
│   ├── opinion_parser.py             # Persona response parsing and validation
│   ├── orchestrator.py               # High-level discussion coordinator
│   ├── participants.py               # Reproducible persona selection
│   ├── prompting.py                  # Per-round persona prompt construction
│   ├── research.py                   # Initial shared research and evidence bounds
│   ├── round_execution.py            # One-round execution and state commit
│   ├── runner.py                     # Public run_discussion entry point
│   └── week2_adapter.py              # Boundary around Week 2 APIs
│
├── week2/                            # Reused Week 2 research framework
│   ├── src/agents/
│   │   ├── consolidator.py           # ResearchDossier and evidence qualification
│   │   ├── config.py                 # Week 2 provider/retrieval settings
│   │   ├── evidence_pool.py           # Parallel query resolution
│   │   ├── graph.py                  # Persona prompt/call graph
│   │   ├── llm.py                    # LLM provider client
│   │   ├── objective.py              # Objective parser and Pydantic models
│   │   ├── planner.py                # Bounded research planning
│   │   ├── prompts.py                # Persona prompt construction
│   │   ├── resolve_context.py        # Retrieval, grading, expansion, fallback
│   │   └── retrieval.py              # RAG client integration
│   ├── src/configs/
│   │   └── Personas.yaml             # Persona definitions and shared rules
│   └── Tools/                        # RAG and web-search integrations
│
├── tests/                            # Week 3 tests
│   ├── test_graph.py
│   ├── test_participants.py
│   ├── test_research.py
│   ├── test_runner.py
│   └── test_week2_adapter.py
│
├── week2/tests/                      # Week 2 unit tests
├── outputs/discussions/              # Concise result JSON files (created at runtime)
└── logs/discussions/                 # Detailed per-discussion logs (created at runtime)
```

Generated files and local environment files should not be committed unless the project explicitly requires them.

## Execution lifecycle

### 1. Load the objective

The CLI receives an objective path. The Week 2 objective loader parses it into an `Objective` model containing:

- `goal`
- `constraints`
- `core_topics`
- `evidence_requirements`
- `decision_criteria`
- `strict_notes`
- `final_goal`

The parser supports the current uppercase section format and a legacy lower-case `key:` / `key ->` format. If a free-form file does not contain the expected sections, the loader can attempt an LLM-based extraction fallback.

### 2. Select participants

Random selection remains the default: personas are sorted before a seeded sampler selects the configured number of participants. This makes selection reproducible for the same persona set, participant count, and seed. The browser API can also submit an explicit custom roster, whose order is preserved for graph construction.

The default participant count is four. Every random or custom roster requires between three and six participants so the discussion topology forms a graph rather than a two-node line.

### 3. Build the discussion graph

The graph is directed and must be strongly connected. Strong connectivity means every participant can reach every other participant through directed edges.

The graph is used for visibility and message routing. It does not decide which architecture is correct.

### 4. Build shared research

The initial research cycle runs once per discussion, before round one:

```text
Objective
  → planner
  → N research queries
  → evidence pool
  → context resolution and source fallback
  → neutral consolidator
  → shared dossier
```

The dossier is shared by all personas. This prevents every persona from independently running the full research pipeline and keeps the evidence boundary visible.

### 5. Run discussion rounds

For each round:

1. The orchestrator creates a snapshot of each persona's previous opinion.
2. Each persona receives the objective, shared dossier, its persona rules, its own previous opinion, and visible neighbor opinions.
3. Personas respond independently through bounded concurrent model calls.
4. The orchestrator parses structured markers from each response.
5. The complete round is committed only after all selected participants have spoken.
6. Missing evidence is recorded as a gap or critical uncertainty; no round-time retrieval is started.

This round barrier prevents a participant early in a round from seeing partial opinions from the same round while another participant sees a different state.

### 6. Persist results

The CLI writes a concise result file under `outputs/discussions/` and the orchestrator writes detailed data under `logs/discussions/<discussion_id>/`.

## Installation

The commands below assume a supported Python environment and a shell opened in this directory.

### Create a virtual environment

Dependencies for this service are listed in the repository-root `requirements.txt`. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd services/discussion-engine
```

All commands in this README assume the virtual environment is active and the shell is in `services/discussion-engine/`. Using `python -m ...` is the safest form for tests and automation because it cannot accidentally select another Python installation.

### Configure Week 3

Copy the example settings file:

```powershell
Copy-Item .env.example .env
```

Set `DISCUSSION_RANDOM_SEEDS` to a JSON list of integers. The checked-in example
already provides a recommended pool. Existing deployments may continue using
the single `DISCUSSION_RANDOM_SEED` variable instead.

### Configure Week 2 providers

Week 2 reads provider and retrieval settings from its own environment file at:

```text
week2/src/.env
```

At minimum, configure the required `API_KEY`. The Week 2 settings also support an OpenAI-compatible provider through values such as:

```dotenv
API_KEY=replace-with-your-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-model-name
RETRIEVAL_API_BASE_URL=http://localhost:8000
```

For a local or compatible provider, set `LLM_BASE_URL` and `LLM_MODEL` accordingly. Do not place secrets in `README.md`, `.env.example`, objective files, or persona configuration.

## Configuration

Week 3 settings use the `DISCUSSION_` environment prefix and are defined in [discussion/config.py](discussion/config.py).

| Variable | Meaning | Default or requirement |
|---|---|---|
| `DISCUSSION_NUM_PARTICIPANTS` | Number of personas selected for a discussion | `4`, range `3–6` |
| `DISCUSSION_PARTICIPANT_SELECTION_MODE` | Use seeded `random` selection or a `custom` roster | `random` |
| `DISCUSSION_SELECTED_PARTICIPANT_IDS` | JSON list of persona ids when selection mode is `custom` | Empty for random selection; 3–6 unique ids for custom |
| `DISCUSSION_NUM_ROUNDS` | Number of discussion rounds | `4`, minimum `3` |
| `DISCUSSION_RANDOM_SEEDS` | JSON seed pool; one seed is selected for each new discussion | Required unless using `DISCUSSION_RANDOM_SEED` |
| `DISCUSSION_RANDOM_SEED` | Backward-compatible single seed | Used when no seed pool is configured |
| `DISCUSSION_WEEK2_ROOT` | Week 2 root containing `src/agents` | `week2` |
| `DISCUSSION_OUTPUT_DIR` | Concise result directory | `outputs/discussions` |
| `DISCUSSION_LOGS_DIR` | Detailed log directory | `logs/discussions` |
| `DISCUSSION_MAX_INITIAL_QUERIES` | Number of initial research queries | `4` in the model; example uses `5` |
| `DISCUSSION_MAX_EVIDENCE_CHUNKS_PER_QUERY` | Maximum chunks retained from each query | `4` |
| `DISCUSSION_MAX_EVIDENCE_CHARS_PER_CHUNK` | Maximum characters retained per chunk | `6000` |
| `DISCUSSION_MAX_TOTAL_EVIDENCE_CHUNKS` | Global evidence bound before consolidation | `50` in the model; example uses `40` |
| `DISCUSSION_MAX_PERSONA_EVIDENCE_CHARS` | Maximum dossier characters repeated in each persona prompt; full evidence remains in artifacts | `24000` |
| `DISCUSSION_MAX_PARALLEL_AGENTS` | Maximum concurrent persona model calls within one round | `4`; set to `1` for serial execution |
| `DISCUSSION_MAX_CONCURRENT_DISCUSSIONS` | Maximum discussions executing at once | `2` |
| `DISCUSSION_MAX_QUEUED_DISCUSSIONS` | Additional discussion jobs accepted before returning `429` | `8` |
| `DISCUSSION_DYNAMIC_CHALLENGER_ROUTING` | Route the most decision-distant previous-round opinion to each persona in addition to graph neighbors | `true` |
| `DISCUSSION_UTILITY_CONSISTENCY_TOLERANCE` | Maximum weighted-utility gap within which a persona may retain a non-leading candidate as a near tie | `0.25` |
| `DISCUSSION_PERSIST_ARTIFACTS` | Persist the current public snapshot and delivery journal | `true` |
| `DISCUSSION_EVENT_ENDPOINT` | Optional consumer endpoint that receives lifecycle events | unset |
| `DISCUSSION_EVENT_TOKEN` | Optional bearer token for the consumer endpoint; never serialized into discussion output | unset |
| `DISCUSSION_EVENT_TIMEOUT_SECONDS` | Timeout for one consumer request | `10` |
| `DISCUSSION_EVENT_MAX_ATTEMPTS` | Maximum delivery attempts for each event | `5` |
| `DISCUSSION_EVENT_RETRY_INITIAL_SECONDS` | Initial exponential-backoff delay | `1` |
| `DISCUSSION_EVENT_RETRY_MAX_SECONDS` | Maximum exponential-backoff delay | `30` |
| `DISCUSSION_EVENT_DELIVERY_REQUIRED` | Fail the producer when an event cannot be delivered | `false` |

The example file is intentionally explicit and is a useful starting point:

```dotenv
DISCUSSION_NUM_PARTICIPANTS=4
DISCUSSION_NUM_ROUNDS=4
DISCUSSION_RANDOM_SEEDS=[47,73,101,149]
DISCUSSION_WEEK2_ROOT=week2
DISCUSSION_OUTPUT_DIR=outputs/discussions
DISCUSSION_LOGS_DIR=logs/discussions
DISCUSSION_MAX_INITIAL_QUERIES=5
DISCUSSION_MAX_EVIDENCE_CHUNKS_PER_QUERY=4
DISCUSSION_MAX_EVIDENCE_CHARS_PER_CHUNK=6000
DISCUSSION_MAX_TOTAL_EVIDENCE_CHUNKS=40
DISCUSSION_MAX_PERSONA_EVIDENCE_CHARS=24000
DISCUSSION_MAX_PARALLEL_AGENTS=4
DISCUSSION_MAX_CONCURRENT_DISCUSSIONS=2
DISCUSSION_MAX_QUEUED_DISCUSSIONS=8
DISCUSSION_DYNAMIC_CHALLENGER_ROUTING=true
DISCUSSION_UTILITY_CONSISTENCY_TOLERANCE=0.25
DISCUSSION_PERSIST_ARTIFACTS=true
DISCUSSION_EVENT_ENDPOINT=http://consumer:8000/api/v1/discussion-events
DISCUSSION_EVENT_TOKEN=
DISCUSSION_EVENT_TIMEOUT_SECONDS=10
DISCUSSION_EVENT_MAX_ATTEMPTS=5
DISCUSSION_EVENT_RETRY_INITIAL_SECONDS=1
DISCUSSION_EVENT_RETRY_MAX_SECONDS=30
DISCUSSION_EVENT_DELIVERY_REQUIRED=false
```

Relative paths are resolved relative to the Week 3 project root. Absolute paths are also accepted.

### Utility consistency tolerance

`DISCUSSION_UTILITY_CONSISTENCY_TOLERANCE` controls how strictly a persona's declared architecture must follow its own weighted candidate matrix. For every candidate, the model scores every objective criterion from `0` to `5`. The runner maps the persona's semantic priority weights onto those criteria, normalizes the weights, and computes:

```text
weighted utility(candidate) = sum(criterion score × normalized persona weight)
```

If the declared architecture trails the matrix winner by more than the tolerance, the runner normalizes the architecture to the matrix winner. If the difference is within the tolerance, the declared option is treated as a defensible near tie.

- A **higher** tolerance gives personas more flexibility to retain a close non-leading option. It normally produces fewer architecture switches.
- A **lower** tolerance makes the matrix ranking more binding. It normally makes agents more responsive to changed scores, but an extremely low value can amplify minor scoring noise.
- The tolerance is measured on the weighted `0`–`5` utility scale, not as a percentage.
- The tolerance does not by itself authorize a mind change. A switch must still satisfy the switch-validation contract described below.

`0.25` is the conservative default. For an influence-focused experiment, `0.10` is a reasonable starting value. Compare several fixed-seed runs before concluding that a setting is stable.

### Week 2 settings

Week 2 settings are separate from Week 3 settings. Important values include:

| Variable | Meaning |
|---|---|
| `API_KEY` | Provider credential or required local-provider placeholder |
| `LLM_BASE_URL` | OpenAI-compatible model endpoint |
| `LLM_MODEL` | Model name sent to the provider |
| `LLM_MAX_CONCURRENT_REQUESTS` | Process-wide provider request cap |
| `LLM_REQUEST_TIMEOUT_SECONDS` | Timeout for one provider request |
| `LLM_MAX_RETRIES` | Provider retry count |
| `RETRIEVAL_API_BASE_URL` | RAG API base URL |
| `RETRIEVAL_TOP_K` | Retrieval depth |
| `QUERY_EXPANSION_ATTEMPTS` | Expansion attempts per source/query |
| `CONTEXT_SOURCES_ORDER` | Source order, normally `rag,web` |
| `EVIDENCE_POOL_MAX_WORKERS` | Bounded parallel query resolution |
| `RESEARCH_MAX_MODEL_CALLS` | Shared cap for grading and expansion calls in one research run |
| `MEMORY_ENABLED` | Cross-run rolling memory switch |

The retrieval resolver accepts `rag` and `web` as source names. If all configured sources fail or are rejected, it returns best-effort context and marks the result as insufficient rather than silently treating it as strong evidence.

## Running a discussion

Run the Week 3 CLI from the Week 3 root:

```powershell
python -m discussion --objective week2/src/configs/objective.txt
```

The command prints progress and writes a result similar to:

```text
discussion result saved to outputs\discussions\<discussion_id>.json
```

Use a custom objective path:

```powershell
python -m discussion --objective path\to\objective.txt
```

### Programmatic usage

```python
from discussion.config import DiscussionSettings
from discussion.runner import run_discussion
from discussion.week2_adapter import load_objective

settings = DiscussionSettings()
objective = load_objective(
    "week2/src/configs/objective.txt",
    settings.resolve_week2_root(),
)
result = run_discussion(objective, settings=settings)

print(result.discussion_id)
print(result.to_json())
```

The public runner returns a `DiscussionResult`.

### Running the direct Week 2 cycle

Week 2 also has its own CLI. It is useful for testing the research/persona pipeline without Week 3 round orchestration:

```powershell
python -m src.agents --objective src/configs/objective.txt
```

Run this command from the `week2` directory, or ensure the Week 2 root is on `PYTHONPATH`.

The direct Week 2 cycle has different output and memory behavior. Week 3 discussions should normally be started with `python -m discussion`.

## Local FastAPI

The local development API starts the configured Week 2 RAG service as a child
process, waits for its health endpoint, and shuts it down with the API. It also
hosts an in-memory mock consumer for incremental producer events. Only one
long-running command is needed:

```powershell
python -m discussion.api
```

The API packages are part of the repository-root `requirements.txt`; install them from the repository root if they are not already available:

```bash
python -m pip install -r requirements.txt
```

Check both the producer and RAG status:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Upload a UTF-8 `.txt` objective and wait for the complete discussion result:

```powershell
$response = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8001/api/v1/discussions/upload `
  -Form @{ objective_file = Get-Item "C:\path\to\objective.txt" }

$response | ConvertTo-Json -Depth 100
```

Swagger UI is available at `http://127.0.0.1:8001/docs`. The discussion
request includes shared research/RAG, every configured agent round, artifact
persistence, final synthesis, and delivery of lifecycle events to the mock
consumer at `POST /api/v1/discussion-events`.

## Discussion history and replay

Every persisted snapshot automatically appears in the frontend's Discussion
history section. Opening an entry loads the saved snapshot and analytics; it
does not rerun research or call any persona. Replay controls step through room
preparation, each completed round, and the final completed or failed state.
The controls support previous/next navigation, scrubbing, playback, and speed
selection.

The history API returns lightweight summaries rather than full discussion
artifacts:

```text
GET /api/v1/discussions?limit=20&offset=0
GET /api/v1/discussions?status=completed&limit=20&offset=0
```

Load the full saved discussion and its replay checkpoints separately:

```text
GET /api/v1/discussions/<discussion_id>
GET /api/v1/discussions/<discussion_id>/replay
```

Replay checkpoint timing comes from the durable `events.jsonl` journal. Runs
created before lifecycle journaling are also supported: their round snapshots
are replayed in order and the response includes a warning that timing is
approximate. Corrupt legacy files are skipped by the history listing so one
bad artifact cannot make the archive unavailable.

## Objective files

An objective needs only an `OBJECTIVE`, `PRIMARY CONSTRAINTS`, and `STRICT NOTES` section. The other sections are optional but useful when the decision needs explicit evidence or evaluation guidance:

```text
OBJECTIVE
Recommend the best primary architecture for the target system. Discover the solution from the evidence rather than from an allowed-answer list.

PRIMARY CONSTRAINTS
- Training compute is limited.
- The system must be practical to operate.

CORE TOPICS TO DISCUSS
1. Decision trade-offs
- Quality versus cost
- Latency versus reliability
- Simplicity versus flexibility

EVIDENCE REQUIREMENTS
- Separate measured evidence from engineering judgment.
- Record source limitations and applicability.

DECISION CRITERIA
- Compare the proposed solution with its strongest serious alternative.
- Explain which priority shift changes the recommendation.

STRICT NOTES
- Do not assume one architecture is objectively best.

FINAL GOAL
Recommend the architecture that best matches justified priorities.
```

### Parsing rules

- Plain section headers must match the expected uppercase names exactly for deterministic parsing. Markdown headings and bold headings are also supported.
- `OBJECTIVE` and `FINAL GOAL` are joined into paragraphs.
- Constraint, evidence, criteria, and note sections accept bullet, numbered, or plain non-empty lines.
- Under `CORE TOPICS TO DISCUSS`, a top-level numbered line starts a topic.
- Bullet lines beneath a topic become that topic's subpoints.
- Topic IDs are derived by slugifying the topic title.
- `OBJECTIVE`, `PRIMARY CONSTRAINTS`, and `STRICT NOTES` are the minimum deterministic uppercase format. If `FINAL GOAL` is omitted, a neutral evidence-backed final instruction is supplied.
- If the minimum sections are missing, the loader attempts an LLM extraction fallback.
- For reproducibility and predictable planning, use the explicit uppercase format rather than relying on fallback extraction.

### Writing a good objective

An objective should describe a decision, not enumerate its allowed answers. Good objectives:

- Define the problem boundary while leaving solution discovery open.
- State constraints without silently converting them into absolute vetoes.
- Name the trade-offs that matter.
- Specify what evidence would change the decision.
- Distinguish hard requirements from preferences.
- Avoid language such as “the only viable architecture” unless that is itself what the research must test.

### Designing an objective that permits genuine influence

An objective cannot guarantee that agents will change architecture, and it should not order them to switch. It can make real influence possible by creating a decision in which reasonable priorities lead to different initial rankings and peer arguments can cross explicit decision thresholds.

A strong debate objective should:

1. Define tensions that allow more than one technically serious solution to emerge without naming those solutions in advance.
2. Use atomic criteria such as capability, latency, reliability, auditability, operational complexity, and lifecycle cost. Avoid combining several values into one criterion because that weakens persona-specific weighting.
3. Include crossover trade-offs. For example, the raw-capability leader may lose after workload weighting, or the cheapest average proposal may lose under tail-risk analysis.
4. State enough workload detail for the agents to calculate consequences instead of relying on slogans.
5. Ask each persona to identify a measurable condition under which its strongest alternative would win.
6. Avoid a hard constraint that eliminates every option except one before the discussion begins.

The dossier is shared from Round 1 onward. Later rounds do not introduce secret facts. A genuine influence event occurs when a peer exposes an overlooked consequence, challenges an assumption, changes the application of fixed evidence, or shows that the persona's own switch condition is already satisfied.

## Personas

Personas are defined in [week2/src/configs/Personas.yaml](week2/src/configs/Personas.yaml). The checked-in configuration includes roles such as:

- `capability_maximizer`
- `practical_architect`
- `strategic_consensus_builder`
- `product_user_advocate`

The exact available set is loaded at runtime, so a deployment may add or remove personas through the YAML file.

Each persona should provide:

- A point of view or priority weighting.
- Important concerns and failure modes.
- A willingness to consider evidence that challenges its initial view.
- A requirement to participate in discussion without blindly agreeing.

Personas are not supposed to discuss in isolation. The orchestrator supplies neighbor opinions based on the graph, and the prompt asks each persona to evaluate the strongest version of the neighbor's claim.

### Persona selection

Only persona mappings are selectable. The special `shared_rules` mapping is excluded from participant selection and is merged into individual persona prompts by the Week 2 configuration layer.

If the requested participant count exceeds the number of available personas, the run fails with `ParticipantSelectionError`.

### Keeping personas independent

A useful persona response should:

1. State its current architecture and priorities.
2. Evaluate evidence rather than merely repeat the dossier.
3. Identify the strongest alternative from another value perspective.
4. Name a falsifiable switch condition.
5. State at least one uncertainty or limitation.
6. Update only when the evidence changes the ranking of criteria.

Agreement is not automatically progress. Convergence is meaningful only when competing alternatives received comparable evidence and the personas explain why the disagreement was resolved.

## Research pipeline

### Planner

The planner receives the complete structured objective and optional prior memory. It must produce exactly the configured number of queries.

Queries are decision-level coverage units. The planner is instructed to group related requirements rather than create one query for every bullet. It also attempts to:

- Cover distinct decision dimensions.
- Cover the goal and decision dimensions broadly enough for different solutions to emerge.
- Include limitations, counterevidence, or decision-changing conditions.
- Avoid near-duplicates and generic padding.
- Treat components as supporting details unless the objective makes them separate alternatives.

The returned plan is validated for:

- Exact query count.
- Non-empty queries.
- Duplicate normalized queries.

If the LLM returns malformed JSON, an incorrect number of queries, empty queries, or duplicates, the planner falls back to a deterministic bounded plan. The fallback uses objective topics and broad cross-cutting questions instead of repeating the overall goal.

### Evidence pool

The evidence pool resolves each planned query independently, with bounded concurrency. It preserves input order in the returned list even though queries may execute in parallel.

Each `EvidenceResult` includes:

- Original query.
- Resolution context.
- Original query order.
- Retrieval status.

Retrieval status describes sufficiency, not preference:

- `accepted`: the context grader accepted the result.
- `partial`: chunks exist but the context was not fully accepted.
- `insufficient`: no usable chunks were returned.
- `failed`: the resolver raised an exception.

The evidence pool does not choose an architecture.

### Context resolution

For each query, Week 2's resolver:

1. Calls the configured source, normally local RAG first.
2. Grades the returned chunks against the original query.
3. Expands the query when the context is rejected and expansion budget remains.
4. Escalates to the next configured source when necessary.
5. Preserves rejected chunks for audit without passing them into the valid evidence stream for objective-aware runs.

The configured source order is controlled by `CONTEXT_SOURCES_ORDER`. The default source registry recognizes `rag` and `web`.

### Evidence bounding

Before consolidation, Week 3 applies three bounds:

- Maximum chunks per query.
- Maximum characters per chunk.
- Maximum total chunks across all queries.

This protects prompt size and keeps a single prolific query from dominating the dossier.

### Consolidator

The consolidator produces a neutral `ResearchDossier`. It is responsible for organization and qualification, not recommendation.

The dossier preserves metadata such as:

- Research and resolved query.
- Source name and source origin.
- Section and chunk ID.
- Evidence tier.
- Evidence quality.
- Applicability metadata.
- Limitations.
- Applicability reasoning.
- Retrieval resolution status.
- Answered requirement IDs.

The consolidator also:

- Deduplicates exact repeated evidence.
- Assigns evidence tiers.
- Assigns excerpts to objective topics.
- Detects potential disagreements.
- Reports topic coverage.
- Reports unresolved requirements.
- Renders transferability limits.

### Evidence tiers

The dossier uses the following conceptual tiers:

- `F`: the source directly supports the stated claim.
- `I`: engineering inference derived from evidence.
- `S`: speculation or an unverified extrapolation.

An `F` label does not mean the source proves the claim for the current objective. It only means the source supports the claim in its own reported setup. Scale, hardware, workload, metric, and model differences still matter.

### Contradictions

Different results are not automatically contradictions. A difference in hardware, model scale, workload, metric, or evaluation setup should not be classified as a contradiction unless the compared conditions are materially comparable.

When disagreements are detected, the dossier records the relevant excerpt indices and description rather than silently selecting one result.

## Discussion protocol

### How influence works

Mind changes are produced by an explicit update loop rather than by asking personas to be agreeable:

```text
persona priorities
  → discover a primary solution and at least one serious alternative
  → score those proposals on every decision criterion
  → compute persona-weighted utilities
  → declare an architecture and a falsifiable switch test
  → receive previous-round neighbor and challenger opinions
  → reconsider scores, assumptions, and the strongest alternative
  → retain, narrow, lower confidence, or switch
  → validate the update against the matrix and switch contract
```

Round 1 establishes an independent baseline. In subsequent rounds, every persona receives its own prior state and the previous-round opinions visible through the discussion graph. When dynamic challenger routing is enabled, the orchestrator also selects the opinion with the greatest decision distance: first by architecture disagreement, then utility-distance, then decisive-criterion disagreement. This gives each persona a substantive counterargument without making that counterargument authoritative.

The round barrier matters. All participants in a round reason from the same completed previous-round snapshot, so an early participant cannot influence a later participant in the same round through partially committed state.

Influence can produce several valid outcomes:

- `switch`: the selected architecture changes and the counterfactual condition is satisfied.
- `narrow`: the ranking or interpretation moved, but the leading options remain within the utility tolerance.
- `lower_confidence`: a decision-critical issue remains unresolved, so the architecture is retained conditionally.
- `retain`: the strongest challenge does not overturn the persona's ranking.

Only `switch` sets `changed_mind=true`. The other outcomes still allow the output to record ranking, assumption, and confidence changes as material influence.

### Opinion markers

Persona responses are free-form text followed by structured markers. The current protocol asks for:

```text
CANDIDATE_EVAL: <persona-defined solution_id> | scores=<criterion_id>:<0..5>,... | evidence=<citation numbers or NONE> | counter=<citation numbers or NONE> | feasibility=feasible|conditional|infeasible|unknown | assumptions=<semicolon-separated or NONE> | gaps=<semicolon-separated or NONE> | combination=direct|partial|indirect|none|not_applicable
ARCHITECTURE: <one canonical architecture family>
COMPONENTS: <optional implementation choices, or NONE>
CONSIDERED: <alternatives evaluated, or NONE>
SWITCH_TEST: current=<current solution> | alternative=<different serious solution> | condition=<measurable condition under which the alternative wins> | effect=alternative_wins | status=satisfied|contradicted|unresolved
STRONGEST_ALTERNATIVE: <one seriously evaluated alternative>
CRITERIA_SCORES: <current and alternative scores from 0 to 5>
DECISIVE_CRITERION: <criterion driving the current view>
UPDATE_DECISION: retain, narrow, lower_confidence, or switch
UPDATE_REASON: <reason the strongest neighbor claim did or did not change the view>
POSITION: <human-readable repeat of ARCHITECTURE>
CONFIDENCE: <number from 0 to 1>
CHANGE_OF_MIND: yes or no
CHANGE_REASON: <short reason if yes; otherwise NONE>
CHALLENGED_CLAIM: <architecture-level claim tested, or NONE>
DECISION_CRITICAL: yes or no
CRITICAL_UNCERTAINTY: <unresolved decisive claim, or NONE>
```

The objective supplies no candidate registry. In Round 1, each persona independently proposes one primary solution and at least one serious alternative. It emits a `CANDIDATE_EVAL` for both and scores every configured criterion. In later rounds, previously proposed solutions are shown for comparison but are explicitly not an allowed-answer list; a persona may introduce a new solution in any round.

A complete per-persona matrix contains at least two evaluated solutions and every objective criterion for each one. Missing alternatives or scores produce `incomplete_candidate_evaluation` and cap confidence.

`CRITERIA_SCORES` is a human-readable summary. The machine-computed utility comes from the repeated `CANDIDATE_EVAL` records, not from that summary marker.

### Open solution identifiers

The parser no longer contains a domain-specific alias dictionary or predefined architecture families. `ARCHITECTURE` is normalized mechanically into a stable lowercase identifier:

```text
"Novel Recurrent Design" → novel_recurrent_design
"Hierarchical Memory Architecture" → hierarchical_memory_architecture
```

No identifier is accepted or rejected because it appears in a hard-coded catalog. Components such as normalization, quantization, batching, or optional modules should still not be used to hide the primary decision. The architecture field names the proposed primary solution; components describe implementation choices.

### Switch validation

A persona may not change architecture merely because it weakened its current argument. From Round 2 onward, a switch is accepted only when all of the following are true:

- `ARCHITECTURE` differs from the previously validated architecture.
- `UPDATE_DECISION` is `switch`.
- `SWITCH_TEST.current` equals the previously validated architecture.
- `SWITCH_TEST.alternative` equals the newly selected architecture.
- The alternative differs from the current solution.
- `SWITCH_TEST.condition` is concrete and states when the alternative wins.
- `SWITCH_TEST.effect` is exactly `alternative_wins`.
- `SWITCH_TEST.status` is `satisfied`.

Round 1 has no previous architecture. Its switch test establishes a counterfactual baseline: `current` must equal the Round 1 architecture and `alternative` must be another serious solution discovered by the persona.

If the switch condition is unresolved, contradicted, unsatisfied, or missing, the orchestrator preserves the prior architecture and records an unresolved switch status.

If structured metadata says `retain` while the proposed architecture differs from the previous architecture, the orchestrator preserves the previous architecture and records a metadata conflict.

This prevents a persona from changing its vote while its own stated evidence condition remains unresolved.

Example of a valid later-round switch:

```text
ARCHITECTURE: dynamically_computed_transformer
SWITCH_TEST: current=dense_transformer | alternative=dynamically_computed_transformer | condition=the fixed evidence shows the severe-incident reliability gain exceeds the stated operational-complexity penalty | effect=alternative_wins | status=satisfied
UPDATE_DECISION: switch
CHANGE_OF_MIND: yes
CHANGE_REASON: The safety persona showed that my original workload weighting understated severe-incident loss, crossing my declared reliability threshold.
```

The prose marker `CHANGE_OF_MIND` is informative, but the validated architecture transition is the source of truth. Claiming `yes` without a valid transition does not create a mind change.

### Utility ranking and architecture normalization

For each persona and round, the orchestrator:

1. Maps persona priorities onto the objective's criteria.
2. Normalizes the resulting criterion weights so they sum to `1`.
3. Parses all `CANDIDATE_EVAL` scores.
4. Computes the weighted utility of each candidate.
5. Records the matrix winner in `ranked_candidate` and the top-two difference in `ranking_margin`.
6. Replaces a declared architecture with the matrix winner when the declared choice trails it by more than `DISCUSSION_UTILITY_CONSISTENCY_TOLERANCE`.

When the matrix winner changes between rounds but its margin is within the tolerance, a claimed `retain` is recorded as `narrow` with status `near_tie_ranking_update`. This captures real movement without falsely calling a close ranking change an architecture switch.

### Confidence and material updates

Confidence is bounded rather than accepted blindly. It is capped when the persona's alternative evaluation is incomplete, its evaluation records lack supporting and adverse evidence, or a decision-critical switch condition is unresolved. Because the dossier is fixed, confidence cannot increase above the prior round. A `lower_confidence` update reduces confidence by at least `0.05`, subject to the other caps.

The output records several forms of movement:

- `changed_mind`: the validated architecture changed.
- `ranking_changed`: the persona's matrix winner changed.
- `assumptions_changed`: candidate assumptions changed.
- `confidence_changed`: the bounded confidence changed.
- `material_update`: the architecture, ranking, assumptions, or confidence changed.

These fields make it possible to distinguish “nobody switched” from “the discussion had no influence.”

### Deterministic final decision summary

After the final round, the orchestrator computes `decision_summary` without another model call. Because open-ended proposals are not guaranteed to be scored by every persona, the preferred solution is selected from validated final positions. The most-supported solution wins; aggregate recorded utility breaks a vote tie. This prevents a proposal evaluated only by its advocate from winning solely because no other persona scored it.

The summary reports `consensus`, `plurality`, or `no_consensus`, plus supporting personas, dissenting positions, the strongest alternative, unresolved uncertainties, and switch conditions. The preferred candidate is therefore a transparent aggregation result rather than a hidden judge's recommendation.

### LLM-generated conclusion

After the last round, a model call generates the top-level `conclusion`. Its
prompt contains only final-round opinion records: agent identifier, canonical
architecture, position, confidence, decision-critical uncertainty, and opinion
text. It does not receive the objective, research dossier, earlier rounds, or
deterministic `decision_summary`. The generator distinguishes exact agreement
on canonical identifiers from broader semantic convergence and requires agent
claims to remain attributed rather than presenting them as independently
proven facts. A draft containing unsupported consensus or absolute certainty is
rewritten once; if the rewrite still overstates the records, conclusion
generation fails instead of publishing misleading text. Incremental snapshots
contain `"conclusion": null`; the completed snapshot contains the generated
plain-text conclusion.

## Research boundary

Research runs once before round one. The planner produces one bounded batch of broad,
goal-level comparison queries, retrieval executes that batch in parallel, and the results
are consolidated into a shared dossier. `DISCUSSION_MAX_INITIAL_QUERIES` is the total query
budget; solution-specific or debate-time queries are not appended to it.

While that work is running, the latest discussion snapshot includes a lightweight
`preparation` object. Its stages are `queued`, `room`, `planning`, `retrieving`,
`consolidating`, and `ready`. Retrieval checkpoints expose query status, source type,
and accepted/rejected excerpt counts so clients can render honest progress without
exposing model reasoning or embedding the full dossier in the public snapshot. These
frequent snapshots are live-status checkpoints rather than durable replay events, so
the discussion event sequence remains `started`, completed rounds, and the terminal
event.

The dossier remains fixed throughout the debate. Personas use `CRITICAL_UNCERTAINTY` and
evaluation `gaps` to report missing evidence, but those details do not trigger document
searches or dossier reconsolidation. This keeps the discussion focused on evaluating the
same evidence instead of recursively researching every claim it raises.

## Outputs and logs

### Incremental lifecycle

The producer now persists and optionally publishes the discussion throughout its
lifecycle instead of waiting for the final round. The ordering contract is:

```text
discussion.started
round.completed (round 1)
round.completed (round 2)
...
round.completed (final round)
discussion.completed
```

`discussion.failed` replaces normal completion after an unrecoverable producer
error. A round event is created only after every selected persona has either
produced an opinion or received a recorded error result, preserving the
synchronous round barrier.

The initial `discussion.started` snapshot contains the real participants and
graph before the potentially expensive shared-research phase begins. Every
`round.completed` event contains both the newly completed round and the complete
discussion snapshot through that round. The round object is produced by the
same serializer used for `rounds[]` in the public JSON.

Snapshots use schema version `1.0` and include `status`, progress counters, UTC
timestamps, and an optional failure record. `decision_summary` remains `null`
until all configured rounds finish. Consequently, analytics computed during a
run are provisional; influence in particular may require a later round.
The machine-readable contracts are checked in under
`schemas/discussion-snapshot-v1.schema.json` and
`schemas/discussion-event-v1.schema.json`.

### Persistence and delivery guarantees

The public file remains:

```text
outputs/discussions/<discussion_id>.json
```

After every lifecycle transition it is rewritten through a temporary file and
atomically replaced, so readers never observe half-written JSON. The producer
then appends the event to:

```text
logs/discussions/<discussion_id>/events.jsonl
```

Only after the snapshot and event are durable does the producer call
`DISCUSSION_EVENT_ENDPOINT`. Delivery is at least once. Each request includes a
deterministic `event_id` in both its JSON body and the `Idempotency-Key` header,
so consumers must accept an exact duplicate without appending the round twice.
A repeated event ID with different content or a different `snapshot_sha256`
should be rejected as a conflict.

Successful `2xx` responses are acknowledged. Timeouts, connection errors,
`408`, `429`, and selected `5xx` responses are retried with bounded exponential
backoff. Failed and skipped deliveries remain in the local journal. Unless
`DISCUSSION_EVENT_DELIVERY_REQUIRED=true`, consumer downtime does not stop the
discussion or lose producer data.

Every HTTP event uses this envelope:

```json
{
  "schema_version": "1.0",
  "event_id": "discussion-id:round:1",
  "event_type": "round.completed",
  "discussion_id": "discussion-id",
  "sequence": 1,
  "occurred_at": "2026-09-18T12:03:00Z",
  "snapshot_sha256": "...",
  "data": {
    "completed_round": 1,
    "round": {},
    "discussion": {}
  }
}
```

The optional bearer token is sent as `Authorization: Bearer <token>` and is
excluded from serialized configuration and public artifacts.

### Replaying undelivered events

After the consumer becomes available, replay unacknowledged events without
rerunning agents:

```powershell
python -m discussion --replay-events <discussion_id>
```

Replay reads `events.jsonl`, skips acknowledged event IDs, preserves original
sequence order, and updates:

```text
logs/discussions/<discussion_id>/delivery-state.json
```

### Concise result

The CLI writes one JSON result under:

```text
outputs/discussions/<discussion_id>.json
```

The concise result contains:

- `discussion_id`
- `objective`
- `core_topics`
- `participants`
- `graph`
- `configuration`
- `rounds`
- `opinions`
- `decision_summary`
- `logs`
- `conclusion`

Heavy research data is kept out of the concise result and referenced through log paths.

### Detailed discussion log directory

For a discussion ID such as `example-id`, the orchestrator writes:

```text
logs/discussions/example-id/
├── details.json
├── dossier.json
├── events.jsonl
├── delivery-state.json
├── rejected_evidence.jsonl
└── messages.jsonl
```

#### `details.json`

Contains the full serialized discussion state, including objective, configuration, graph, research plan, evidence pool, dossier, rejected evidence, and round data.

#### `dossier.json`

Contains the consolidated research dossier and its metadata. Use this file to inspect whether evidence was balanced across topics and whether requirements remained unresolved.

#### `rejected_evidence.jsonl`

Contains evidence candidates rejected during context resolution. These are audit records and are not valid evidence supplied to personas.

#### `messages.jsonl`

Contains routed discussion messages. Each line includes round number, sequence, sender, recipients, and content.

### Interpreting output quality

When reviewing a run, inspect in this order:

1. `configuration`: confirm the run used the intended limits and seed.
2. `research.plan`: check query diversity and broad coverage.
3. `research.dossier.coverage`: check for missing or partial topics.
4. `research.dossier.unresolved_requirements`: identify unsupported decisions.
5. `research.dossier.excerpts`: inspect source setup, evidence tier, limitations, and applicability.
6. `rounds[*].opinions`: check whether personas used the evidence independently.
7. `changed_mind` and `update_decision`: verify that switches have satisfied conditions.

High agreement is not by itself a quality signal. Agreement based on a dossier with missing alternatives or one-sided evidence is weak convergence.

## Data model

### `Objective`

The structured decision problem loaded from an objective file.

### `ResearchPlan`

```python
ResearchPlan(
    queries: list[str],
    rationale: str,
)
```

The query list is bounded and validated.

### `EvidenceResult`

```python
EvidenceResult(
    query: str,
    context: ContextResolutionResult,
    order: int,
    status: str,
)
```

### `Excerpt`

An excerpt is a typed, traceable evidence unit. Important fields include `text`, `source_name`, `evidence_tier`, `topic_id`, `research_query`, `final_query`, `applicability`, `limitations`, and `resolution_status`.

### `ResearchDossier`

The neutral consolidated evidence object. It contains:

- `objective_id`
- `excerpts`
- `contradictions`
- `per_topic_summaries`
- `coverage`
- `unresolved_requirements`
- `summary`

### `OpinionResult`

The parsed opinion and structured metadata for one persona in one round. It records both the current architecture and whether a substantive mind change was accepted.

Important decision fields include `candidate_assessments`, `persona_weights`, `switch_assessment`, `ranked_candidate`, `ranking_margin`, `ranking_changed`, `near_tie`, `changed_mind`, `assumptions_changed`, `confidence_changed`, and `material_update`.

### `CandidateAssessment`

The structured evaluation of one persona-proposed solution. It stores criterion scores, weighted utility, supporting and counter-evidence references, assumptions, evidence gaps, feasibility, and combination-specific evidence status.

### `DecisionSummary`

A deterministic, minority-preserving summary of the final round. It reports the final-position preferred solution, available aggregate utilities, consensus status, supporting personas, dissenting positions, strongest alternative, unresolved uncertainties, and each persona's switch condition. It does not erase dissent or ask another model to invent a new recommendation.

### `DiscussionRound`

Contains the round's opinion snapshot and routed messages.

### `DiscussionResult`

The top-level returned object. It exposes both concise serialization through `to_dict()` and heavy-data serialization through `logs_payload()`.

## Evidence discipline

The system is designed to reduce, not eliminate, common research failures.

### Source fact versus transferable concept

An excerpt should answer three separate questions:

1. What did the source actually report?
2. What general principle might be transferable?
3. What prevents direct transfer to this objective?

For example, hardware bandwidth in one benchmark is a source-specific observation. A possible transferable concept is that memory bandwidth can constrain throughput in a bandwidth-bound workload. It is not automatically evidence that the same hardware or architecture is best for a different model, scale, or workload.

### Hard constraints versus preferences

The objective parser stores constraints as text; it does not decide which constraints are absolute. Personas must explain whether a constraint is:

- A hard veto.
- A reliability floor.
- A preference that can be traded away.
- An assumption requiring validation.

If an objective says that long-context use is required, the discussion still needs to establish what “required” means operationally: context length, latency target, memory limit, retrieval quality, concurrency, or some combination.

### Claims that deserve scrutiny

Reviewers should challenge claims using words such as:

- only
- always
- guarantees
- impossible
- infeasible
- production-grade
- required
- objectively best

These may be justified, but they require comparable evidence and explicit assumptions.

## Testing

Run the Week 3 tests from the project root:

```bash
python -m pytest tests -q
```

Run the Week 2 tests:

```bash
python -m pytest week2/tests -q
```

Run the most relevant focused suites:

```bash
python -m pytest tests/test_decision.py tests/test_runner.py tests/test_research.py -q
python -m pytest week2/tests/test_planner.py week2/tests/test_evidence_pool.py week2/tests/test_consolidator.py -q
```

Prefer `python -m pytest` over the bare `pytest` executable so test collection uses the intended interpreter and includes the project root on `sys.path`.

### What the tests cover

Week 3 tests cover:

- Graph construction and connectivity.
- Deterministic participant selection.
- Shared research execution order.
- Evidence bounding.
- Dossier query-support behavior.
- Round barriers and opinion routing.
- Provider failures and null opinions.
- Architecture switch validation.
- Open solution discovery, persona-weighted utility, and deterministic decision synthesis.
- Dynamic challenger selection and near-tie update handling.
- Output shape and log path behavior.

Week 2 tests cover:

- Objective parsing.
- Planner schema and fallback behavior.
- Duplicate query rejection.
- Evidence pool statuses.
- Context resolution and web fallback.
- Consolidation, deduplication, evidence tiers, coverage, and transferability metadata.
- Memory behavior.

### Offline testing

Most unit tests monkeypatch LLM and retrieval boundaries. They should not require live provider credentials. A successful unit test run does not prove that the configured provider, RAG service, or model can execute a real discussion; use a configured integration run for that.

## Troubleshooting

### `Week 2 agents package was not found`

The configured Week 2 root must contain:

```text
<week2-root>/src/agents/
```

Check `DISCUSSION_WEEK2_ROOT`. Relative paths are resolved relative to the Week 3 project root.

### Random-seed configuration error

Set a JSON seed pool in `.env`:

```dotenv
DISCUSSION_RANDOM_SEEDS=[47,73,101,149]
```

For backward compatibility, a single seed is also accepted:

```dotenv
DISCUSSION_RANDOM_SEED=47
```

### `API_KEY` or provider configuration error

Week 2 loads provider settings from `week2/src/.env`, not the Week 3 `.env`. Confirm that the active provider configuration is present and that the model endpoint is reachable.

### RAG connection or retrieval failures

Check:

- `RETRIEVAL_API_BASE_URL`.
- Whether the local RAG service is running.
- Whether the configured source order is valid.
- Whether the service accepts the retrieval request format.
- The stderr lines beginning with `[resolver]`, `[grader]`, or `[evidence_pool]`.

The resolver can fall back from RAG to web search when configured, but web results do not automatically become strong evidence. Inspect source, applicability, and limitations in the dossier.

### Planner JSON failure

Messages such as:

```text
[planner] LLM planning failed (JSONDecodeError: ...). Falling back to deterministic 5-query plan.
```

mean the planner's provider response was not valid structured JSON or failed validation. The discussion can continue using the deterministic fallback. Inspect the generated plan and verify that it covers the objective rather than assuming the fallback is sufficient.

### Pydantic validation errors in `Excerpt`

Retrieval metadata is provider-dependent. Nullable fields should be normalized at the consolidation boundary. If a new metadata field is added, update both the normalization path and the consolidator regression tests.

### Personas all converge too quickly

Inspect:

- Dossier coverage by topic.
- Number of excerpts assigned to each architecture.
- Whether the planner produced distinct queries.
- Whether proposed solutions and their strongest alternatives received direct evidence.
- Whether the objective silently makes one option a hard veto.
- Whether personas copied neighbor reasoning instead of evaluating it.

Convergence caused by one-sided evidence is a research coverage problem, not proof of consensus.

### Personas disagree but nobody changes architecture

This can be a correct outcome: influence does not require every persona to switch. Inspect the structured state before deciding that influence failed:

1. Check `ranking_changed`, `assumptions_changed`, `confidence_changed`, and `material_update`. A persona may have moved without crossing the architecture threshold.
2. Check `switch_assessment.validation_errors`. A malformed test cannot authorize a switch.
3. Confirm that later-round `SWITCH_TEST.current` equals the previously validated architecture and `alternative` equals the newly proposed architecture.
4. Check whether the switch status remained `unresolved` or `contradicted`. Weakening the current option is insufficient; the alternative must satisfy the stated condition.
5. Compare `ranking_margin` with `DISCUSSION_UTILITY_CONSISTENCY_TOLERANCE`. A high tolerance allows close non-leading choices to remain stable.
6. Inspect `challenger_id` and `challenged_claim` to confirm that the persona received and addressed a decision-relevant disagreement.
7. Compare the evidence and counter-evidence references in each `CANDIDATE_EVAL`. An alternative with missing or one-sided evidence may never satisfy a responsible switch condition.
8. Check the objective criteria. Compound criteria can cause different persona priorities to map onto the same effective weights.

If matrices change but architecture choices remain close, try `DISCUSSION_UTILITY_CONSISTENCY_TOLERANCE=0.10`. Do not set it to zero merely to manufacture switching; first confirm that the objective has atomic criteria, an open solution space, and evidence-backed crossover conditions.

If neither rankings nor assumptions change, inspect the prompts and neighbor visibility. If rankings change but every switch is rejected, inspect the switch-test formatting and validation statuses.

### `ModuleNotFoundError: No module named 'discussion'` during pytest collection

Run tests from `services/discussion-engine/` with the virtual-environment interpreter:

```bash
cd services/discussion-engine
python -m pytest tests -q
```

The repository includes `pytest.ini` with `pythonpath = .`. If the error persists, verify that the command is running in the directory containing both `discussion\` and `pytest.ini`, and confirm the selected interpreter with:

```powershell
python -c "import sys; print(sys.executable)"
```

### A persona switches while its switch condition is unresolved

The orchestrator should preserve the previous architecture and record `unresolved_switch_condition`. Check the raw response markers and the parsed `OpinionResult`. If the switch is still accepted, add a regression test using the exact response shape and inspect the marker parser.

### Permission errors from pytest cache or temporary directories

The code can pass while pytest emits cache or temporary-directory permission warnings in managed environments. Use a writable cache directory when necessary:

```powershell
python -m pytest tests -q -o cache_dir=.pytest-cache-local
```

If temporary-directory creation itself fails, the issue is environmental rather than a test assertion failure.

## Development guide

### Adding a new persona

1. Add a persona mapping to `week2/src/configs/Personas.yaml`.
2. Give it explicit priorities and concerns.
3. Include instructions to evaluate evidence independently.
4. Run participant and runner tests.
5. Run a discussion with a fixed seed and inspect graph membership.

### Supporting a new architecture family

No code or objective registry change is required. A persona introduces the architecture through `ARCHITECTURE`, evaluates it with `CANDIDATE_EVAL`, and may compare or adopt it through `SWITCH_TEST`. Downstream consumers must therefore treat solution identifiers as open-ended strings rather than enums. Avoid treating a component as a new architecture unless the discussion actually compares it as a primary solution.

### Adding evidence metadata

When adding a new field:

1. Define its type and default in the Pydantic model.
2. Normalize `None` at the retrieval/consolidation boundary.
3. Preserve it in raw evidence collection.
4. Render it only where it helps reviewability.
5. Add a fixture with the field present and absent.
6. Confirm serialization in both concise and detailed outputs.

### Changing retrieval behavior

Keep retrieval decisions in `research.py`, `resolve_context.py`, and the Week 2 adapter boundary. Do not add retrieval calls inside persona graph nodes. The current contract performs one bounded research phase before Round 1 and keeps the resulting dossier fixed. Personas report missing information through evaluation `gaps` and `CRITICAL_UNCERTAINTY`; these fields are audit signals, not retrieval commands. Any future change to this boundary should update the prompts, logs, tests, and this README together.

### Changing result serialization

The concise result intentionally excludes the heavy evidence pool. If a consumer needs more information, add it to `logs_payload()` or a dedicated log file rather than inflating every result file.

### Preserving reproducibility

For comparable runs, keep these stable:

- Objective file.
- Persona YAML.
- Participant count.
- Round count.
- Random seed.
- Planner query limit.
- Evidence bounds.
- Provider model and retrieval data.

Provider responses and web/RAG indexes can still change even when local settings are identical.

## Known limitations

### Evidence quality is not proof of transferability

The dossier records applicability and limitations, but a persona or downstream consumer can still overgeneralize. Reviewers must inspect setup comparability.

### Open alternatives can be named inconsistently

Because there is no alias dictionary, mechanically different names produce different identifiers. For example, `memory_augmented_model` and `hierarchical_memory_model` are not automatically merged. Personas are shown previous proposals and should reuse an existing identifier when they mean the same solution. Reviewers may need to normalize synonymous proposals downstream.

### Planner fallback is general, not domain-optimal

If the planner LLM fails, the deterministic fallback covers objective topics and broad decision dimensions. It cannot know which specific literature or data source is most valuable for a specialized domain.

### Provider and source failures can produce partial dossiers

The system is designed to continue after individual retrieval failures. A completed discussion can therefore contain insufficient evidence. Always inspect status, coverage, rejected evidence, and unresolved requirements.

### There is no automatic truth arbiter

The orchestrator can enforce protocol rules, metadata validity, and switch conditions. It cannot determine whether a cited empirical claim is scientifically correct without a separate verification process.

### Final synthesis is deterministic rather than deliberative

The system computes a preferred solution from validated final positions, uses aggregate utilities only to break vote ties, and preserves minority positions. It does not run a separate judge or fact-checking model, so the summary inherits errors or unsupported scores from persona evaluations. Review the votes, available utilities, dissent, and unresolved uncertainties before treating it as an operational decision.


