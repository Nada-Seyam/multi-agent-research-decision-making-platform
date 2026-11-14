# Discussion Analytics & Intelligence Layer

## Streaming consumer API

The consumer accepts producer lifecycle events and emits JSON analysis after
each completed round. Influence is calculated only after the complete
discussion is received.

Start it by itself with:

```powershell
python consumer_api.py
```

Endpoints:

```text
GET  http://127.0.0.1:8002/health
POST http://127.0.0.1:8002/api/v1/discussion-events
GET  http://127.0.0.1:8002/api/v1/discussions/{discussion_id}/analysis
GET  http://127.0.0.1:8002/api/v1/discussions/{discussion_id}/analysis/rounds/{round_number}
```

The root-level `python main.py` command starts this API and the producer
together. Streaming analysis writes JSON artifacts only.

Accepted events are journaled before background analysis begins. The response
reports `processing_status: queued`; processing failures remain retryable and
are exposed in the artifact as `status: analysis_failed` (or
`completed_with_errors`) plus a `processing_failures` record. Reposting the
same event idempotency key, or restarting the service, retries an unprocessed
event.

A reusable analytics engine turns a completed Week 3 multi-agent discussion
into measurable insight: opinion change, agreement/disagreement, influence,
and sentiment.

* **Opinion Change** (`src/analytics/opinion_change.py`)
* **Agreement / Disagreement** (`src/analytics/agreement.py`)
* **Sentiment** (`src/analytics/sentiment.py`)
* **Influence** (`src/analytics/influence.py`)
* **Unified Engine** (`src/analytics/engine.py`) — single entry point returning all four analytics categories

---

## Scope

The implementation accepts the versioned Week 3 discussion snapshot schema.
Tests use self-contained input dictionaries so a separate fixture dataset is
not required to validate the package.

---

# Setup

```bash
pip install -r requirements.txt
```

---

# Testing

```bash
python -m pytest -q
```

The test suite covers normal behavior and edge cases such as:

* missing confidence
* missing position
* duplicate opinions
* missing `decision_summary`
* empty messages
* insufficient data for agreement
* insufficient data for influence

---

# Opinion Change

## Purpose

Calculate a numeric representation of an agent's opinion across rounds and identify changes between rounds.

## Input

For each agent and round:

* committed `architecture` / `position`
* discussion-level anchor candidate

## Candidate Selection

The implementation uses `architecture`, falling back to `position`, as the
agent's committed candidate. In the observed Week 3 output these fields agree.

`ranked_candidate` has a different meaning: it is the weighted candidate
matrix leader. It may differ legitimately from the committed architecture
when the lead is a defensible near tie or when the agent's switch-validation
contract has not authorized a mind change. Opinion Change therefore does not
use it to calculate stance or change.

Structured `candidate_assessments` are validated when available. Utilities
are recomputed from numeric criterion scores and normalized persona weights.
Zero is a valid score only when backed by criterion values; zero placeholders
with no criterion scores are treated as unavailable. Free-text
`criteria_scores` is not parsed as a numeric source.

Candidate identifiers are normalized using:

```text
utils.normalize_candidate_id
```

---

## Stance Formula

```text
stance = +1.0    if committed candidate == anchor
stance = -1.0    if committed candidate != anchor
stance = null    if committed candidate or anchor is unavailable
```

The scalar is a deliberately narrow **committed anchor-alignment** measure.
It is not a complete embedding of an unordered multi-candidate opinion space.
Candidate identity is retained so changes between two non-anchor candidates
remain visible even though both candidates have stance `-1`.

Interpretation:

* `+1.0` = the agent is committed to the anchor candidate.
* `-1.0` = the agent is committed to another candidate.
* `0.0` change = anchor alignment did not change; another kind of opinion
  change may still have occurred.

Confidence, matrix preference, and update metadata are not calculated as part
of Opinion Change because no downstream Week 4 metric requires them.

A negative stance does **not** identify which alternative candidate the agent supports.

---

## Anchor Candidate

The anchor is selected in this order:

1. `decision_summary.preferred_candidate`
2. Majority candidate from the final round
3. `None` if neither is available

When no anchor can be determined, stance values are `None` rather than fabricated as zero.

---

## Opinion Change

For immediately consecutive discussion rounds:

```text
change(round) =
    stance(round) - stance(previous discussion round)
```

The first round has no change value. A switch to the anchor produces `+2`, a
switch away produces `-2`, and unchanged anchor alignment produces `0`.

Missing snapshots are emitted explicitly with `stance=None` and `change=None`;
the implementation does not skip the agent/round cell or compare across a gap.

The generated report contains only:

* round
* committed candidate
* stance
* stance change
* candidate changed (`yes`, `no`, or `n/a` when no comparison is available)

---

# Agreement / Disagreement

## Purpose

Measure how often agents select the same or different committed candidates
within a round.

## Formula

For every pair of agents with usable committed candidates:

```text
pair agrees    when candidate_i == candidate_j
pair disagrees when candidate_i != candidate_j
```

Then:

```text
agreement    = agreeing_pairs / total_pairs
disagreement = disagreeing_pairs / total_pairs
```

The score is in:

```text
[0.0, 1.0]
```

Interpretation:

* agreement `1.0` = every usable pair selected the same candidate
* agreement `0.0` = no usable pair selected the same candidate
* disagreement is always `1 - agreement`

The calculation uses normalized committed candidate identifiers and is
independent of the discussion's stance anchor.

At least two agents with valid committed candidates are required.

If insufficient data exists, the result is `None` with an explanatory reason.

The metric should still be interpreted separately from Week 3's
`consensus_status`: pairwise candidate agreement describes the distribution
of committed choices, while the runner's consensus status may apply additional
decision rules.

---

# Sentiment

## Purpose

Measure how each message evaluates each recipient's current committed
candidate. In this technical discussion domain, `positive`, `neutral`, and
`negative` describe candidate support or criticism, not the sender's emotional
tone.

## Method

Each directed message-recipient pair is evaluated separately:

```text
sender message → recipient → recipient candidate in the message round
```

This allows one message to be positive toward one recipient's candidate and
negative toward another recipient's candidate.

Evidence is evaluated in this order:

1. The sender's structured weighted candidate matrix, when the target
   candidate has a usable assessment.
2. The sender's explicit committed candidate and any consecutive-round switch
   away from the target.
3. Candidate-focused technical statements in the message.

General emotional wording outside candidate-focused text is ignored. A
different committed candidate is not automatically criticism.

For a usable candidate matrix, the target's utility is compared with the
sender's mean assessed utility:

```text
matrix score = (target utility - mean candidate utility) / 2.5
```

When only one candidate utility exists, `2.5` is used as the neutral point on
the declared `0`-`5` utility scale. Matrix evidence has the greatest weight.
Shared commitment, switching away, and candidate-focused text provide
additional deterministic evidence. The weighted result is clipped to the
required sentiment range.

The score is in:

```text
[-1.0, 1.0]
```

Labels use candidate-evaluation thresholds:

```text
score > +0.15          positive
-0.15 <= score <= 0.15 neutral
score < -0.15          negative
```

Here:

* `positive` means supportive or favorable toward the target candidate
* `negative` means critical or unfavorable toward the target candidate
* `neutral` means factual, balanced, or mixed about the target candidate

Candidate-focused text is classified with
`cross-encoder/nli-deberta-v3-base`. For each target excerpt, the model tests
two candidate-specific hypotheses:

```text
The author supports choosing {candidate}.
The author opposes choosing {candidate}.

text score = P(entail support) - P(entail oppose)
```

This lets the classifier interpret technical claims, negation, and contrasts
without assigning fixed sentiment to words such as `risk`, `complex`, or
`feasible`. Candidate names and acronyms declared in discussion text, such as
`Bitemporal Hybrid Graph-Vector Memory (BHGV)`, are linked automatically and
provided to the model as a candidate glossary. Structured candidate evidence
still has greater weight than NLI text evidence.

This metric does not perform graph propagation. It describes the evaluation in
the message that reaches a recipient; Influence separately traces whether ideas
move through neighbors and are associated with later candidate movement.

---

## Missing Evaluations

Every message receives one `SentimentRecord`. It contains one target evaluation
per recipient. A valid message that has no matrix, commitment, switch, alias,
or text reference related to a recipient's candidate is classified as neutral:

```text
score = 0.0
label = neutral
evidence_source = no_candidate_reference
```

Missing or empty messages, messages without recipients, missing recipient
candidates, and missing sender opinion snapshots still produce an explanatory
reason instead of a fabricated score:

```text
score = None
label = None
```

These structurally missing cases are not treated as neutral because:

```text
missing evaluation inputs ≠ neutral candidate evaluation
```

---

## Message References

Week 3 messages do not contain unique message IDs.

Therefore, the analytics layer creates a deterministic reference identifier:

```text
r{round}_s{sequence}_{sender}
```

This is only an analytics reference and does not modify the original Week 3 data.

---

# Influence

## Purpose

Estimate whether stronger textual uptake of an agent's argument, either
directly or through a directed intermediary, is associated with another agent
subsequently moving toward the source agent's exact committed candidate.

This is an **association metric, not a causal claim**.

## Input

The metric uses:

* committed candidates from Opinion Change
* directed message content
* each downstream recipient's `challenged_claim` and `update_reason`
* optional attribution support through `challenger_id`
* strictly consecutive round transitions

## Directed temporal paths

The metric evaluates two path types:

```text
direct:   i → j       message in r, response in r+1
indirect: i → k → j   messages in r and r+1, response in r+2
```

Only simple paths are used, so a path cannot revisit an agent and create
self-influence. Missing rounds are never bridged. At most one observation is
retained for each source, target, starting round, and ending round; when
multiple intermediary paths exist, the strongest textually supported path is
used.

## Textual uptake

For a direct observation:

```text
x = TF-IDF cosine(message(i → j, r), response(j, r+1))
```

For an indirect observation:

```text
x = geometric mean(
        TF-IDF cosine(message(i → k, r), message(k → j, r+1)),
        TF-IDF cosine(message(k → j, r+1), response(j, r+2))
    )
```

The source-to-intermediary message must have at least `0.05` cosine relevance
to the intermediary's later message. This prevents a graph path alone from
being treated as propagated influence. TF-IDF vectors are length-normalized;
total message word count is not an influence signal.

## Candidate movement

Candidate distance is:

```text
distance(a, b) = 0 when a == b, otherwise 1
```

The source candidate is frozen at the path's starting round. Movement is
measured around the final message in the path:

```text
y =
distance(candidate(j,before final message), candidate(i,start))
- distance(candidate(j,after final message), candidate(i,start))
```

A positive `y` means the recipient switched to the sender's exact candidate;
a negative value means the recipient switched away. Moving between two other
candidates produces zero.

The raw association is the Spearman correlation between textual uptake (`x`)
and exact candidate movement (`y`). To avoid treating indirect evidence as
equivalent to direct evidence, the correlation is multiplied by average path
support, where direct observations have weight `1.0` and two-hop observations
have weight `0.5`.

`challenger_id` is supporting evidence rather than an exclusion rule. A
matching direct challenger, or a fully supported challenger chain, is marked
explicit. Missing or different challenger declarations are retained as
unconfirmed observations because `challenger_id` identifies the strongest
challenge, not necessarily the only influence.

The result is in:

```text
[-1.0, 1.0]
```

Interpretation:

* positive = greater textual uptake is associated with movement toward the sender's candidate
* negative = greater textual uptake is associated with movement away
* near zero = no clear monotonic association

---

## Insufficient Data

Influence returns `None` with an explanatory reason when:

* an agent sent no usable directed messages
* fewer than three usable direct/indirect observations exist
* textual relevance has no variation
* candidate movement has no variation
* the correlation is otherwise undefined

A defined score is labelled by evidence reliability:

```text
low       = the minimum evidence needed for a defined association
moderate  = >= 6 observations, >= 2 movement events, >= 50% opportunity coverage
high      = >= 12 observations, >= 4 movement events, >= 75% opportunity coverage
```

Opportunity coverage is:

```text
usable direct and indirect observations
----------------------------------------
temporally reachable graph opportunities
```

No artificial score is generated for insufficient data.

---

# Unified Analytics Engine

`src/analytics/engine.py` provides the single entry point:

```python
analyze_discussion(discussion) -> AnalyticsResult
```

The engine executes the four analytics modules in dependency order.

The Opinion Change result is calculated once and then reused by:

* Agreement, which compares committed candidates directly
* Influence, which compares exact committed-candidate transitions

This avoids recomputing the same underlying data independently for each metric.

The engine returns one structured result containing all four analytics categories.

---

# Data Validation

`data_loader.load_discussion()` distinguishes between fatal structural problems and recoverable data-quality problems.

## Fatal problems

The loader raises `DiscussionValidationError` when:

* the input is not a JSON object
* `discussion_id` is missing
* `rounds` is missing or empty
* two round entries use the same round number

These prevent meaningful analytics from being performed.

## Recoverable problems

The loader records issues in:

```text
Discussion.issues
```

Examples include:

* malformed opinions
* malformed messages
* invalid agent IDs
* duplicate opinions
* missing candidate
* missing/non-numeric confidence
* confidence outside `[0,1]`
* empty message content
* duplicate messages
* unknown message sender
* missing preferred candidate
* fallback anchor selection

Recoverable issues are logged rather than silently ignored.

---

# Repository Structure

```text
.
├── README.md
├── requirements.txt
├── conftest.py
│
├── src/
│   └── analytics/
│       ├── __init__.py
│       ├── models.py
│       ├── utils.py
│       ├── data_loader.py
│       ├── opinion_change.py
│       ├── agreement.py
│       ├── sentiment.py
│       ├── influence.py
│       └── engine.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_opinion_change.py
│   ├── test_agreement.py
│   ├── test_sentiment.py
│   ├── test_influence.py
│   └── test_streaming_api.py
```

---

# Final Validation

Run the complete test suite:

```bash
python -m pytest -q
```

The suite validates the four analytics modules, data loading, and streaming API
without requiring external fixture files.
