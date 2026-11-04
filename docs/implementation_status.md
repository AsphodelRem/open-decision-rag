# Implementation status

This is an initial implementation, not completion of specification revision 3.

Implemented: immutable Pydantic source/tree/query contracts; snapshot validation
and revision hashing with linear tree validation; prepared-chunk passthrough; token-span and character
chunkers; a duck-typed RAGU splitter adapter; flat/metadata/KMeans/recursive
builders with deterministic node order; bounded summary repair; cached HTTP generation and batched embeddings; local
sentence-transformer embeddings; memory/SQLite cache; replay; in-process
singleflight; cancellation shielding; HTTP retries and shared semaphore;
memory/read-only SQLite tree access with batched reads and ancestor reuse;
JSON import/export; locked atomic publication with artifact checksums;
shared-context binary navigation, batching, parent weighting, choice reorder and
top1 promotion retaining all computed scores; source-aware context in reading
order; history-preserving answer generation; offline CLI and gold metrics with
explicit chunk/node namespaces and validated evaluation depths.

Actual builder, chunker and source identities are persisted in a build manifest.
Custom components can implement `semantic_identity()`; the fallback inspects
stored named constructor arguments, not arbitrary hidden state or source code.
Storage injection uses the structural `TreeStorage` protocol. Unknown facade
components and cache modes are rejected before model requests.

Structured retrieval traces retain stage names, numerical scores, selected/pruned
IDs and rendered context sizes. `ModelDiagnostics` separately collects run-scoped
HTTP cache keys/hits, shared waits, attempts, latency, usage and failure classes.

Also implemented: `DecisionModelSystemOne` for the Jev-compatible HTTP API of
`razorback16/openjev`, and generated JSON binary/choice judging in `LLMOpenAI`.
Both validate scores before caching and support search, answer generation and
replay through the shared execution layer. See [real model instructions](real_models.md).
These additions do not implement the direct `openjev/openjev` vLLM backend.

`DecisionModelFRIDA` supports native `noul` and `choice` through the FRIDA-Decisions
vLLM plugin's `/pooling` endpoint. It uses neutral option IDs, retains shared
candidate reading order, validates scores and reported state truncation before
caching, and supports persistent replay. The [full experiment script](frida_decisions.md)
covers clustered building, SQLite reopen, JSON round-trip, retrieval, generation,
optional gold evaluation and forced cache-only replay. Live weights remain untested.

`DecisionModelVLLM` now implements direct one-step DiffusionGemma binary and choice
readout through raw vLLM, with tokenizer-derived canvas plans, exact-token logprobs,
strict response validation, persistent wire caching and tokenizer-free replay.
It supports an optional input-plus-output token budget through the matching
tokenizer chat template and persisted prompt counts. Other providers currently
expose rendered character estimates and reject an explicit exact token budget.
Both real-model examples accept `DECISION_BACKEND=vllm`. See
[direct vLLM instructions](vllm_decisions.md). Real GPU/tokenizer compatibility
remains unverified; the original autoregressive OpenJev profile is separate.

Outstanding acceptance requirements:

- Verified direct OpenJev renderer, tokenizer mapping and exact-token readout,
  pinned weights revision, reference parity and real server fixtures.
- Paper profiles and stage plans, session notepad, heading selection, rescore,
  dense candidate union, custom selection policies and uniform model-context
  budgets across providers and summary/answer stages.
- Dataset-specific loaders for the four official benchmarks and conversational
  runner, aggregated latency/cost/billing reports and downstream answer metrics.
  Request diagnostics do not establish benchmark quality or billing accuracy.
- Markdown heading/fence parsing, tokenizer-name loading, source sections,
  arbitrary nested metadata and complete model response schemas.
- Cross-process cache leases, lease renewal and cache payload integrity envelope.
- Build checkpoints, artifact reuse/resume and vector manifests.
- Local-runner parity, prompt override API and opt-in live smoke infrastructure.

`resume=True` raises `NotImplementedError`; the default is `False`. Stage checkpoint
recovery is absent. Repeated HTTP operations can reuse request-cache entries.
`make_group_summary=False` is rejected; recursive `branching_factor` and physical
`tree_backend` are honored. Non-generic profiles
and session arguments fail explicitly rather than silently using generic search.
The CLI uses mock scoring only; it is not a deployment factory for live models.

## Validation

Offline integration tests exercise SQLite reopen and JSON round-trip, persistent
replay, concurrent singleflight, real sklearn KMeans/recursive construction,
source offsets and multi-gold metrics. See the final task report for actual
commands and results. No live embedding, generation, OpenJev, or GPU benchmark
was run. Offline KMeans uses deterministic mock embeddings and summaries.

Actual validation in this environment:

- `.venv/bin/flake8 --select=WPS,E999 --format=default odr tests examples`: exit 0.
- `PYTHONPATH=. python /tmp/decisionrag-run.py`: 83 tests passed. The sandbox
  runner adds a periodic event-loop wakeup for worker-thread completion.
- `mypy --config-file mypy.ini odr tests examples`: no issues in 78 source files
  (run using the locally available mypy environment).
- `PYTHONPATH=. python /tmp/decisionrag-run.py examples/offline.py`: exit 0.
- `setuptools.build_meta.build_wheel('/tmp/odr-wheel')`: wheel built successfully.
- Wheel extracted into an isolated temporary directory: public `decisionrag`
  import, source document and chunk creation succeeded.
- Runtime available for these checks: Python 3.10.19, httpx 0.28.1,
  Pydantic 2.14.2. Python 3.11+ installation/runtime verification: not_run.

Linter configuration was unchanged and no noqa suppressions were added.

See [review resolutions](review_2026-11-05.fixes.ru.md) for defect-to-test mappings
and contract migrations. CI now targets Python 3.10/3.12 tests and runs pinned WPS,
mypy, distribution building and installed-wheel API checks; remote CI was not run.
