# Single-Agent Tool-Orchestrated Mode

## Thesis

This repo should add a new execution mode where a single high-level agent uses
deterministic tools, typed artifacts, and explicit validators to solve the task.
It should not replace the existing structured pipeline outright.

The right inversion is:

- keep deterministic components as the product core
- expose those components as first-class tools
- let one planner agent decide when to call them
- preserve explicit artifacts, telemetry, and validators

The *wrong* inversion is:

- collapse Architect/Hunter/Synthesizer into one opaque prompt
- move control flow from code into model hidden state
- lose CDGs, typed handoffs, ghost validation, and compiler-grounded checks

## Motivation

The current system already states the correct architectural direction:

- move work out of prompts and into deterministic transforms
- simplify runtime paths
- benchmark against direct and lighter-weight baselines
- keep auditability, reuse, and correctness as the differentiator

The implementation still pays substantial prompt and orchestration overhead:

- many task-local prompts exist for ranking, reformulation, critique, repair,
  and failure handling
- prompt-key routing and provider selection add operational complexity
- some control flow is encoded in prompt loops instead of deterministic policies
- telemetry shows large variance in prompt dispatch counts across runs

This creates three concrete problems.

### 1. Too much prompt-shaped control flow

Several steps are useful as capabilities but do not need to be separate agent
identities. In practice, the system often wants:

- decompose this goal
- ground this leaf
- verify this candidate
- repair this artifact
- explain this failure

Those are better modeled as tool calls with typed inputs and outputs than as a
fixed chain of prompt wrappers.

### 2. Operational burden from per-prompt runtime decisions

The current routing model is powerful, but it also creates maintenance cost:

- prompt-specific providers
- prompt-specific models
- prompt-specific fallback behavior
- prompt-specific benchmarks

That complexity is justified only where benchmarked gains are real.

### 3. The current "rapid" direction is correct but incomplete

The repo already wants distinct execution modes. A single-agent mode is the most
natural realization of that idea for low-friction runs, but it needs stronger
tooling than the current thin rapid path. If it simply skips decomposition and
suppresses useful search behavior, it will be fast and weak instead of fast and
well-scaffolded.

## Recommendation

Add a new mode, tentatively `single_agent`, `tool_orchestrated`, or `planner`.

This mode should:

- accept the same high-level task input as the current orchestrator
- run one main agent loop
- allow that agent to call deterministic tools with strict schemas
- persist the same major artifacts when relevant
- reuse the same validators, compiler checks, and telemetry sinks

This mode should not:

- remove `structured` or `verified`
- remove CDG export as an artifact
- remove typed handoff validation
- replace compiler or ghost checks with model judgment

The existing pipeline should remain the strongest correctness path. The new mode
should become the simplest path with disciplined tooling.

## Design Principles

### 1. Preserve explicit artifacts

The planner may choose the path, but its work products must still be explicit:

- `GoalAnalysis`
- `CDGExport` or a lighter `PlanDraft`
- `PDGNode` list
- candidate sets and verification attempts
- assembled skeletons
- repair actions
- final verification certificate or failure report

If a user cannot inspect what happened after the run, the mode is not aligned
with the product thesis.

### 2. Keep deterministic ownership boundaries

The model should not own:

- type synthesis rules
- edge repair
- graph invariant enforcement
- candidate verification
- compilation
- ghost simulation
- telemetry aggregation

The model may own:

- deciding which tool to call next
- deciding whether more decomposition is needed
- deciding whether retrieval is worth attempting
- summarizing failure causes
- prioritizing among valid next actions

### 3. Replace agent boundaries with tool boundaries

Today the repo has conceptual boundaries like Architect, Hunter, and
Synthesizer. Those should become tool namespaces rather than mandatory runtime
actors.

Example tool surface:

- `architect.select_strategy(goal) -> StrategyDecision`
- `architect.decompose(node, context) -> DecompositionDraft`
- `architect.validate(cdg) -> ValidationReport`
- `hunter.search(pdg_node, policy) -> CandidateSet`
- `hunter.rank(candidate_set, policy) -> RankedCandidates`
- `hunter.verify(pdg_node, candidates) -> VerificationBatch`
- `synth.assemble(cdg, matches) -> SkeletonFile`
- `synth.ghost_check(cdg, matches) -> GhostReport`
- `synth.compile(artifact) -> CompileResult`
- `synth.repair(artifact, errors) -> PatchProposal`
- `catalog.lookup(query) -> PrimitiveMatches`
- `context.search(namespace, query) -> ContextBlock`

Each tool should be callable outside the single-agent mode as well.

### 4. Policies should be deterministic where possible

The planner should choose between deterministic policies, not invent them each
time. For example:

- retrieval policy: `skip`, `light`, `aggressive`
- decomposition policy: `none`, `shallow`, `full`
- repair policy: `none`, `bounded`, `until_verified`
- context policy: `off`, `recent_only`, `full`

This limits drift while preserving flexibility.

## Proposed Architecture

### Layer 1: Deterministic tool API

Refactor current capabilities behind typed service functions with minimal prompt
awareness.

Requirements:

- stable input/output models
- no CLI parsing inside tool code
- reusable by CLI, benchmarks, and planner
- telemetry emitted at tool boundaries
- deterministic error types

### Layer 2: Planner runtime

Add a planner loop that:

1. reads the goal
2. selects an execution policy
3. calls tools
4. inspects tool outputs
5. decides next action
6. stops on verified success, bounded failure, or budget exhaustion

This is where a single LLM session can be valuable. The model sees the whole
task state and chooses among a small, explicit action space.

### Layer 3: Existing explicit modes remain

The current flows become named policy bundles:

- `rapid`: minimal tool use, low overhead
- `single_agent`: one planner agent plus tools
- `structured`: explicit graph-first pipeline
- `verified`: full graph validation and traceability

This matches the repo's stated desire for execution modes without forcing one
control pattern onto every workload.

## What To Refactor First

Start with the highest prompt-to-value ratio items: the steps that are useful
but mostly mechanical.

### Phase 1: Extract reusable tool functions

Goal: make current components callable as deterministic/typed tools without
changing behavior.

Tasks:

- extract pure service entrypoints from Architect, Hunter, Synthesizer, and
  Orchestrator code
- normalize inputs and outputs around Pydantic/dataclass models
- make tool calls emit consistent telemetry
- move prompt templates behind tool implementations instead of exposing prompt
  identity at the top level

Deliverable:

- a `services/` or `runtime/tools/` layer with no mode-specific branching

### Phase 2: Collapse the most mechanical prompts

Goal: reduce prompt count before adding the planner.

Priority candidates:

- architect strategy selection
- hunter candidate ranking
- hunter query reformulation where deterministic heuristics are good enough
- orchestrator failure splitting if deterministic split templates or catalog
  hints can cover common cases

Principle:

- use deterministic or embedding-based replacements first
- keep LLM fallback only where ambiguity is real

Deliverable:

- lower prompt dispatch count in current modes
- cleaner tool surface for the planner mode

### Phase 3: Add the single-agent planner mode

Goal: run one high-level agent over the tool API.

Planner loop responsibilities:

- decide whether to decompose or act directly
- choose retrieval intensity
- decide when to stop iterating
- decide when to escalate from shallow to full structure
- decide whether repair is worth attempting

Guardrails:

- bounded step budget
- bounded retry budget per artifact
- mandatory validation after each state-changing tool call
- mandatory persistence of intermediate artifacts

Deliverable:

- new mode in CLI and benchmarks

### Phase 4: Benchmark and prune

Goal: prove which runtime paths deserve to survive.

Measure:

- total prompt calls
- time to first verified result
- total latency
- success rate
- structural correctness
- run-to-run variance
- artifact completeness
- human intervention needed

Outcomes:

- keep prompt-key routing only where it wins materially
- remove redundant prompt stages
- simplify provider/runtime matrix

## Concrete Tool Mapping From Current Code

The current code already contains the right building blocks.

Convert these into planner-callable tools:

- Architect strategy selection and decomposition
- Architect critique and handoff validation
- Hunter search, ranking, verification, and failure analysis
- Orchestrator refinement logic
- Synthesizer assembly, ghost simulation, compile, and repair
- shared-context search and write
- catalog and skill index lookup

The planner should call those tools. It should not reimplement their internals.

## Data Model Changes

Add a planner-state model that is narrower than the current graph states but
rich enough for tool orchestration.

Suggested fields:

- `goal`
- `mode`
- `policy`
- `artifacts`
- `current_focus`
- `open_failures`
- `attempt_history`
- `tool_trace`
- `verification_status`
- `budget`

Important rule:

the planner state is not the source of truth for domain artifacts. It points to
explicit artifacts produced by tools.

## Telemetry Requirements

The new mode should improve observability, not reduce it.

Add:

- tool dispatch counts
- tool latency by name
- planner step count
- planner decision trace
- escalation events, such as `direct -> decompose` or `light -> aggressive retrieval`
- artifact creation and mutation counts
- termination reason

Keep:

- prompt telemetry inside tools where prompts still exist
- verification and compile telemetry
- unresolved-leaf and failure metrics

## Risks

### Risk 1: Hidden-state orchestration

If the planner is allowed to hold too much implicit state, debugging gets worse.

Mitigation:

- require structured planner action outputs
- persist intermediate artifacts
- log rationale in short, typed fields

### Risk 2: Losing the correctness identity of the repo

If single-agent mode becomes the default mental model, the project can drift
toward a general-purpose coding agent.

Mitigation:

- keep `verified` as the flagship mode for correctness-critical work
- document that the core value remains deterministic validation and reuse

### Risk 3: Tool API becomes a thin wrapper over old prompt spaghetti

If tools just forward to old prompts with no cleanup, the new mode adds another
layer without reducing complexity.

Mitigation:

- reduce prompt-shaped substeps before or during tool extraction
- delete old routing paths once benchmark evidence supports it

### Risk 4: Planner mode underperforms because tools are too coarse

A single agent only helps if tool granularity is usable. Too-coarse tools force
the agent to guess; too-fine tools create chatter.

Mitigation:

- design tools around meaningful decision points
- benchmark tool granularity explicitly

## Decision Rules

Use the single-agent mode when:

- the task is cross-stage but not maximally correctness-critical
- you want lower orchestration overhead
- you want one coherent planner to adaptively choose the path
- the domain benefits from existing deterministic tools and validators

Use `structured` or `verified` when:

- artifact traceability is primary
- decomposition quality must be explicit and inspectable
- failure analysis needs strong stage separation
- correctness pressure is high enough that explicit control flow is preferable

## Recommended First Milestone

The first milestone should be small and falsifiable.

Build:

- extracted tool APIs for current Architect and Hunter flows
- deterministic replacements for the most mechanical prompt steps
- a planner that can choose between:
  - direct grounding
  - shallow decomposition plus grounding
  - full decomposition escalation on failure

Success criteria:

- prompt dispatch count drops materially versus current structured runs
- success rate is not worse than current rapid mode
- artifact trace is still inspectable
- telemetry clearly shows why the planner chose its path

## Bottom Line

Yes, the repo should move toward "deterministic tools used by a single agent"
as one of its main runtime modes.

No, it should not invert all the way into "one agent replaces the pipeline."

The durable advantage of this repo is not that it has many named agents. It is
that it has deterministic structure, reusable artifacts, and compiler- or
simulation-grounded checks. The refactor should preserve that substrate and
simplify the orchestration layered on top of it.
