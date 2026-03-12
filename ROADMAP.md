# ROADMAP

## Positioning

This package should not try to be the fastest possible way to create any algorithm.

Strong general LLMs can already do well on many algorithm-design tasks when:
- the specification is clear
- evaluation is cheap
- the human is comfortable iterating directly on generated code

The niche for this package is **verified retrieval-augmented composition**: decompose a goal into typed sub-problems, match those sub-problems against a catalog of real library functions, verify that the matches actually work, and assemble the result. The value is narrower and more defensible than general code generation:

- **Verification-first output.** The system checks its own work — every candidate match is verified (importable, callable, type-compatible, arity-correct) before it is accepted. A raw LLM can hallucinate a function name; this system proves the function exists and fits.
- **Deterministic-first, LLM-fallback.** Every prompt call that can be replaced by a regex, AST walk, embedding lookup, or type check is a permanent cost, latency, and reliability win. LLMs handle conceptual decomposition and ambiguous cases; deterministic tools handle everything with a known structure. This is the core architectural bet, not just an optimization.
- **Compounding reuse.** Every successful match enriches the catalog. Every solved decomposition pattern can be reused. Every failure analysis refines future routing. The system gets cheaper and faster over time on the same domain — a flywheel, not a one-shot tool.
- **Typed, auditable decomposition.** Goals are split into a typed dependency graph with explicit interfaces between sub-problems. The decomposition is an inspectable artifact, not a hidden prompt chain.
- **Graduated execution tiers.** Not every task needs full orchestration. The system supports distinct modes — from lightweight single-agent (deterministic planner with partial acceptance and selective re-decomposition) through full verified orchestration — so the overhead scales with the task's correctness requirements.

The package has two failure modes to guard against:
1. If it expands toward a general-purpose LLM coding platform, the complexity will outrun the value.
2. If new capabilities add prompt keys instead of deterministic tools, cost and latency grow linearly with scope instead of amortizing.

## Where This Package Is Better Than Direct LLM Coding

This package is ideal when one or more of the following are true:

- correctness matters beyond passing a narrow benchmark
- the domain is expensive to debug
- decomposition quality matters as much as final code
- typed interfaces and intermediate invariants matter
- reusable algorithmic components are valuable after the first run
- multiple agents or repeated prompts would otherwise waste time reinventing the same structure
- auditability and provenance matter

Typical examples:
- scientific computing
- biomedical signal processing
- finance or risk pipelines
- compilers and formal methods
- safety-sensitive control or verification logic
- library-grounding tasks where retrieval and type compatibility matter

## Where Direct LLM Coding Is Better

Direct LLM coding usually wins when:

- the algorithm is small or medium in scope
- the task has a strong gold-standard eval loop
- implementation speed matters more than structural guarantees
- the artifact is likely one-off
- the user can cheaply test and patch failures manually

Typical examples:
- scripts
- utilities
- quick prototypes
- standard textbook algorithms with straightforward tests

## Core Product Thesis

The value of this package is not “agents can write code.”

The value is:
- constrain early
- normalize structure deterministically
- reuse prior verified work
- validate continuously
- fail explicitly and observably

That is the product thesis the roadmap should reinforce.

## Strategic Goals

### 1. Narrow The Mission

Optimize for:
- verified decomposition
- primitive and atom reuse
- typed composition
- structural correctness
- traceability

Do not optimize for:
- replacing direct LLM coding on simple tasks
- supporting every possible provider/runtime path
- maximizing agent autonomy for its own sake

### 2. Introduce Execution Modes

The package should support distinct modes so the deterministic overhead is used only when needed.

Recommended modes:

- `rapid`
  - direct LLM + eval loop
  - minimal decomposition
  - minimal retrieval
  - optimized for speed

- `structured`
  - decomposition
  - catalog assistance
  - deterministic normalization and typed checks
  - selective retrieval

- `verified`
  - full graph validation
  - shared context
  - richer telemetry
  - strict handoff and artifact traceability

This is likely the highest-leverage product simplification.

### 3. Keep Moving Work Out Of Prompts

Prompts should not be responsible for:
- inventing types
- inventing control flow wrappers
- repairing graph structure
- deciding low-level atomic bindings when deterministic rules can do it better

Prompts should focus on:
- conceptual decomposition
- primitive hints
- prioritization
- short structured analyses

Deterministic systems should own:
- type synthesis
- edge repair
- wrapper elimination
- concept normalization
- primitive resolution
- graph invariants

## Technical Priorities

### Priority A. Benchmark Against Direct Baselines

The package needs proof that it improves outcomes in its niche.

Build benchmark tracks for:
- direct LLM baseline
- `rapid` mode
- `structured` mode
- `verified` mode

Measure:
- time to first correct result
- total prompt calls
- total latency
- human intervention required
- structural correctness
- run-to-run stability
- reuse dividend from prior catalog/context

This should become a standing evaluation artifact, not a one-off exercise.

### Priority B. Make Retrieval Conditional

Retrieval should not dominate all flows.

Use retrieval aggressively when:
- catalog coverage is strong
- source-derived primitives exist
- grounding to libraries is likely to help

Use retrieval lightly or skip it when:
- the task is clearly greenfield
- primitive coverage is weak
- direct synthesis is cheaper than search

The package should make this decision explicitly from confidence signals.

### Priority C. Improve Reuse Dividend

The architecture is justified only if prior work compounds.

Invest in:
- source-to-catalog extraction
- alias and concept normalization
- template reuse for successful decompositions
- shared-context retrieval that is actually consumed
- persistent memory of common failure patterns
- richer use of `ageoa` and other atom sources

The package should get better every time it solves something.

### Priority D. Simplify Runtime Paths

Provider/shim complexity has become a significant operational burden.

Reduce to a small stable set:
- one strong local path
- one strong remote path
- prompt-key routing only where benchmarked gains justify it

Avoid multiplying maintenance burden with many partially overlapping execution paths.

### Priority E. Deepen Observability

Observability is part of the product.

Continue improving:
- per-stage progress
- per-prompt latency
- dispatch counts
- retry counts
- blocked-node reasons
- unresolved leaf counts
- deterministic rewrite actions
- retrieval hit quality
- shared-context reuse statistics

When the system is slow or wrong, the reason should be visible quickly.

## Recommended Near-Term Roadmap

### Phase 1. Product Simplification

- add `rapid`, `structured`, and `verified` modes
- define which subsystems are enabled in each mode
- make retrieval and shared context conditional by mode
- reduce default execution to the minimum needed for the selected mode

### Phase 2. Benchmark Discipline

- formalize cross-domain prompt-key benchmarks
- add direct LLM baseline comparisons
- add a small suite of end-to-end algorithm tasks across domains
- start tracking benchmark artifacts in CI or release validation

### Phase 3. Deterministic Assistive Layer

- continue concept normalization improvements
- expand wrapper elimination and routing collapse
- improve typed IO synthesis
- improve primitive-binding confidence scoring
- harden graph invariants and validation failures

### Phase 4. Reuse And Catalog Growth

- strengthen source-derived primitive extraction
- better align catalog and atom registries
- store and reuse successful decompositions
- make context reuse measurable and visible

### Phase 5. Runtime Hardening

- reduce provider/shim sprawl
- keep only benchmark-justified routing overrides
- improve failure reporting for live providers
- make long-running prompt behavior more predictable

## Concrete Recommendations

1. Stop treating every task as a full-stack orchestration problem.
2. Make deterministic guarantees opt-in by mode rather than always-on.
3. Treat decomposition as a constrained planning problem, not a code-generation problem.
4. Make retrieval conditional on catalog confidence.
5. Keep investing in deterministic transforms that remove low-value LLM work.
6. Benchmark the package against direct LLM baselines continuously.
7. Keep runtime/provider complexity on a tight budget.
8. Expand observability until failure causes are obvious.
9. Maximize reuse from prior solved tasks, primitives, and source repos.
10. Optimize for auditability and structural correctness, not just end artifact generation.

## Success Criteria

This package is succeeding if:

- it is clearly better than direct LLM coding in the high-correctness niche
- reuse from prior work reduces future search and prompt cost
- failures are explicit and diagnosable
- decomposition quality is stable across runs
- provider/runtime issues are a small part of engineering effort
- users can choose lighter or stricter execution modes depending on task needs

This package is failing if:

- it remains slower than direct LLM coding without delivering stronger guarantees
- infrastructure complexity keeps growing without measured benchmark gains
- retrieval and orchestration run even when they do not help
- runtime/shim engineering dominates product value
- prior solved work does not materially improve future performance

## Immediate Next Steps

1. Implement `rapid`, `structured`, and `verified` modes.
2. Add direct-baseline comparisons to the benchmark harness.
3. Make retrieval conditional on catalog confidence and mode.
4. Expand the benchmark suite from prompt-key tests to small full-flow task comparisons.
5. Continue reducing prompt responsibility in favor of deterministic transforms.
