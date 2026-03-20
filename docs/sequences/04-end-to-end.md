# End-to-End: Creating an Algorithm from Scratch

High-level view of the full pipeline from a user's goal to a verified,
exportable artifact.

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant Architect
    participant Critic
    participant Catalog
    participant Index as FAISS Index
    participant Hunter
    participant Judge
    participant GhostSim as Ghost Simulator
    participant Assembler
    participant RepairAgent as Repair Agent
    participant Extractor

    Note over User,Extractor: Prerequisites
    User->>Index: sciona index build --prover lean4
    Note over Index: Extract declarations from Mathlib/Coq/Python<br/>Embed with UniXcoder, store in FAISS

    Note over User,Extractor: Pipeline Start
    User->>Orchestrator: sciona run "Implement FFT-based spectral analysis"

    rect rgb(59, 130, 246, 0.08)
        Note over Orchestrator,Catalog: Round 1: Decomposition
        Orchestrator->>Architect: decompose(goal)
        Architect->>Architect: Select paradigm via LLM
        Architect->>Architect: Instantiate skeleton template

        loop For each non-atomic node
            Architect->>Catalog: Find matching primitives
            Architect->>Architect: LLM decomposes node into sub-nodes
            Architect->>Critic: Validate decomposition
            alt Approved
                Architect->>Architect: Accept, queue new PENDING children
            else Rejected (retries remain)
                Critic-->>Architect: Rejection reason
                Architect->>Architect: Retry with feedback
            end
        end

        Architect-->>Orchestrator: CDGExport (all leaves ATOMIC)
    end

    rect rgb(16, 185, 129, 0.08)
        Note over Orchestrator,Judge: Round 2: Grounding (with feedback loop)

        loop Orchestration rounds (max 3)
            Orchestrator->>Orchestrator: Convert CDG atomic leaves to PDGNodes

            loop For each unmatched PDGNode
                Orchestrator->>Hunter: find_match(pdg_node)

                Hunter->>Index: Embedding + type search
                Index-->>Hunter: Candidate declarations

                Hunter->>Hunter: LLM ranks candidates

                Hunter->>Judge: Verify top-K candidates
                Judge-->>Hunter: VerificationResult[]

                alt Verified match found
                    Hunter-->>Orchestrator: MatchResult(success=true)
                else No match after retries
                    Hunter-->>Orchestrator: MatchResult(success=false)
                end
            end

            alt All nodes matched
                Note over Orchestrator: Exit grounding loop
            else Some nodes failed
                Orchestrator->>Architect: refine_on_failure(failures, cdg)
                Note over Architect: LLM splits failed nodes<br/>into finer sub-predicates
                Architect-->>Orchestrator: Updated CDG with new atomic nodes
                Note over Orchestrator: Next round matches new nodes
            end
        end
    end

    rect rgb(245, 158, 11, 0.08)
        Note over Orchestrator,Extractor: Round 3: Synthesis

        Orchestrator->>GhostSim: run_ghost_simulation(cdg, matches)
        GhostSim->>GhostSim: Execute abstract witnesses
        GhostSim-->>Orchestrator: GhostSimReport (structural check)

        Orchestrator->>Assembler: assemble(cdg, matches)
        Assembler->>Assembler: Build units, infer glue edges,<br/>topological sort, emit source
        Assembler-->>Orchestrator: SkeletonFile

        Orchestrator->>RepairAgent: repair(skeleton)
        loop Until compiled or budget exhausted
            RepairAgent->>Judge: Compile source
            Judge-->>RepairAgent: CompilerFeedback

            alt Clean compilation
                RepairAgent-->>Orchestrator: Verified SkeletonFile
            else Errors remain
                RepairAgent->>RepairAgent: DeterministicFix / LLMRepair /<br/>SorryElimination
            end
        end

        Orchestrator->>Extractor: extract(result, target)
        Extractor->>Extractor: Build artifact + certificate
        Extractor-->>Orchestrator: ExportBundle
    end

    Orchestrator-->>User: Verified artifact + certificate
    Note over User: output/<br/>  Verified.lean (source)<br/>  .lake/build/ (compiled)<br/>  certificate.json (SHA-256)<br/>  trace.jsonl (optional)
```

## With Principal (optimised pipeline)

When invoked via `sciona optimize`, the Principal wraps the above pipeline in a
meta-optimisation loop. See [05-principal.md](05-principal.md) for the detailed
sequence diagram.

```
Goal + Benchmark dataset
  |
  |  Principal: seed (Optuna suggests parameters)
  v
Architect decomposes goal (new thread)
  |
  |  Principal: forward (ghost sim + synthesis)
  v
ExportBundle (instrumented)
  |
  |  Principal: evaluate (subprocess benchmark)
  v
BenchmarkResult (global_loss + per-node telemetry)
  |
  |  Principal: backward (credit assignment)
  v
NodeGradient[] (bottleneck identified)
  |
  |  Principal: time-travel update (fork Architect checkpoint)
  v
New CDG with constraint injected
  |
  |  (loop back to forward)
  v
Best trial's artifact after budget exhausted
```

## Pipeline phases summary

```
Goal (string)
  |
  |  Round 1: DECOMPOSITION
  |  Architect + Critic + Catalog + Skill Index
  v
CDGExport (tree of atomic nodes + edges)
  |
  |  Round 2: GROUNDING (with feedback)
  |  Hunter + Index + Judge + Orchestrator refinement
  v
CDGExport (refined) + list[MatchResult]
  |
  |  Round 3: SYNTHESIS
  |  Ghost Sim + Assembler + Repair Agent + Extractor
  v
ExportBundle (verified source + compiled artifact + certificate)
  |
  |  (Optional) PRINCIPAL META-OPTIMISATION
  |  Evaluator + CreditAssigner + Optuna + Time-Travel
  v
Best ExportBundle across all trials
```

## Role participation by phase

| Role | Round 1 | Round 2 | Round 3 | Principal |
|------|---------|---------|---------|-----------|
| **Orchestrator** | Invokes Architect | Drives match loop + refinement | Drives assembly + repair + export | -- |
| **Architect** | Decomposes goal | Refines on failure | -- | Time-travel fork + re-decompose |
| **Critic** | Validates decompositions | -- | -- | -- |
| **Catalog** | Atomicity oracle | -- | -- | -- |
| **Skill Index** | Semantic primitive search | -- | -- | -- |
| **Hunter** | -- | Searches + ranks candidates | -- | -- |
| **Index** | -- | Candidate retrieval | -- | -- |
| **Judge** | -- | Type-checks candidates | Compiles skeleton each repair iteration | -- |
| **Ghost Simulator** | -- | -- | Pre-assembly structural validation | Early pruning + precision gradients |
| **Assembler** | -- | -- | CDG + matches -> SkeletonFile | Telemetry instrumentation |
| **Repair Agent** | -- | -- | Iterative compilation + patching | -- |
| **Extractor** | -- | -- | Artifact build + certificate | -- |
| **Principal** | -- | -- | -- | Outer optimisation loop |

## Feedback loops

| Loop | Trigger | Mechanism | Max iterations |
|------|---------|-----------|----------------|
| **Architect retry** | Critic rejects decomposition | Re-decompose with rejection reason | 3 per node |
| **Orchestrator refinement** | Hunter fails to match a node | LLM splits node into sub-predicates | 3 rounds |
| **Hunter reformulation** | Verification fails | LLM analyzes errors, generates new queries | 5 per node |
| **Repair iteration** | Compilation fails | DeterministicFix / LLMRepair / SorryElimination | 10 iterations |
| **Principal trial** | Bottleneck identified | Time-travel fork + constraint injection | configurable (default 50) |
