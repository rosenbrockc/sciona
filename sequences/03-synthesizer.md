# Round 3: Synthesizer -- Assembly, Simulation, Repair, Export

Composes matched atoms into a single compilable source file, validates it
through ghost simulation, repairs compilation errors, and exports the verified
artifact.

```mermaid
sequenceDiagram
    participant Orchestrator
    participant GhostSim as Ghost Simulator
    participant Registry as Ghost Registry<br/>(ageoa)
    participant Assembler
    participant RepairAgent as Repair Agent
    participant LLM as Synthesizer LLM
    participant Judge
    participant ProofEnv as Proof Environment<br/>(Lean/Coq/Python)
    participant Extractor

    Orchestrator->>GhostSim: run_ghost_simulation(cdg, matches)

    Note over GhostSim: Phase 1: Ghost Simulation

    GhostSim->>Registry: list_registered()
    Registry-->>GhostSim: registered atom names

    loop For each atomic leaf node
        GhostSim->>GhostSim: Extract atom name from match
        alt Atom has registered witness
            GhostSim->>GhostSim: Mark simulable
        else No witness
            GhostSim->>GhostSim: Mark skipped
        end
    end

    GhostSim->>GhostSim: Compute coverage ratio

    GhostSim->>GhostSim: Build abstract input values<br/>(AbstractSignal, AbstractArray, etc.)

    loop For each simulable node (topological order)
        GhostSim->>Registry: get_witness(atom_name)
        Registry-->>GhostSim: witness function
        GhostSim->>GhostSim: Execute witness on abstract values
        GhostSim->>GhostSim: Propagate output to downstream nodes
    end

    alt Simulation passes
        GhostSim-->>Orchestrator: GhostSimReport(passed=true, coverage)
    else Structural mismatch detected
        GhostSim-->>Orchestrator: GhostSimReport(passed=false,<br/>error_node, error_function)
    end

    Note over Assembler: Phase 2: Assembly

    Orchestrator->>Assembler: assemble(cdg, match_results)

    Assembler->>Assembler: Build AssemblyUnits<br/>from verified matches
    Assembler->>Assembler: Infer GlueEdges + cast expressions
    Assembler->>Assembler: Topological sort
    Assembler->>Assembler: Emit language-specific source<br/>(Lean 4 / Python / Coq)
    Assembler->>Assembler: Generate composition functions<br/>for root/decomposed nodes

    Assembler-->>Orchestrator: SkeletonFile(source_code,<br/>units, glue_edges, sorry_count)

    Note over RepairAgent: Phase 3: Repair Loop

    Orchestrator->>RepairAgent: repair(skeleton)

    loop Until compiled or budget exhausted (max 10 iterations)
        Note over RepairAgent: CompileCheck

        RepairAgent->>Judge: compile(source_code)
        Judge->>ProofEnv: _run(source_code)
        ProofEnv-->>Judge: CompilerFeedback
        Judge-->>RepairAgent: {success, errors[], goals[]}

        RepairAgent->>RepairAgent: Snapshot source for rollback
        RepairAgent->>RepairAgent: Track best_source (lowest errors)

        alt Regression detected (errors increased)
            RepairAgent->>RepairAgent: Rollback to previous snapshot
        end

        alt Compiled successfully
            RepairAgent-->>Orchestrator: SkeletonFile (verified)
        else Deterministic fixes available
            Note over RepairAgent: DeterministicFix
            RepairAgent->>RepairAgent: Batch-apply all<br/>missing imports/opens
            RepairAgent->>RepairAgent: Loop back to CompileCheck
        else LLM-solvable errors
            Note over RepairAgent: LLMRepair
            RepairAgent->>RepairAgent: Pick highest-priority error
            RepairAgent->>LLM: Analyze error + generate patch<br/>(ANALYZE_ERROR prompt)
            LLM-->>RepairAgent: {line_start, line_end,<br/>replacement, description}
            RepairAgent->>RepairAgent: Apply patch,<br/>loop back to CompileCheck
        else Only sorry/Admitted remain
            Note over RepairAgent: SorryElimination
            RepairAgent->>RepairAgent: Find first sorry location
            RepairAgent->>LLM: Generate proof tactic / implementation<br/>(GENERATE_TACTIC prompt)
            LLM-->>RepairAgent: tactic body or implementation
            RepairAgent->>RepairAgent: Replace sorry,<br/>loop back to CompileCheck
        else Budget exhausted
            RepairAgent-->>Orchestrator: best_source (lowest error count)
        end
    end

    Note over Extractor: Phase 4: Export

    Orchestrator->>Extractor: extract(synthesis_result, target)

    Extractor->>Extractor: Write verified source to disk
    Extractor->>ProofEnv: Build artifact<br/>(lake build / coqc / mypy)
    ProofEnv-->>Extractor: compiled artifact
    Extractor->>Extractor: Generate SHA-256 certificate
    Extractor->>Extractor: Generate FFI bindings (optional)

    Extractor-->>Orchestrator: ExportBundle(source, artifact,<br/>certificate, ffi_files)
```

## Phases and data flow

| Phase | Input | Output | LLM? | Compiler? |
|-------|-------|--------|------|-----------|
| Ghost Simulation | CDG + MatchResults | GhostSimReport | No | No |
| Assembly | CDG + MatchResults | SkeletonFile | No | No |
| Repair Loop | SkeletonFile | SkeletonFile (repaired) | Yes | Yes (each iteration) |
| Export | SynthesisResult | ExportBundle + Certificate | No | Yes (final build) |

## Repair Agent error priority

| Priority | Category | Fix strategy |
|----------|----------|--------------|
| 0 | TYPE_MISMATCH | LLMRepair |
| 1 | UNIVERSE_MISMATCH | LLMRepair |
| 2 | MISSING_IMPORT | DeterministicFix (batch) |
| 3 | SYNTAX | LLMRepair |
| 4 | UNKNOWN | LLMRepair |
| 5 | UNSOLVED_GOAL | SorryElimination |

## LLM calls per repair iteration

| Node | Prompt | Output |
|------|--------|--------|
| LLMRepair | ANALYZE_ERROR_SYSTEM/USER | `{line_start, line_end, replacement}` |
| SorryElimination | GENERATE_TACTIC_SYSTEM/USER | tactic body or implementation code |
