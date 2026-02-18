# Round 2: Hunter -- Predicate Grounding

Grounds each atomic predicate from the CDG to a verified library function by
searching a FAISS index and type-checking candidates against the target prover.

```mermaid
sequenceDiagram
    participant Orchestrator
    participant Hunter
    participant LLM as Hunter LLM
    participant Index as FAISS Index
    participant Judge
    participant ProofEnv as Proof Environment<br/>(Lean/Coq/Python)

    Orchestrator->>Hunter: find_match(pdg_node)

    Note over Hunter: InitialSearch

    Hunter->>Hunter: Build query from<br/>statement + informal_desc

    Hunter->>Index: search_by_embedding(query, k=20)
    Index-->>Hunter: [(Declaration, score), ...]

    Hunter->>Index: search_by_type(statement, k=20)
    Index-->>Hunter: [Declaration, ...]

    Hunter->>Hunter: Merge, deduplicate,<br/>wrap as CandidateMatch[]

    Note over Hunter: RankCandidates

    Hunter->>LLM: Rank candidates by relevance<br/>(SCORE_CANDIDATES prompt)
    LLM-->>Hunter: [0, 2, 1, ...] (reordered indices)
    Hunter->>Hunter: Reorder candidates

    Note over Hunter: VerifyTopK

    loop For each of top-K unverified candidates
        Hunter->>Judge: verify_candidate(pdg_node, candidate)
        Judge->>ProofEnv: check_term(@candidate.name, statement)

        alt Lean 4
            ProofEnv->>ProofEnv: Compile: example : {type} := @{name}
        else Coq
            ProofEnv->>ProofEnv: Compile: Definition _check : {type} := {name}.
        else Python
            ProofEnv->>ProofEnv: mypy --strict: _result: {type} = {name}
        end

        ProofEnv-->>Judge: (success, compiler_output)
        Judge->>Judge: Resolve VerificationLevel<br/>(KERNEL_PROOF / TYPE_CHECKED / UNVERIFIED)
        Judge-->>Hunter: VerificationResult
    end

    alt Verified match found
        Hunter-->>Orchestrator: MatchResult(success=true,<br/>verified_match=result)
    else No match, iterations remain
        Note over Hunter: ReformulateQuery

        Hunter->>LLM: Why did verification fail?<br/>(ANALYZE_FAILURE prompt + compiler errors)
        LLM-->>Hunter: failure analysis

        Hunter->>LLM: Generate new search queries<br/>(REFORMULATE_QUERY prompt + analysis)
        LLM-->>Hunter: ["new_query_1", "new_query_2", ...]

        Hunter->>Hunter: Loop back to InitialSearch<br/>with reformulated queries
    else Budget exhausted
        Hunter-->>Orchestrator: MatchResult(success=false,<br/>all_candidates, all_verifications)
    end
```

## Data produced

| Artifact | Type | Description |
|----------|------|-------------|
| Match Result | `MatchResult` | Verified match (if found) + full audit trail of candidates and verifications |
| Verification Level | `VerificationLevel` | KERNEL_PROOF (Lean/Coq), TYPE_CHECKED (Python), or UNVERIFIED |
| Compiler Feedback | `list[str]` | Error messages from failed verifications, used for query reformulation |

## Hunter state accumulated per PDGNode

| Field | Description |
|-------|-------------|
| `candidates_found` | All discovered candidates across iterations |
| `verification_results` | All verification attempts (pass and fail) |
| `queries_tried` | Query history for deduplication |
| `compiler_feedback` | Error messages for reformulation context |
| `iteration` | Current iteration counter (vs. max_iterations) |

## LLM calls per iteration

| Step | Prompt | Output |
|------|--------|--------|
| RankCandidates | SCORE_CANDIDATES | `[indices]` (reordered) |
| ReformulateQuery | ANALYZE_FAILURE | failure analysis text |
| ReformulateQuery | REFORMULATE_QUERY | `["query", ...]` |
