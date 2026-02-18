# Round 1: Architect -- Decomposition

Decomposes a high-level goal into a Conceptual Dependency Graph (CDG) of atomic
algorithmic operations.

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant Architect
    participant LLM as Architect LLM
    participant Catalog
    participant SkillIndex as Skill Index
    participant Critic

    User->>Orchestrator: goal string
    Orchestrator->>Architect: decompose(goal)

    Note over Architect: select_strategy

    Architect->>LLM: Pick paradigm for goal<br/>(SELECT_STRATEGY prompt)
    LLM-->>Architect: {paradigm, rationale, variant_hint}

    Architect->>Architect: Create root node (status=DECOMPOSED)
    Architect->>Architect: Instantiate skeleton template<br/>for chosen paradigm

    Architect->>Catalog: is_atomic(skeleton_node)?
    Catalog-->>Architect: true/false per node
    Architect->>Architect: Mark matched nodes ATOMIC,<br/>queue remaining as PENDING

    loop For each PENDING node
        Note over Architect: decompose_node

        Architect->>Catalog: find_matching_primitives(node, k=5)
        Catalog-->>Architect: keyword-matched primitives

        Architect->>SkillIndex: search(name + description, k=5)
        SkillIndex-->>Architect: embedding-matched primitives

        Architect->>Architect: Merge & deduplicate primitives (top 10)

        Architect->>LLM: Decompose node into sub-nodes<br/>(DECOMPOSE_NODE prompt + primitives)
        LLM-->>Architect: {sub_nodes[], edges[]}

        Architect->>Catalog: Validate atomic claims
        Catalog-->>Architect: confirmed / rejected

        Note over Architect,Critic: critique_decomposition

        Critic->>Critic: Phase A: Deterministic checks<br/>(arity, depth, self-loops, I/O names)

        alt Deterministic checks pass
            Critic->>LLM: Is this decomposition correct?<br/>(CRITIQUE prompt + parent + children)
            LLM-->>Critic: {approved, reason, flagged_nodes}
        else Deterministic checks fail
            Critic-->>Architect: REJECT (reason)
        end

        alt Critique approved
            Architect->>Architect: Mark parent DECOMPOSED,<br/>add PENDING children to queue
        else Critique rejected (retries < 3)
            Note over Architect: prepare_retry
            Architect->>Architect: Mark children REJECTED,<br/>inject rejection reason into next prompt
            Architect->>LLM: Re-decompose with retry context
            LLM-->>Architect: revised {sub_nodes[], edges[]}
        else Critique rejected (retries exhausted)
            Architect->>Architect: Mark parent HIGH_RISK
        end
    end

    Note over Architect: All leaves ATOMIC or HIGH_RISK

    Architect->>Architect: Filter out REJECTED nodes
    Architect->>Architect: validate_handoff_strict(cdg)<br/>(syntax, connectivity, arity)
    Architect->>Architect: to_pdg_nodes(cdg) for Round 2

    Architect-->>Orchestrator: CDGExport + list[PDGNode]
```

## Data produced

| Artifact | Type | Description |
|----------|------|-------------|
| CDG | `CDGExport` | Tree of `AlgorithmicNode` + `DependencyEdge` with all leaves ATOMIC |
| PDG Nodes | `list[PDGNode]` | One per atomic leaf, carrying formal statement + informal description |
| Metadata | `dict` | goal, paradigm, thread_id, num_nodes, num_edges |

## LLM calls per node

| Step | Prompt | Output |
|------|--------|--------|
| select_strategy | SELECT_STRATEGY_SYSTEM/USER | `{paradigm, rationale}` |
| decompose_node | DECOMPOSE_NODE_SYSTEM/USER | `{sub_nodes[], edges[]}` |
| critique | CRITIQUE_SYSTEM/USER | `{approved, reason, flagged_nodes[]}` |
