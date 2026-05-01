# Implementation Plan: `sciona.api` High-Level Python API

## Context

The only entry point to sciona is the CLI (`sciona run`, `sciona decompose`,
etc.). The Kaggle validation pipeline and programmatic integrations need a
Python-native API. A service layer already exists (`sciona/services/`) but
requires manual setup of catalog, indexes, LLM router, and proof environment.

## Design

A single `Sciona` class that:
1. Loads and holds all shared state (catalog, indexes, LLM, config)
2. Exposes every useful CLI operation as an async method
3. Returns typed dataclass/Pydantic results, not JSON strings
4. Is stateless per-call (shared state is read-only after init)

```python
from sciona.api import Sciona

s = await Sciona.create(
    atom_repos=["~/personal/sciona-atoms", "~/personal/sciona-atoms-ml", ...],
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-20250514",
)

result = await s.propose("Predict smartphone location from raw GNSS...")
print(result.cdg.stages)
print(result.grounding.bound_count)
```

## File: `sciona/api.py`

### `Sciona` class

```python
class Sciona:
    """High-level programmatic interface to the sciona framework."""

    def __init__(
        self,
        catalog: PrimitiveCatalog,
        skill_index: SkillIndex,
        semantic_index: SemanticIndex,
        llm_router: LLMRouter,
        config: AgeomConfig,
        proof_env: ProofEnvironment | None = None,
    ) -> None: ...

    @classmethod
    async def create(
        cls,
        atom_repos: list[str | Path] | None = None,
        llm_provider: str = "anthropic",
        llm_model: str | None = None,
        prover: str = "python",
        config_overrides: dict | None = None,
    ) -> Sciona: ...
```

### Methods — organized by pipeline stage

#### Catalog & Retrieval

```python
    @property
    def catalog(self) -> PrimitiveCatalog:
        """The loaded atom catalog (read-only after init)."""

    @property
    def atom_count(self) -> int:
        """Number of atoms in the catalog."""

    def search_atoms(
        self, query: str, k: int = 10
    ) -> list[AtomSearchResult]:
        """Search the catalog by keyword or semantic similarity.
        Wraps: sciona skill search"""

    def find_matching_atoms(
        self, description: str, concept_type: str | None = None, k: int = 5,
    ) -> list[AtomMatch]:
        """Find atoms matching a stage description.
        Wraps: catalog.find_matching_primitives()"""

    def catalog_gaps(
        self, cdg: CDGExport, threshold: float = 0.5,
    ) -> GapReport:
        """Detect atom coverage gaps in a CDG.
        Wraps: sciona catalog-gaps"""
```

#### Decomposition (Round 1 — Architect)

```python
    async def decompose(
        self,
        goal: str,
        max_depth: int = 8,
        thread_id: str | None = None,
    ) -> DecompositionResult:
        """Decompose a high-level goal into a CDG.
        Wraps: sciona decompose
        Returns: CDGExport + metadata (template_used, reasoning, timing)"""

    async def propose(
        self,
        problem: str,
        data: str = "",
        metric: str = "",
        constraints: str = "",
        domain_hints: list[str] | None = None,
    ) -> ProposalResult:
        """High-level: compose a problem prompt, decompose, match, and ground.
        Wraps: sciona run --mode structured (single pass)
        This is the main entry point for the Kaggle validation pipeline."""
```

#### Matching (Round 2 — Hunter)

```python
    async def match_stages(
        self, cdg: CDGExport,
    ) -> list[StageMatchResult]:
        """Match all CDG stages to catalog atoms.
        Wraps: sciona match (batched over CDG nodes)"""

    async def match_statement(
        self, statement: str,
    ) -> MatchResult:
        """Match a single predicate statement to library functions.
        Wraps: sciona match --statement"""
```

#### Assembly & Synthesis (Round 3 — Synthesizer)

```python
    def assemble(
        self, cdg: CDGExport, matches: list[StageMatchResult],
    ) -> SkeletonFile:
        """Assemble a CDG + matches into a compilable skeleton.
        Wraps: sciona assemble"""

    async def compile(
        self, skeleton: SkeletonFile,
    ) -> CompilationResult:
        """Compile a skeleton and return feedback.
        Wraps: sciona synthesize (compile-only, no repair)"""

    async def synthesize(
        self, cdg: CDGExport, matches: list[StageMatchResult],
        max_iterations: int = 5,
    ) -> SynthesisResult:
        """Full assembly + compile + repair cycle.
        Wraps: sciona synthesize"""

    def generate_code(
        self, cdg: CDGExport, matches: list[StageMatchResult],
    ) -> GeneratedCode:
        """Generate pure Python/NumPy code from a grounded CDG.
        Wraps: numpy_codegen + simplifier pipeline
        Returns source code string + imports + dim check report"""
```

#### End-to-End Orchestration

```python
    async def run(
        self,
        goal: str,
        mode: str = "verified_orchestration",
        max_rounds: int = 3,
    ) -> OrchestrationResult:
        """Full pipeline: decompose → match → refine → assemble → compile.
        Wraps: sciona run"""

    async def optimize(
        self,
        goal: str,
        benchmark_path: str | Path,
        metric: str = "minimize",
        trials: int = 20,
        dataset_vars: dict[str, str] | None = None,
    ) -> OptimizationResult:
        """NAS/AutoML optimization loop.
        Wraps: sciona optimize"""
```

#### Export & Inspection

```python
    async def export(
        self, synthesis_result: SynthesisResult,
        target: str = "python-pkg",
        output_dir: str | Path = ".",
    ) -> ExportBundle:
        """Export verified source to FFI bindings and artifacts.
        Wraps: sciona export"""

    def inspect_cdg(
        self, cdg: CDGExport,
    ) -> CDGInspection:
        """Inspect a CDG: stage count, grounding status, topology, dim check.
        No CLI equivalent — new convenience method."""

    def dim_check(
        self, skeleton: SkeletonFile,
    ) -> DimCheckResult:
        """Run dimensional analysis on a skeleton.
        Wraps: dim_checker.check_dimensional_consistency()"""
```

#### Ingestion

```python
    async def ingest(
        self,
        source: str | Path,
        class_name: str,
        output_dir: str | Path | None = None,
        procedural: bool = False,
    ) -> IngestionResult:
        """Ingest source code into atom framework.
        Wraps: sciona ingest"""
```

#### Source Management

```python
    def list_sources(self) -> list[AtomSource]:
        """List resolved atom sources.
        Wraps: sciona sources list"""

    async def sync_sources(self, name: str | None = None) -> SyncResult:
        """Fetch/update git atom sources.
        Wraps: sciona sources sync"""
```

### Result types — `sciona/api_models.py`

```python
@dataclass
class ProposalResult:
    """Result of s.propose() — the main validation entry point."""
    cdg: CDGExport
    template_used: str | None          # Which CDG template was selected
    template_match_score: float
    grounding: GroundingReport
    matches: list[StageMatchResult]
    reasoning: str                     # Architect reasoning trace
    alternatives: list[CDGExport]      # Alternative proposals
    wall_time_seconds: float

@dataclass
class GroundingReport:
    total_stages: int
    bound_active: int
    bound_approximate: int
    orchestration: int
    trivial_inline: int
    external_knowledge: int
    external_tool: int
    unbound: int

    @property
    def grounding_rate(self) -> float:
        resolved = (self.bound_active + self.bound_approximate +
                    self.orchestration + self.trivial_inline +
                    self.external_knowledge + self.external_tool)
        return resolved / self.total_stages if self.total_stages else 0.0

@dataclass
class StageMatchResult:
    stage_id: str
    stage_name: str
    action_class: str                  # replace_stage, orchestration, etc.
    matched_atom: str | None           # Atom FQDN
    match_confidence: float
    top_candidates: list[AtomMatch]
    reasoning: str

@dataclass
class AtomMatch:
    atom_name: str
    atom_fqdn: str
    score: float
    category: str
    description: str

@dataclass
class AtomSearchResult:
    atom_name: str
    atom_fqdn: str
    source: str
    category: str
    description: str
    score: float

@dataclass
class GeneratedCode:
    source: str                        # Python source code
    imports: list[str]                 # Required import statements
    dim_check: DimCheckResult          # Dimensional consistency report
    atom_fqdns_used: list[str]         # Which atoms were used

@dataclass
class CDGInspection:
    total_stages: int
    total_edges: int
    grounding: GroundingReport
    topology: str                      # "linear", "dag", "has_fixed_point", etc.
    concept_types: dict[str, int]      # Count per concept type
    dim_annotations: int               # Stages with dimensional signatures
    max_depth: int
```

## Implementation strategy

### Step 1: Create `sciona/api.py` and `sciona/api_models.py`

The `Sciona` class wraps existing infrastructure:

```python
# sciona/api.py
class Sciona:
    def __init__(self, catalog, skill_index, semantic_index, llm_router, config, proof_env):
        self._catalog = catalog
        self._skill_index = skill_index
        self._semantic_index = semantic_index
        self._llm = llm_router
        self._config = config
        self._proof_env = proof_env

        # Pre-build services
        self._architect_svc = ArchitectService(...)
        self._hunter_svc = HunterService(...)
        self._synthesizer_svc = SynthesizerService(...)

    @classmethod
    async def create(cls, atom_repos=None, llm_provider="anthropic", ...):
        config = AgeomConfig()
        # Apply overrides
        catalog, _ = _load_architect_catalog(args_proxy, config)
        # If atom_repos specified, load from those instead of config
        if atom_repos:
            catalog = _load_catalog_from_repos(atom_repos)
        skill_index = _load_skill_index_or_empty(config)
        semantic_index, _ = _load_semantic_index(config.index_dir, config)
        llm_router = _create_llm_router(config, provider=llm_provider, model=llm_model)
        proof_env = _create_proof_env(Prover(prover), config) if prover else None
        return cls(catalog, skill_index, semantic_index, llm_router, config, proof_env)
```

### Step 2: Wrap each service call

Each method delegates to the existing service layer, converting between
API-level result types and internal types:

```python
    async def propose(self, problem, data="", metric="", constraints="", domain_hints=None):
        # 1. Format prompt
        goal = self._format_goal(problem, data, metric, constraints, domain_hints)
        # 2. Decompose
        cdg = await self._architect_svc.decompose(ArchitectDecomposeRequest(goal=goal, ...))
        # 3. Match
        matches = await self._hunter_svc.match(HunterBatchMatchRequest(nodes=cdg.nodes, ...))
        # 4. Build grounding report
        grounding = self._build_grounding_report(cdg, matches)
        # 5. Return
        return ProposalResult(cdg=cdg, grounding=grounding, matches=matches, ...)
```

### Step 3: Add `_load_catalog_from_repos()` helper

The `create()` classmethod needs to load atoms from specified repo paths
(not just the config-based discovery). Extract and generalize the loading
logic from `test_retrieval_solution_cdgs.py::_load_all_atom_primitives()`.

```python
def _load_catalog_from_repos(repo_paths: list[Path]) -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    seed_builtin_primitives(catalog)
    for repo in repo_paths:
        for cdg_path in repo.rglob("cdg.json"):
            if "solution_cdgs" in str(cdg_path):
                continue
            data = json.loads(cdg_path.read_text())
            for node in data.get("nodes", []):
                if node.get("status") != "atomic":
                    continue
                catalog.add(AlgorithmicPrimitive(
                    name=node["node_id"],
                    source=str(cdg_path.parent.relative_to(repo)),
                    category=ConceptType(node.get("concept_type", "custom")),
                    description=node.get("description", ""),
                    inputs=[IOSpec(**inp) for inp in node.get("inputs", [])],
                    outputs=[IOSpec(**out) for out in node.get("outputs", [])],
                    type_signature=node.get("type_signature", ""),
                    aliases=node.get("aliases", []),
                ))
    seed_solution_retrieval_aliases(catalog)
    return catalog
```

### Step 4: Tests

```
tests/test_api.py
  - test_create_loads_catalog
  - test_search_atoms_returns_results
  - test_find_matching_atoms_returns_ranked
  - test_inspect_cdg_grounding_report
  - test_propose_returns_cdg  (requires LLM mock or integration)
  - test_generate_code_returns_source
```

## Files to create/modify

| File | Action |
|------|--------|
| `sciona/api.py` | **Create** — Sciona class |
| `sciona/api_models.py` | **Create** — Result dataclasses |
| `sciona/__init__.py` | **Modify** — re-export `from sciona.api import Sciona` |
| `tests/test_api.py` | **Create** — Unit tests |
| `sciona/commands/runtime_helpers.py` | **Modify** — extract `_load_catalog_from_repos()` |

## Methods NOT exposed (intentionally)

- `sciona login` — authentication is out of scope for programmatic API
- `sciona receipt sign/verify` — specialized bounty workflow
- `sciona prompt-benchmark` — internal developer tooling
- `sciona upsert-cdg` — Memgraph admin operation
- `sciona telemetry list/show` — internal observability
- `sciona benchmark-validate` / `release-validate` — CI-only
- `sciona visualize` — browser-based, not programmatic
- `sciona history` — checkpoint inspection, use thread_id on decompose instead

## Dependency notes

- The API module itself has no new dependencies
- `create()` is async because LLM router setup may involve network calls
- Methods that invoke the LLM are async; pure catalog/assembly methods are sync
- The `proof_env` is optional — code generation works without it, but
  `compile()` and `synthesize()` require it
