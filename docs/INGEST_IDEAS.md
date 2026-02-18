# Smart Ingester: Converting Existing Python to Stateless Atom ASTs

Design notes for a smart ingester that takes existing Python classes/functions
and converts them into stateless function graphs mappable onto the AGEO
framework.

## Reference: `PPGProcessor`

`PPGProcessor` (`happyml/sensors/ppg.py`) is the motivating example. It's a
stateful DSP pipeline with:

1. **A preprocessing chain** (`__init__`): median filter, conditional de-drift
   (histogram / envelope / TV1D / butterworth LP), optional inversion, optional
   bandpass filter.
2. **Peak detection** with 4 pluggable strategies (Zong, Elgendi, Kavsaoglu,
   Conrad) -- some of which internally upsample.
3. **Beat slicing** from detected onsets into `BeatData` objects.
4. **A multi-stage filtering pipeline** (`process()`): RAW, FIXED (notch
   repair), MOTION (accel filtering), HR (SQI threshold), IBI (IBI filter),
   IBI_POST (outlier removal).
5. **Cross-window state** via `parent` (SQI accumulator, variance tracker).

---

## The Granularity Problem

Atomization granularity is the central design tension. Get it wrong in either
direction and the system fails:

- **Too fine** (exposing `np.mean`, `scipy.signal.lfilter` as individual nodes):
  the agent drowns in noise and hallucinates invalid DSP chains. There are too
  many possible wirings and no semantic guard rails.
- **Too coarse** (one black box called `GetHeartRate`): the agent can't inspect,
  debug, or improve the logic. You've just wrapped the existing code in a new
  API without gaining composability.

**The right level is functional intent** -- atoms named for *what they
accomplish* (e.g., "Detect Onsets", "Assess Signal Quality"), not *how they do
it* (e.g., "Find Zero Crossings", "Apply Butterworth Filter").

---

## Architecture: Functional Core, Imperative Shell

`PPGProcessor` is currently an OO "Manager" class. It holds state
(`self.beats`, `self.internals`) and mutates it over time. To make it
agent-friendly, refactor into **pure functional nodes** where state is
externalized into the graph edges.

### The Macro-Atom Mapping

Break `process()` into 5 distinct macro-atoms:

| Current PPGProcessor Method        | Proposed Agentic Atom        | Input State                      | Output State                       |
|------------------------------------|------------------------------|----------------------------------|------------------------------------|
| `__init__` (filters, detrending)   | **Signal Conditioner**       | `RawWindow`                      | `ConditionedSignal`, `Envelope`    |
| `peak_detect_X()`                  | **Onset Detector**           | `ConditionedSignal`              | `OnsetIndices`                     |
| `slice_beats()`                    | **Beat Slicer**              | `ConditionedSignal`, `OnsetIndices` | `BeatCollection[RAW]`           |
| `beats.compute_sqi()`             | **SQI Evaluator**            | `BeatCollection[RAW]`, `SQIPool (Prev)` | `BeatCollection[HR]`, `SQIPool (Next)` |
| `post_process()` / `filter_acc`   | **Logic Gate**               | `BeatCollection[HR]`, `AccelData` | `IbiResult`                       |

---

## Handling State: History as an Edge

The hardest part is `BeatSQI`. It maintains a rolling heap (pool) of good beats
across windows. In a graph architecture, nodes must be stateless.

**Solution: treat history as an edge.** Instead of the node holding the pool,
the graph passes the pool as an argument into the node and receives an updated
pool out.

```python
from pydantic import BaseModel, Field
from typing import List, Tuple

# 1. Define the data contract (the "edge")
class BeatSQIState(BaseModel):
    """The persistent memory of the SQI agent."""
    beat_pool: List[Tuple[float, BeatData]] = Field(default_factory=list)
    running_threshold: float = 0.5

class SQIInput(BaseModel):
    current_beats: BeatCollection
    history: BeatSQIState

class SQIOutput(BaseModel):
    scored_beats: BeatCollection
    updated_history: BeatSQIState

# 2. Define the atom (the "node")
def atom_evaluate_sqi(inputs: SQIInput) -> SQIOutput:
    """
    Functional wrapper around the BeatSQI class.
    Hydrates state, runs logic, dehydrates result.
    """
    # Hydrate the legacy object from Pydantic state
    sqi_runner = BeatSQI(sqi_threshold=inputs.history.running_threshold)
    sqi_runner.pool = inputs.history.beat_pool

    # Run existing logic
    for beat in inputs.current_beats:
        sqi_runner(beat)  # mutates runner and beat

    # Dehydrate state back to Pydantic
    new_state = BeatSQIState(
        beat_pool=sqi_runner.pool,
        running_threshold=sqi_runner.threshold,
    )

    return SQIOutput(
        scored_beats=inputs.current_beats,
        updated_history=new_state,
    )
```

Why this works for agents:

- **Time travel**: debug a specific failure by re-running just the SQI node
  with the saved JSON state input.
- **No hidden side effects**: the agent sees exactly what data is required to
  make a decision (the pool).
- **Testability**: each node can be unit-tested with synthetic state inputs.

---

## The Zoom-In Architecture

The granularity problem has a dynamic solution: **macro-atoms that decompose on
demand**.

At the top level, expose the 5 macro-atoms. The agent plans and executes at
this level. But when the agent needs to *optimize* a specific step, it can
request decomposition of that macro-atom into its sub-graph.

### Example: Onset Detector

At the macro level, `OnsetDetector` is a single node with a `strategy`
parameter:

- **Standard mode**: the agent picks `strategy="zong"` and moves on.
- **Expert mode** ("zoom in"): the agent requests decomposition. The
  `OnsetDetector` expands into its sub-graph of scipy calls:

```
[Diff] -> [Smooth(Boxcar)] -> [Threshold] -> [FindPeaks]
```

This enables **algorithm refinement**:

> "The Zong detector is failing on this high-motion data. I will swap the
> `Smooth(Boxcar)` sub-atom with a `Smooth(Gaussian)` atom and re-run."

**Rule of thumb**: do NOT use subgraph decomposition for high-level
orchestration (too noisy). DO use it for the optimizer agent when refining a
specific macro-atom's internals.

---

## Typed Graph Edges: Preventing Garbage Combinations

To prevent the agent from wiring `BeatSlicer` directly into `LogicGate`
(skipping SQI), use **typed graph edges**. In pydantic-graph, define valid
transitions based on types:

```python
# The agent CANNOT connect Slicer -> Gate because types don't match.
class SlicerOutput(BaseModel):
    beats: BeatCollection[Raw]      # Type: Raw

class SQIInput(BaseModel):
    beats: BeatCollection[Raw]

class GateInput(BaseModel):
    beats: BeatCollection[Scored]   # Type: Scored

# The type system prevents skipping SQI --
# there's no way to get BeatCollection[Scored] without running the SQI node.
```

This is the verification layer that prevents hallucinated pipelines. The ghost
witness system already does this for abstract types; typed edges extend it to
the concrete pipeline level.

---

## Extraction Approaches

Three complementary techniques for the mechanical extraction:

### Approach 1: AST-driven method extraction + data flow analysis

Parse the class with `ast`, extract each method as a standalone function
candidate. For each method, trace `self.*` reads/writes to build a data-flow
graph. The key insight: `self.foo` on the right-hand side of an assignment is an
*input*, `self.foo` on the left-hand side is an *output*. This turns each method
into `f(inputs) -> outputs` where the inputs/outputs are the `self.*`
attributes. The `__init__` becomes a chain of transforms on `self.raw` ->
`self.medraw` -> `self.dedrifted` -> `self.ppg` -> `self.filtered`, each
conditionally applied.

The hard part: conditional branches. The `__init__` has
`if self.options.dedrift_hist`, `if self.options.dedrift_env`, etc. These are
**configuration-gated pipeline stages**, not true control flow. The ingester
would need to recognize this pattern and emit the pipeline as a graph with
optional nodes, not as a single monolithic function.

### Approach 2: Trace-based decomposition

Instead of static analysis, instrument the class and run it on representative
data. Record which methods are called and what the actual data shapes/types are
at each step. This gives you a concrete execution trace that can be converted to
a CDG. Much more robust than AST analysis for dynamic Python, but requires test
data.

### Approach 3: LLM-assisted semantic chunking

Use the LLM (which the Architect already has) to read the class and identify the
logical pipeline stages, then emit a CDG. The LLM is good at recognizing that
the `__init__` is really "preprocess, dedrift, filter" and that `process()` is
"detect beats, slice, filter stages". This is the most flexible but least
deterministic approach.

### Recommended: Hybrid of all three

1. **Static AST pass** -- extract methods, map `self.*` reads/writes, build the
   dependency graph. Identifies `invert()`, `peak_detect_*()`,
   `refine_onsets()`, `slice_beats()`, `window_sqi()` as nearly-pure functions.

2. **Configuration flattening** -- recognize `if self.options.X` as pipeline
   variant selection. Emit each variant as a separate optional node.

3. **State hoisting** -- externalize cross-window state into typed edges
   (the "history as an edge" pattern above).

4. **LLM refinement** -- after mechanical extraction, use the Architect LLM to
   name atoms semantically, match to known DSP primitives in the catalog, and
   infer type signatures. Recognizes that `butter_lowpass_filter(signal, 0.5,
   fs, 3)` maps to existing `butter` + `lfilter` atoms, that `tv1d` is
   total-variation denoising, etc.

---

## Concrete Output for PPGProcessor

### Macro-atoms (the primary agent interface)

```
Node A: SignalConditioner
  Input:  RawWindow(ppg_t, ppg_signal, options)
  Output: ConditionedSignal(signal, time, fs, envelope?)

Node B: OnsetDetector
  Input:  ConditionedSignal
  Output: OnsetIndices(indices, time_base)
  Config: strategy = zong | elgendi | kavsaoglu | conrad

Node C: BeatSlicer
  Input:  ConditionedSignal, OnsetIndices
  Output: BeatCollection[Raw]

Node D: SQIEvaluator
  Input:  BeatCollection[Raw], BeatSQIState(prev)
  Output: BeatCollection[Scored], BeatSQIState(next)

Node E: LogicGate
  Input:  BeatCollection[Scored], AccelData?
  Output: IbiResult(hr, hr_t, ibi, sqi, hrv)
```

### Sub-atoms (available via zoom-in)

```
SignalConditioner decomposes to:
  median_filter_demean(signal, kernel_size) -> signal
  histogram_detrend(signal) -> signal              [optional: dedrift_hist]
  get_ppg_baseline(t, signal, lp_cutoff) -> baseline, env  [optional: dedrift_env]
  tv1d_denoise(signal, alpha) -> signal            [optional: dedrift_tv1d]
  butter_lowpass_subtract(signal, cutoff, fs) -> signal     [optional: dedrift_lp_butter]
  invert_signal(signal) -> signal                  [optional: invert_ppg]

OnsetDetector(strategy=conrad) decomposes to:
  [tv1d] -> [find_periodic_peaks] -> [conditional upsample] -> [refine_onsets]

OnsetDetector(strategy=zong) decomposes to:
  [diff] -> [smooth(boxcar)] -> [threshold] -> [find_peaks]
```

### Pipeline graph

```
RawWindow
  -> SignalConditioner -> ConditionedSignal
  -> OnsetDetector     -> OnsetIndices
  -> BeatSlicer        -> BeatCollection[Raw]
  -> SQIEvaluator      -> BeatCollection[Scored]  (+ SQIState passed through)
  -> LogicGate         -> IbiResult
```

---

## Ghost Witnesses for Macro-Atoms

Each macro-atom gets a ghost witness that describes the data transformation at
the metadata level without executing the heavy code:

```python
def witness_signal_conditioner(
    raw: AbstractSignal,
) -> AbstractSignal:
    """Signal shape preserved, dtype float64, domain stays time."""
    return AbstractSignal(
        shape=raw.shape, dtype="float64",
        sampling_rate=raw.sampling_rate, domain="time",
    )

def witness_onset_detector(
    signal: AbstractSignal,
) -> AbstractOnsetIndices:
    """Output is integer indices, count unknown until runtime."""
    return AbstractOnsetIndices(
        max_index=signal.shape[0] - 1,
        dtype="int64",
    )

def witness_sqi_evaluator(
    beats: AbstractBeatCollection,
    history: AbstractSQIState,
) -> Tuple[AbstractBeatCollection, AbstractSQIState]:
    """Beat count may decrease (filtering), pool size increases."""
    return (
        AbstractBeatCollection(max_count=beats.max_count),
        AbstractSQIState(pool_size=history.pool_size + beats.max_count),
    )
```

This lets the planner verify that a `SignalConditioner -> OnsetDetector ->
BeatSlicer -> SQIEvaluator -> LogicGate` pipeline is structurally sound
*before* running any actual DSP code.

---

## Key Design Decisions

1. **Granularity strategy** -- atomize at functional intent. Macro-atoms are the
   default. Sub-atom decomposition is available on demand for the optimizer
   agent. Match sub-atoms against the existing index to decide whether to
   decompose further (if `butter` and `lfilter` are already atoms, decompose
   `butter_lowpass_filter` into those; if not, keep it as one atom).

2. **Configuration handling** -- single CDG with optional/conditional nodes.
   Each `options.*` flag maps to an optional node in the graph. The agent
   selects which optional nodes to activate based on signal characteristics.

3. **State boundary** -- "history as an edge" pattern. Cross-window state is
   explicitly threaded through typed Pydantic models. The atoms are stateless;
   the graph carries the state. This gives time-travel debugging for free.

4. **Output format** -- the ingester produces both:
   - **CDG nodes** for the Architect (the 5 macro-atoms as `AlgorithmicNode`s
     with typed edges).
   - **AGEO atoms** for the library (Python functions with `@register_atom` +
     icontract contracts + ghost witnesses).

5. **Verification** -- typed graph edges prevent invalid wirings at the macro
   level. Ghost witnesses validate structural compatibility at the sub-atom
   level. Together they prevent hallucinated pipelines without requiring
   heavyweight formal verification.
