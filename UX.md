# CDG Visualizer UX Refinements

## Context

The visualizer is a Cytoscape.js + FastAPI app serving 1,124 atoms across 122 CDGs from memgraph. The largest CDGs have 70 nodes and 148 data-flow edges. After the concept_type reclassification, there are 30 distinct concept types in use.

---

## 1. Collapse/expand decomposed nodes

**Problem:** The biggest CDGs (baseline=66, spo2=66, signal_particle_filter=70) render as a single flat canvas. With 70 nodes and 148 edges the dagre layout becomes a wall of spaghetti.

**Fix:**
- Decomposed nodes start collapsed, showing only top-level macro-atoms (typically 5-12 per CDG)
- Click or double-click a decomposed node to expand its children inline
- A breadcrumb trail at the top shows the zoom path (`HPYSpO2Runner > InitializeRunnerState > ...`)
- Turns a 70-node flat graph into a 12-node overview with drill-down

**Impact:** Highest | **Effort:** Medium

---

## 2. Hover-highlight upstream/downstream

**Problem:** Edge labels show `output_name -> input_name` as text, but with 30+ edges they are unreadable. Users cannot answer "where does this signal come from?" at a glance.

**Fix:**
- On hover over any node, highlight all upstream edges blue and downstream edges orange, dimming everything else to 15% opacity
- On hover over an edge, show a tooltip: `source.output_name (type) -> target.input_name (type)`

**Impact:** High | **Effort:** Low

---

## 3. Color-code by concept_type families

**Problem:** concept_type is encoded as shape, but 15+ Cytoscape shapes (hexagon, diamond, vee, rhomboid...) are indistinguishable at small sizes. The 6 new types (state_init, data_assembly, etc.) have no shape mapping and fall through to the default ellipse.

**Fix:** Swap the encoding: **color = concept_type**, **shape = status** (only 3 values: atomic=circle, decomposed=rounded-rectangle, external=diamond).

Group the 30 concept_types into 6 color families:

| Family | Color | Types |
|--------|-------|-------|
| Math/algo | Blue | sorting, searching, divide_and_conquer, greedy, dynamic_programming, combinatorics, algebra, analysis, arithmetic, number_theory, geometry, set_theory |
| Probabilistic | Purple | sampler, log_prob, posterior_update, variational_inference, prior_init, prior_distribution, likelihood_evaluation, probabilistic_oracle, oracle_gradient, mcmc_kernel, mcmc_proposal, vi_elbo, conjugate_update |
| Signal | Teal | signal_filter, signal_transform, graph_signal_processing, sequential_filter, smc_reweight |
| Orchestration | Amber | state_init, data_assembly, conditional_routing, data_extraction |
| Presentation | Green | visualization, observability |
| Other | Gray | custom, external_tool, message_passing, neural_network |

Add a toggleable legend panel showing the color mapping.

**Impact:** High | **Effort:** Low

---

## 4. In-graph search

**Problem:** Users can search the CDG browser, but once a CDG is loaded there is no way to find a node in a 70-node graph.

**Fix:**
- Add a search/filter input to the toolbar
- Typing `particle` highlights matching nodes and dims the rest
- Support structured queries: `type:sampler`, `status:atomic`, `depth:>1`
- Enter cycles through matches, panning the view to center each one

**Impact:** High | **Effort:** Low

---

## 5. Grouped tree browser

**Problem:** With 122 CDGs the slide-out browser is a long flat list with no way to understand the landscape.

**Fix:**
- Group repos by top-level namespace: `hpy-atoms/` (26), `ageo-atoms/biosppy/` (10), `ageo-atoms/mcmc_foundational/` (12), etc.
- Show collapsible groups with atom count badges
- Add filter chips for concept_type families (click "probabilistic" to show only CDGs with sampler/mcmc/vi nodes)
- Show a mini bar chart per CDG of concept_type distribution

**Impact:** Medium | **Effort:** Medium

---

## 6. Tabbed detail panel with lineage

**Problem:** The right panel is a static dump. For a node with 8 inputs and 5 outputs the tables dominate and the description is buried.

**Fix:** Replace with tabs:
- **Summary** (default): name, description, concept_type badge (colored), status badge, type signature in monospace
- **Ports**: inputs/outputs tables with type badges; clicking a type highlights all edges carrying it
- **Lineage**: upstream/downstream mini-graph (3 levels) showing data-flow neighbors of the selected node
- **Code** (if available): syntax-highlighted source from witnesses.py or atoms.py

**Impact:** Medium | **Effort:** Medium

---

## 7. Animated flow direction

**Problem:** Data-flow direction is only visible from arrowheads, which are tiny.

**Fix:**
- Add subtle animated dashes on data-flow edges (CSS animation on stroke-dashoffset) so data appears to "flow" from source to target
- Hierarchy edges remain static dotted gray

**Impact:** Low-medium | **Effort:** Low

---

## 8. Side-by-side CDG comparison

**Problem:** The isomorphism audit found meaningful structural matches (spo2_perfusion <-> stats_distributions at Jaccard=0.80). Users have no way to see this.

**Fix:**
- Add a "Compare" button that splits the canvas into left/right panes
- User selects two CDGs; both render with the same layout
- Nodes with matching concept_type signatures get linked by dashed cross-lines
- Show Jaccard similarity score and diff summary at the top

**Impact:** Low | **Effort:** High

---

## Implementation order

| Phase | Items | Rationale |
|-------|-------|-----------|
| Phase 1 | 2, 3, 4 | Low effort, high impact. Makes existing graphs immediately more useful. |
| Phase 2 | 1 | The single most impactful change but requires non-trivial graph state management. |
| Phase 3 | 5, 6 | Medium effort polish for browsing and inspection. |
| Phase 4 | 7, 8 | Nice-to-have. Animated flow is easy; comparison mode is a bigger project. |
