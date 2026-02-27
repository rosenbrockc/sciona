# CDG Visualizer

Browser-based network visualization for Conceptual Dependency Graphs stored in Memgraph or exported as JSON by `ageom decompose`.

## Starting the visualizer

### API mode (recommended)

Connects to Memgraph and serves the full UI with CDG browsing, compare mode, and isomorphism search.

```bash
# Make sure Memgraph is running
docker compose up -d memgraph

# Start the visualizer
ageom visualize --api
```

Opens `http://127.0.0.1:8080` in your default browser. Press `Ctrl+C` to stop.

### File mode

Load a single CDG JSON file without Memgraph.

```bash
# Pre-load a CDG file
ageom visualize cdg.json

# Specific port
ageom visualize cdg.json --port 9000

# Open file:// directly (no server, drag-drop only)
ageom visualize --no-serve
```

### CLI options

```
ageom visualize [cdg_file] [--port PORT] [--no-serve] [--api]
```

| Option | Description |
|--------|-------------|
| `cdg_file` | Path to a CDG JSON file to pre-load (optional) |
| `--port PORT` | HTTP server port. Default `0` picks a random port; `--api` defaults to `8080` |
| `--no-serve` | Open `file://` directly, no HTTP server. Drag-drop still works |
| `--api` | Start FastAPI server with Memgraph-backed CDG browsing |

## Reading the graph

### Node color = concept type family

Color encodes which domain family a node's `concept_type` belongs to.

| Color | Family | Concept types |
|-------|--------|---------------|
| Blue | Math / Algo | sorting, searching, divide\_and\_conquer, greedy, dynamic\_programming, combinatorics, algebra, analysis, arithmetic, number\_theory, geometry, set\_theory |
| Purple | Probabilistic | sampler, log\_prob, posterior\_update, variational\_inference, prior\_init, prior\_distribution, likelihood\_evaluation, probabilistic\_oracle, oracle\_gradient, mcmc\_kernel, mcmc\_proposal, vi\_elbo, conjugate\_update |
| Teal | Signal | signal\_filter, signal\_transform, graph\_signal\_processing, sequential\_filter, smc\_reweight |
| Amber | Orchestration | state\_init, data\_assembly, conditional\_routing, data\_extraction |
| Green | Presentation | visualization, observability |
| Gray | Other | custom, external\_tool, message\_passing, neural\_network |

### Node shape = status

| Shape | Status |
|-------|--------|
| Ellipse (circle) | `atomic` — leaf node, maps to a known primitive |
| Rounded rectangle | `decomposed` — has children |
| Diamond | `external` — external tool dependency |
| Ellipse | `pending` — not yet decomposed |
| Cut-rectangle | `rejected` — critic rejected this decomposition |
| Triangle | `high_risk` — requires novel proof |

### Edge types

- **Dotted gray arrows** — hierarchy edges (parent to child)
- **Dashed dark arrows** with animated flow — data-flow edges labeled `output_name -> input_name`
- **Dashed orange arrows** — data-flow edges that require glue code (`requires_glue=true`)

Hierarchy edges are suppressed where a data-flow edge already connects the same pair.

### Node size

Scales with child count: `min(80, 40 + childCount * 8)` pixels.

## Features

### CDG browser

*Requires `--api` mode.*

Click **Browse CDGs** in the toolbar to open the slide-out browser panel. CDGs are grouped by namespace (the repo path prefix). Each entry shows node count and a mini concept-type bar chart. Type in the search box to filter by repo name. Click a CDG to load it.

### Collapse and expand

Decomposed nodes show a `[N]` badge indicating their child count. **Double-click** a decomposed node to expand it and reveal its children inline. Double-click again to collapse. A breadcrumb trail appears at the top showing the current expansion path — click any breadcrumb to collapse back to that level. Click **Overview** to collapse everything.

### Node search

The toolbar search box supports free-text and structured queries:

| Query | Effect |
|-------|--------|
| `kalman` | Matches node name or description containing "kalman" |
| `type:sampler` | Matches nodes whose concept\_type contains "sampler" |
| `status:atomic` | Matches nodes with status "atomic" |
| `depth:2` | Matches nodes at depth 2 |
| `depth:>1` | Matches nodes deeper than 1 |
| `type:signal filter` | Combines structured and free-text filters |

Matching nodes are highlighted; all others dim. Press **Enter** to cycle through matches. Press **Escape** to clear.

### Detail panel

Click any node to open the right sidebar. Click the canvas background to close it.

**Summary tab** — status badge, concept type badge (colored by family), description, type signature, matched primitive, internal metadata (depth, children, parent, decomposition rationale), and critic agent notes.

**Ports tab** — input and output port tables showing name, type descriptor, and constraints.

**Lineage tab** — upstream (sources) and downstream (targets) data-flow neighbors, up to 3 hops deep. Click any lineage entry to navigate to that node.

**Isomorphisms tab** — results from a subgraph similarity search (see below).

### Hover highlighting

Hover over a node to highlight its full upstream chain (blue) and downstream chain (orange). All other elements dim. Hovering an edge shows the full port-to-port type annotation.

### Legend

Click **Legend** in the toolbar to toggle a floating panel showing color-to-family and shape-to-status mappings.

### Compare mode

*Requires `--api` mode.*

Click **Compare** in the toolbar to enter side-by-side comparison. Two dropdowns let you pick any CDG from Memgraph for the left and right panes. Each pane renders an independent Cytoscape graph. A **Jaccard similarity** score (concept-type multiset) is computed and displayed in the compare bar. Click **Exit Compare** to return to the main view.

### Isomorphism search

*Requires `--api` mode.*

Find structurally similar subgraphs across all CDGs in Memgraph.

1. Click a node (decomposed, or child of a decomposed node)
2. Click the **Isomorphisms** button that appears next to the node name
3. Configure the search in the options modal:
   - **Scope** — "This node" uses the selected node (or its nearest decomposed ancestor if atomic); "Parent" walks up one additional level
   - **Min similarity** — Jaccard threshold slider (0.0 to 1.0, default 0.3)
   - **Max results** — cap on returned results (default 20)
   - **Layers** — toggle which retrieval layers to run:
     - **Topo-hash** — exact degree-sequence match (score 1.0)
     - **Structure** — same concept\_type, port arity within ±1 (score ~0.56–0.70)
     - **Jaccard** — concept-type multiset Jaccard above the threshold
4. Click **Search**

Results appear in the **Isomorphisms tab** with:
- Score bar (green ≥ 0.8, orange ≥ 0.5, red below)
- Layer badge (topo / struct / jaccard)
- Repo name and mini concept-type bar for the candidate's children

**Click any result row** to open it in compare mode — the current CDG loads on the left, the matched CDG on the right.

### Drag and drop

Drag a `.json` file anywhere onto the page to load it. Works in all modes including `file://`.

## API endpoints

*Available in `--api` mode.*

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/cdgs` | List all CDGs with node counts, concept types, statuses. Supports `?q=`, `?concept_type=`, `?status=` query params |
| `GET` | `/api/cdgs/{repo}` | Full CDG JSON (nodes + edges + metadata) for a repo |
| `POST` | `/api/isomorphisms` | Find similar subgraphs. Body: `{"repo", "node_id", "radius", "min_jaccard", "max_results", "layers"}` |

Example:

```bash
# List CDGs
curl http://localhost:8080/api/cdgs

# Fetch a specific CDG
curl http://localhost:8080/api/cdgs/hpy-atoms%2Fspo2_perfusion

# Search for similar subgraphs
curl -X POST http://localhost:8080/api/isomorphisms \
  -H "Content-Type: application/json" \
  -d '{"repo": "hpy-atoms/spo2_perfusion", "node_id": "HPYSpO2Perfusion_root"}'
```

## CDG JSON format

The visualizer accepts JSON in the format produced by `ageom decompose --output`:

```json
{
  "nodes": [
    {
      "node_id": "root",
      "parent_id": null,
      "name": "Merge Sort",
      "description": "Sort an array using merge sort",
      "concept_type": "divide_and_conquer",
      "status": "decomposed",
      "children": ["split", "merge"],
      "depth": 0,
      "type_signature": "list[int] -> list[int]",
      "inputs": [{"name": "arr", "type_desc": "list[int]", "constraints": ""}],
      "outputs": [{"name": "sorted", "type_desc": "list[int]", "constraints": ""}],
      "critic_notes": "",
      "decomposition_rationale": "Classic divide-and-conquer decomposition"
    }
  ],
  "edges": [
    {
      "source_id": "split",
      "target_id": "merge",
      "output_name": "halves",
      "input_name": "parts",
      "source_type": "list[int]",
      "target_type": "list[int]",
      "requires_glue": false
    }
  ],
  "metadata": {
    "goal": "Implement merge sort",
    "paradigm": "divide_and_conquer",
    "repo": "my-project/merge_sort",
    "thread_id": "a1b2c3d4..."
  }
}
```

Only `nodes` (array) is required. `edges` defaults to `[]` if absent. `metadata` is optional and populates the header bar.

## Loading without the CLI

Open `ageom/static/index.html` directly in a browser and drag-drop a CDG JSON file. No server or installation required — Cytoscape.js loads from a CDN. The CDG browser, compare mode, and isomorphism search are unavailable without the API server.
