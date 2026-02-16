# CDG Visualization

Browser-based network visualization for Conceptual Dependency Graphs produced by `ageom decompose`.

## Quick start

```bash
# Generate a CDG
ageom decompose "Implement merge sort" --no-persist --output cdg.json

# Open the visualizer (starts a local server and opens your browser)
ageom visualize cdg.json
```

Press `Ctrl+C` to stop the server when you're done.

## CLI options

```
ageom visualize [cdg_file] [--port PORT] [--no-serve]
```

| Option | Description |
|--------|-------------|
| `cdg_file` | Path to a CDG JSON file to pre-load (optional — you can also drag-drop in the browser) |
| `--port PORT` | HTTP server port. Default `0` picks a random available port |
| `--no-serve` | Skip the HTTP server and open `file://` directly. Drag-drop still works, but auto-loading a CDG file does not |

### Examples

```bash
# Auto-pick port, pre-load a CDG
ageom visualize /tmp/cdg.json

# Use a specific port
ageom visualize cdg.json --port 8080

# Open the HTML directly (no server)
ageom visualize --no-serve
```

## Reading the graph

### Node color = lifecycle status

| Color | Status | Meaning |
|-------|--------|---------|
| Gray | `pending` | Not yet decomposed |
| Blue | `decomposed` | Has children |
| Green | `atomic` | Leaf — maps to a known primitive |
| Red | `rejected` | Critic rejected this decomposition |
| Amber | `high_risk` | Requires novel proof, flagged by critic |

### Node shape = concept type

Each of the 16 `ConceptType` values maps to a distinct Cytoscape shape:

| Shape | Concept type |
|-------|-------------|
| Hexagon | divide_and_conquer |
| Diamond | searching |
| Round-rectangle | sorting |
| Triangle | greedy |
| Vee | dynamic_programming |
| Pentagon | graph_traversal |
| Octagon | graph_optimization |
| Rhomboid | string_matching |
| Star | geometry |
| Round-diamond | arithmetic |
| Round-triangle | number_theory |
| Round-pentagon | combinatorics |
| Round-hexagon | algebra |
| Round-octagon | analysis |
| Concave-hexagon | set_theory |
| Ellipse | custom |

### Edge types

- **Dotted gray arrows** — hierarchy edges (parent → child relationship)
- **Solid dark arrows** — data-flow edges with `"output → input"` labels
- **Dashed orange arrows** — data-flow edges that require glue code (`requires_glue=true`)

Where a data-flow edge already connects two nodes, the hierarchy edge between them is suppressed to avoid double lines.

### Node size

Nodes with more children are drawn larger: `min(80, 40 + childCount * 8)` pixels.

## Interacting with the graph

### Toolbar controls

- **Open File...** — load a CDG JSON via file dialog
- **Layout selector** — switch between Hierarchical (dagre, default), Force-directed (cose), and Breadthfirst layouts
- **Fit** — zoom to fit all nodes in view
- **Reset** — re-run the current layout and fit

### Detail panel

Click any node to open the side panel showing:

- Status (color-coded badge)
- Concept type
- Full description
- Type signature
- Matched primitive (if atomic)
- Inputs/outputs tables (name, type, constraints)
- Internal metadata (depth, children, parent, decomposition rationale)
- Critic agent notes

Click the background to close the panel.

### Drag and drop

Drag a `.json` file anywhere onto the page to load it. This works in both served and `file://` modes.

## Loading without the CLI

You can open `ageom/static/index.html` directly in a browser and drag-drop a CDG JSON file. No server or installation required — the page loads Cytoscape.js from a CDN.

## CDG JSON format

The visualizer expects the JSON format produced by `ageom decompose --output`:

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
      "matched_primitive": null,
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
    "thread_id": "a1b2c3d4...",
    "num_nodes": 3,
    "num_edges": 1
  }
}
```

Only `nodes` (array) is strictly required. `edges` defaults to an empty array if absent. `metadata` is optional and populates the header bar.
