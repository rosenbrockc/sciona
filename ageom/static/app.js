(function () {
  "use strict";

  // --- Concept-type color families ---

  var CONCEPT_FAMILY = {
    // Math/algo → Blue
    sorting: "math", searching: "math", divide_and_conquer: "math",
    greedy: "math", dynamic_programming: "math", combinatorics: "math",
    algebra: "math", analysis: "math", arithmetic: "math",
    number_theory: "math", geometry: "math", set_theory: "math",
    // Probabilistic → Purple
    sampler: "prob", log_prob: "prob", posterior_update: "prob",
    variational_inference: "prob", prior_init: "prob",
    prior_distribution: "prob", likelihood_evaluation: "prob",
    probabilistic_oracle: "prob", oracle_gradient: "prob",
    mcmc_kernel: "prob", mcmc_proposal: "prob", vi_elbo: "prob",
    conjugate_update: "prob",
    // Signal → Teal
    signal_filter: "signal", signal_transform: "signal",
    graph_signal_processing: "signal", sequential_filter: "signal",
    smc_reweight: "signal",
    // Orchestration → Amber
    state_init: "orch", data_assembly: "orch",
    conditional_routing: "orch", data_extraction: "orch",
    // Presentation → Green
    visualization: "pres", observability: "pres",
    // Other → Gray
    custom: "other", external_tool: "other",
    message_passing: "other", neural_network: "other"
  };

  var FAMILY_COLORS = {
    math:   { bg: "#bbdefb", border: "#1976d2", text: "#0d47a1" },
    prob:   { bg: "#e1bee7", border: "#8e24aa", text: "#4a148c" },
    signal: { bg: "#b2dfdb", border: "#00897b", text: "#004d40" },
    orch:   { bg: "#ffe0b2", border: "#f57c00", text: "#e65100" },
    pres:   { bg: "#c8e6c9", border: "#43a047", text: "#1b5e20" },
    other:  { bg: "#e0e0e0", border: "#757575", text: "#424242" }
  };

  var FAMILY_LABELS = {
    math: "Math / Algo",
    prob: "Probabilistic",
    signal: "Signal",
    orch: "Orchestration",
    pres: "Presentation",
    other: "Other"
  };

  // Shape encodes status (only 3 values)
  var STATUS_SHAPES = {
    atomic:     "ellipse",
    decomposed: "round-rectangle",
    external:   "diamond",
    pending:    "ellipse",
    rejected:   "cut-rectangle",
    high_risk:  "triangle"
  };

  // --- DOM references ---

  var metaGoal = document.getElementById("meta-goal");
  var metaParadigm = document.getElementById("meta-paradigm");
  var metaNodes = document.getElementById("meta-nodes");
  var metaEdges = document.getElementById("meta-edges");
  var metaThread = document.getElementById("meta-thread");
  var statusText = document.getElementById("status-text");
  var btnOpen = document.getElementById("btn-open");
  var btnFit = document.getElementById("btn-fit");
  var btnReset = document.getElementById("btn-reset");
  var layoutSelect = document.getElementById("layout-select");
  var fileInput = document.getElementById("file-input");
  var cyContainer = document.getElementById("cy-container");
  var dropZone = document.getElementById("drop-zone");
  var detailPanel = document.getElementById("detail-panel");
  var graphSearch = document.getElementById("graph-search");
  var legendPanel = document.getElementById("legend-panel");
  var btnLegend = document.getElementById("btn-legend");

  var breadcrumbBar = document.getElementById("breadcrumb-bar");
  var breadcrumbContent = document.getElementById("breadcrumb-content");

  var cy = null;
  var currentData = null;

  // --- Collapse/expand state ---
  // Set of node_ids that are currently expanded (their children are visible)
  var expandedNodes = {};
  // Lookup maps built from currentData
  var nodeById = {};
  var childrenOf = {};    // parent_id -> [child node_ids]
  var parentOf = {};      // node_id -> parent_id
  var breadcrumbPath = []; // stack of expanded node_ids for breadcrumb

  // --- Search state ---
  var searchMatches = [];
  var searchIndex = -1;

  // --- Data loading ---

  function handleFile(file) {
    var reader = new FileReader();
    reader.onload = function (e) {
      try {
        var data = JSON.parse(e.target.result);
        validateAndLoad(data);
      } catch (err) {
        statusText.textContent = "Error: invalid JSON — " + err.message;
      }
    };
    reader.readAsText(file);
  }

  function validateAndLoad(data) {
    if (!data.nodes || !Array.isArray(data.nodes)) {
      statusText.textContent = "Error: JSON must contain a 'nodes' array";
      return;
    }
    if (!data.edges || !Array.isArray(data.edges)) {
      data.edges = [];
    }
    dropZone.classList.add("hidden");
    currentData = data;
    buildGraph(data);
  }

  function tryLoadDefault() {
    fetch("default_cdg.json")
      .then(function (res) {
        if (!res.ok) throw new Error("not found");
        return res.json();
      })
      .then(function (data) {
        validateAndLoad(data);
      })
      .catch(function () {
        // Silently ignore — user can drag-drop or use Open File
      });
  }

  // --- Drag and drop ---

  document.body.addEventListener("dragover", function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("drag-active");
  });

  document.body.addEventListener("dragleave", function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("drag-active");
  });

  document.body.addEventListener("drop", function (e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("drag-active");
    var files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  // --- File dialog ---

  btnOpen.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
      fileInput.value = "";
    }
  });

  // --- Helper: get node color from concept_type ---

  function getNodeColors(conceptType) {
    var family = CONCEPT_FAMILY[conceptType] || "other";
    return FAMILY_COLORS[family] || FAMILY_COLORS.other;
  }

  // --- Graph construction ---

  function buildIndexes(data) {
    nodeById = {};
    childrenOf = {};
    parentOf = {};
    data.nodes.forEach(function (node) {
      nodeById[node.node_id] = node;
      if (node.children && node.children.length > 0) {
        childrenOf[node.node_id] = node.children.slice();
      }
      if (node.parent_id) {
        parentOf[node.node_id] = node.parent_id;
      }
    });
  }

  function isNodeVisible(nodeId) {
    // A node is visible if it has no parent (root/depth-0),
    // or its parent is expanded
    var pid = parentOf[nodeId];
    if (!pid) return true;
    return !!expandedNodes[pid];
  }

  function isDecomposed(nodeId) {
    return childrenOf[nodeId] && childrenOf[nodeId].length > 0;
  }

  function getDescendants(nodeId) {
    var result = [];
    var stack = childrenOf[nodeId] ? childrenOf[nodeId].slice() : [];
    while (stack.length > 0) {
      var id = stack.pop();
      result.push(id);
      if (childrenOf[id]) {
        stack = stack.concat(childrenOf[id]);
      }
    }
    return result;
  }

  function updateBreadcrumb() {
    if (breadcrumbPath.length === 0) {
      breadcrumbBar.classList.add("hidden");
      return;
    }
    breadcrumbBar.classList.remove("hidden");
    breadcrumbContent.innerHTML = "";

    // Root link
    var rootLink = document.createElement("span");
    rootLink.className = "breadcrumb-link";
    rootLink.textContent = "Overview";
    rootLink.addEventListener("click", function () {
      collapseAll();
    });
    breadcrumbContent.appendChild(rootLink);

    breadcrumbPath.forEach(function (nodeId, idx) {
      var sep = document.createElement("span");
      sep.className = "breadcrumb-sep";
      sep.textContent = ">";
      breadcrumbContent.appendChild(sep);

      var node = nodeById[nodeId];
      var name = node ? node.name : nodeId;

      if (idx < breadcrumbPath.length - 1) {
        var link = document.createElement("span");
        link.className = "breadcrumb-link";
        link.textContent = name;
        link.addEventListener("click", (function (nid, depth) {
          return function () {
            collapseTo(nid, depth);
          };
        })(nodeId, idx));
        breadcrumbContent.appendChild(link);
      } else {
        var current = document.createElement("span");
        current.className = "breadcrumb-current";
        current.textContent = name;
        breadcrumbContent.appendChild(current);
      }
    });
  }

  function collapseAll() {
    expandedNodes = {};
    breadcrumbPath = [];
    rebuildVisibleGraph();
  }

  function collapseTo(nodeId, depth) {
    // Collapse everything deeper than depth
    breadcrumbPath = breadcrumbPath.slice(0, depth + 1);
    var keep = {};
    breadcrumbPath.forEach(function (id) { keep[id] = true; });
    Object.keys(expandedNodes).forEach(function (id) {
      if (!keep[id]) delete expandedNodes[id];
    });
    rebuildVisibleGraph();
  }

  function toggleExpand(nodeId) {
    if (expandedNodes[nodeId]) {
      // Collapse: remove this and all descendants from expanded
      delete expandedNodes[nodeId];
      var desc = getDescendants(nodeId);
      desc.forEach(function (id) { delete expandedNodes[id]; });
      // Update breadcrumb
      var idx = breadcrumbPath.indexOf(nodeId);
      if (idx >= 0) breadcrumbPath = breadcrumbPath.slice(0, idx);
    } else {
      // Expand
      expandedNodes[nodeId] = true;
      // Rebuild breadcrumb path from ancestry
      breadcrumbPath = [];
      var cur = nodeId;
      while (cur) {
        if (expandedNodes[cur]) breadcrumbPath.unshift(cur);
        cur = parentOf[cur];
      }
    }
    rebuildVisibleGraph();
  }

  function rebuildVisibleGraph() {
    if (!cy || !currentData) return;

    var data = currentData;

    // Determine which nodes are visible
    var visibleNodeIds = {};
    data.nodes.forEach(function (node) {
      if (isNodeVisible(node.node_id)) {
        visibleNodeIds[node.node_id] = true;
      }
    });

    // Show/hide nodes
    cy.nodes().forEach(function (n) {
      if (visibleNodeIds[n.id()]) {
        n.removeClass("collapsed-hidden");
      } else {
        n.addClass("collapsed-hidden");
      }
      // Update label for decomposed nodes to show child count
      var nd = n.data("_nodeData");
      if (nd && isDecomposed(n.id())) {
        if (expandedNodes[n.id()]) {
          n.data("label", nd.name + " [-]");
        } else {
          n.data("label", nd.name + " [" + childrenOf[n.id()].length + "]");
        }
      }
    });

    // Show/hide edges
    cy.edges().forEach(function (e) {
      var src = e.data("source");
      var tgt = e.data("target");
      if (visibleNodeIds[src] && visibleNodeIds[tgt]) {
        e.removeClass("collapsed-hidden");
      } else {
        e.addClass("collapsed-hidden");
      }
    });

    updateBreadcrumb();

    // Re-layout only visible elements
    var visible = cy.elements().not(".collapsed-hidden");
    if (visible.length > 0) {
      visible.layout(getLayoutConfig(layoutSelect.value)).run();
    }

    // Update status
    var visCount = Object.keys(visibleNodeIds).length;
    statusText.textContent = visCount + " of " + data.nodes.length + " nodes visible";
  }

  function buildGraph(data) {
    if (cy) {
      cy.destroy();
      cy = null;
    }

    // Reset collapse state
    expandedNodes = {};
    breadcrumbPath = [];

    // Build indexes
    buildIndexes(data);

    // Clear search state
    searchMatches = [];
    searchIndex = -1;
    if (graphSearch) graphSearch.value = "";

    // Update metadata bar
    var meta = data.metadata || {};
    metaGoal.textContent = "Goal: " + (meta.goal || "—");
    metaParadigm.textContent = "Paradigm: " + (meta.paradigm || "—");
    metaNodes.textContent = "Nodes: " + data.nodes.length;
    metaEdges.textContent = "Edges: " + data.edges.length;
    metaThread.textContent = "Thread: " + (meta.thread_id ? meta.thread_id.substring(0, 12) : "—");

    // Build ALL node elements (visibility controlled by classes)
    var elements = [];
    var dataFlowPairs = {};

    // Index data-flow edges to suppress duplicate hierarchy edges
    data.edges.forEach(function (edge) {
      var key = edge.source_id + "->" + edge.target_id;
      dataFlowPairs[key] = true;
    });

    data.nodes.forEach(function (node) {
      var status = node.status || "pending";
      var conceptType = node.concept_type || "custom";
      var colors = getNodeColors(conceptType);
      var shape = STATUS_SHAPES[status] || "ellipse";
      var childCount = (node.children && node.children.length) || 0;
      var size = Math.min(80, 40 + childCount * 8);
      var visible = isNodeVisible(node.node_id);
      var label = node.name;
      if (isDecomposed(node.node_id)) {
        label = node.name + " [" + childrenOf[node.node_id].length + "]";
      }

      elements.push({
        group: "nodes",
        data: {
          id: node.node_id,
          label: label,
          _nodeData: node,
          bgColor: colors.bg,
          borderColor: colors.border,
          textColor: colors.text,
          shape: shape,
          size: size
        },
        classes: visible ? "" : "collapsed-hidden"
      });

      // Hierarchy edge from parent (dotted gray), suppressed if data-flow already covers it
      if (node.parent_id) {
        var pairKey = node.parent_id + "->" + node.node_id;
        var reversePairKey = node.node_id + "->" + node.parent_id;
        if (!dataFlowPairs[pairKey] && !dataFlowPairs[reversePairKey]) {
          var edgeVisible = visible && isNodeVisible(node.parent_id);
          elements.push({
            group: "edges",
            data: {
              id: "hier_" + node.parent_id + "_" + node.node_id,
              source: node.parent_id,
              target: node.node_id,
              edgeType: "hierarchy"
            },
            classes: edgeVisible ? "" : "collapsed-hidden"
          });
        }
      }
    });

    // Data-flow edges
    data.edges.forEach(function (edge, i) {
      var edgeLabel = edge.output_name + " \u2192 " + edge.input_name;
      var classes = edge.requires_glue ? "glue-edge" : "";
      var srcVisible = isNodeVisible(edge.source_id);
      var tgtVisible = isNodeVisible(edge.target_id);
      if (!srcVisible || !tgtVisible) {
        classes = classes ? classes + " collapsed-hidden" : "collapsed-hidden";
      }
      elements.push({
        group: "edges",
        data: {
          id: "df_" + i + "_" + edge.source_id + "_" + edge.target_id,
          source: edge.source_id,
          target: edge.target_id,
          label: edgeLabel,
          edgeType: "dataflow",
          requiresGlue: !!edge.requires_glue,
          sourceType: edge.source_type || "",
          targetType: edge.target_type || "",
          outputName: edge.output_name || "",
          inputName: edge.input_name || ""
        },
        classes: classes
      });
    });

    // Initialize Cytoscape
    cy = cytoscape({
      container: cyContainer,
      elements: elements,
      style: getCytoscapeStyle(),
      layout: { name: "preset" }, // delay layout
      wheelSensitivity: 0.3
    });

    // Layout only visible elements
    var visible = cy.elements().not(".collapsed-hidden");
    if (visible.length > 0) {
      visible.layout(getLayoutConfig(layoutSelect.value)).run();
    }

    // Interactions
    cy.on("tap", "node", onNodeTap);
    cy.on("tap", function (e) {
      if (e.target === cy) {
        detailPanel.classList.remove("visible");
      }
    });

    // Double-click to expand/collapse decomposed nodes
    cy.on("dbltap", "node", function (e) {
      var nodeId = e.target.id();
      if (isDecomposed(nodeId)) {
        toggleExpand(nodeId);
      }
    });

    // Hover-highlight upstream/downstream
    cy.on("mouseover", "node", onNodeMouseOver);
    cy.on("mouseout", "node", onNodeMouseOut);
    cy.on("mouseover", "edge", onEdgeMouseOver);
    cy.on("mouseout", "edge", onEdgeMouseOut);

    var visCount = data.nodes.filter(function (n) { return isNodeVisible(n.node_id); }).length;
    statusText.textContent = visCount + " of " + data.nodes.length + " nodes visible (double-click to expand)";
    updateBreadcrumb();
  }

  // --- Hover-highlight: upstream/downstream ---

  function onNodeMouseOver(e) {
    var node = e.target;
    var upstream = node.predecessors();
    var downstream = node.successors();
    var connected = upstream.union(downstream).union(node);

    cy.elements().not(connected).addClass("dimmed");
    upstream.edges().addClass("upstream-highlight");
    downstream.edges().addClass("downstream-highlight");
    upstream.nodes().addClass("upstream-highlight");
    downstream.nodes().addClass("downstream-highlight");
    node.addClass("hover-focus");
  }

  function onNodeMouseOut() {
    cy.elements().removeClass("dimmed upstream-highlight downstream-highlight hover-focus");
  }

  function onEdgeMouseOver(e) {
    var edge = e.target;
    var d = edge.data();
    if (d.edgeType === "dataflow" && d.outputName) {
      var tip = d.source + "." + d.outputName +
        (d.sourceType ? " (" + d.sourceType + ")" : "") +
        " \u2192 " + d.target + "." + d.inputName +
        (d.targetType ? " (" + d.targetType + ")" : "");
      edge.data("_tipLabel", tip);
      edge.addClass("edge-tooltip");
    }
  }

  function onEdgeMouseOut(e) {
    e.target.removeClass("edge-tooltip");
    e.target.removeData("_tipLabel");
  }

  // --- In-graph search ---

  function parseSearchQuery(raw) {
    var structured = {};
    var freeText = [];
    var tokens = raw.trim().split(/\s+/);
    tokens.forEach(function (tok) {
      var m = tok.match(/^(type|status|depth):(.+)$/i);
      if (m) {
        structured[m[1].toLowerCase()] = m[2];
      } else {
        freeText.push(tok.toLowerCase());
      }
    });
    return { structured: structured, freeText: freeText.join(" ") };
  }

  function nodeMatchesQuery(nodeData, query) {
    // Structured filters
    if (query.structured.type) {
      var ct = (nodeData.concept_type || "").toLowerCase();
      if (ct.indexOf(query.structured.type.toLowerCase()) === -1) return false;
    }
    if (query.structured.status) {
      var st = (nodeData.status || "").toLowerCase();
      if (st !== query.structured.status.toLowerCase()) return false;
    }
    if (query.structured.depth) {
      var depthStr = query.structured.depth;
      var nodeDepth = nodeData.depth != null ? nodeData.depth : -1;
      if (depthStr.charAt(0) === ">") {
        if (nodeDepth <= parseInt(depthStr.substring(1), 10)) return false;
      } else if (depthStr.charAt(0) === "<") {
        if (nodeDepth >= parseInt(depthStr.substring(1), 10)) return false;
      } else {
        if (nodeDepth !== parseInt(depthStr, 10)) return false;
      }
    }
    // Free text match on name + description
    if (query.freeText) {
      var haystack = ((nodeData.name || "") + " " + (nodeData.description || "")).toLowerCase();
      if (haystack.indexOf(query.freeText) === -1) return false;
    }
    return true;
  }

  function runSearch(raw) {
    if (!cy) return;

    searchMatches = [];
    searchIndex = -1;

    if (!raw || !raw.trim()) {
      cy.elements().removeClass("search-match search-dimmed");
      statusText.textContent = (currentData ? currentData.nodes.length + " nodes, " + currentData.edges.length + " data-flow edges" : "No data loaded");
      return;
    }

    var query = parseSearchQuery(raw);
    var matchIds = [];

    cy.nodes().forEach(function (node) {
      var nd = node.data("_nodeData");
      if (nd && nodeMatchesQuery(nd, query)) {
        matchIds.push(node.id());
      }
    });

    if (matchIds.length === 0) {
      cy.elements().addClass("search-dimmed");
      statusText.textContent = "No matches";
      return;
    }

    var matchSelector = matchIds.map(function (id) { return 'node[id="' + id + '"]'; }).join(", ");
    var matchNodes = cy.nodes(matchSelector);

    cy.elements().addClass("search-dimmed");
    matchNodes.removeClass("search-dimmed").addClass("search-match");
    // Also un-dim edges between matches
    matchNodes.edgesWith(matchNodes).removeClass("search-dimmed");

    searchMatches = matchIds;
    searchIndex = 0;
    statusText.textContent = matchIds.length + " match" + (matchIds.length === 1 ? "" : "es");

    // Pan to first match
    panToSearchMatch();
  }

  function panToSearchMatch() {
    if (!cy || searchMatches.length === 0) return;
    var nodeId = searchMatches[searchIndex];
    var node = cy.getElementById(nodeId);
    if (node.length) {
      cy.animate({
        center: { eles: node },
        zoom: Math.max(cy.zoom(), 1.2)
      }, { duration: 200 });
    }
  }

  function cycleSearchMatch() {
    if (searchMatches.length <= 1) return;
    searchIndex = (searchIndex + 1) % searchMatches.length;
    statusText.textContent = "Match " + (searchIndex + 1) + " of " + searchMatches.length;
    panToSearchMatch();
  }

  if (graphSearch) {
    var searchDebounce = null;
    graphSearch.addEventListener("input", function () {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(function () {
        runSearch(graphSearch.value);
      }, 200);
    });

    graphSearch.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        if (searchMatches.length > 0) {
          cycleSearchMatch();
        } else {
          runSearch(graphSearch.value);
        }
      }
      if (e.key === "Escape") {
        graphSearch.value = "";
        runSearch("");
        graphSearch.blur();
      }
    });
  }

  // --- Legend toggle ---

  if (btnLegend) {
    btnLegend.addEventListener("click", function () {
      legendPanel.classList.toggle("visible");
    });
  }

  // Build legend content dynamically
  function buildLegend() {
    var container = document.getElementById("legend-content");
    if (!container) return;
    container.innerHTML = "";

    // Color families
    var colorTitle = document.createElement("div");
    colorTitle.className = "legend-group-title";
    colorTitle.textContent = "Color = Concept Type Family";
    container.appendChild(colorTitle);

    Object.keys(FAMILY_COLORS).forEach(function (key) {
      var row = document.createElement("div");
      row.className = "legend-row";
      var swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = FAMILY_COLORS[key].bg;
      swatch.style.borderColor = FAMILY_COLORS[key].border;
      var label = document.createElement("span");
      label.className = "legend-label";
      label.textContent = FAMILY_LABELS[key] || key;
      row.appendChild(swatch);
      row.appendChild(label);
      container.appendChild(row);
    });

    // Shape = status
    var shapeTitle = document.createElement("div");
    shapeTitle.className = "legend-group-title";
    shapeTitle.style.marginTop = "12px";
    shapeTitle.textContent = "Shape = Status";
    container.appendChild(shapeTitle);

    var shapeLabels = {
      atomic: "Atomic (circle)",
      decomposed: "Decomposed (rounded rect)",
      external: "External (diamond)"
    };
    Object.keys(shapeLabels).forEach(function (key) {
      var row = document.createElement("div");
      row.className = "legend-row";
      var icon = document.createElement("span");
      icon.className = "legend-shape legend-shape-" + key;
      var label = document.createElement("span");
      label.className = "legend-label";
      label.textContent = shapeLabels[key];
      row.appendChild(icon);
      row.appendChild(label);
      container.appendChild(row);
    });
  }

  buildLegend();

  // --- Cytoscape stylesheet ---

  function getCytoscapeStyle() {
    return [
      // Nodes
      {
        selector: "node",
        style: {
          "label": "data(label)",
          "text-valign": "center",
          "text-halign": "center",
          "text-wrap": "wrap",
          "text-max-width": "70px",
          "font-size": "10px",
          "color": "data(textColor)",
          "background-color": "data(bgColor)",
          "border-color": "data(borderColor)",
          "border-width": 2,
          "shape": "data(shape)",
          "width": "data(size)",
          "height": "data(size)",
          "transition-property": "opacity, border-width",
          "transition-duration": "0.15s"
        }
      },
      // Selected node
      {
        selector: "node:selected",
        style: {
          "border-width": 4,
          "overlay-opacity": 0.15,
          "overlay-color": "#42a5f5"
        }
      },
      // Dimmed (hover-highlight)
      {
        selector: ".dimmed",
        style: {
          "opacity": 0.15
        }
      },
      // Upstream highlight
      {
        selector: "edge.upstream-highlight",
        style: {
          "line-color": "#1976d2",
          "target-arrow-color": "#1976d2",
          "width": 3,
          "opacity": 1,
          "z-index": 10
        }
      },
      {
        selector: "node.upstream-highlight",
        style: {
          "border-width": 3,
          "opacity": 1,
          "z-index": 10
        }
      },
      // Downstream highlight
      {
        selector: "edge.downstream-highlight",
        style: {
          "line-color": "#e65100",
          "target-arrow-color": "#e65100",
          "width": 3,
          "opacity": 1,
          "z-index": 10
        }
      },
      {
        selector: "node.downstream-highlight",
        style: {
          "border-width": 3,
          "opacity": 1,
          "z-index": 10
        }
      },
      // Hover focus node
      {
        selector: ".hover-focus",
        style: {
          "border-width": 4,
          "opacity": 1,
          "z-index": 20
        }
      },
      // Edge tooltip (shows full label on hover)
      {
        selector: "edge.edge-tooltip",
        style: {
          "width": 3,
          "z-index": 15
        }
      },
      // Search dimmed
      {
        selector: ".search-dimmed",
        style: {
          "opacity": 0.12
        }
      },
      // Search match
      {
        selector: ".search-match",
        style: {
          "border-width": 4,
          "border-color": "#ff6f00",
          "opacity": 1,
          "z-index": 10
        }
      },
      // Collapsed-hidden (collapse/expand)
      {
        selector: ".collapsed-hidden",
        style: {
          "display": "none"
        }
      },
      // Hierarchy edges
      {
        selector: "edge[edgeType='hierarchy']",
        style: {
          "width": 1.5,
          "line-style": "dotted",
          "line-color": "#b0bec5",
          "target-arrow-color": "#b0bec5",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "arrow-scale": 0.8
        }
      },
      // Data-flow edges (animated dashes show flow direction)
      {
        selector: "edge[edgeType='dataflow']",
        style: {
          "width": 2,
          "line-style": "dashed",
          "line-dash-pattern": [6, 3],
          "line-dash-offset": 0,
          "line-color": "#546e7a",
          "target-arrow-color": "#546e7a",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "label": "data(label)",
          "font-size": "9px",
          "text-rotation": "autorotate",
          "color": "#37474f",
          "text-background-color": "#fff",
          "text-background-opacity": 0.8,
          "text-background-padding": "2px",
          "arrow-scale": 1
        }
      },
      // Glue edges
      {
        selector: ".glue-edge",
        style: {
          "line-style": "dashed",
          "line-color": "#ffa726",
          "target-arrow-color": "#ffa726"
        }
      }
    ];
  }

  // --- Layout configs ---

  function getLayoutConfig(name) {
    if (name === "dagre") {
      return {
        name: "dagre",
        rankDir: "TB",
        nodeSep: 50,
        rankSep: 80,
        animate: true,
        animationDuration: 300
      };
    }
    if (name === "cose") {
      return {
        name: "cose",
        nodeRepulsion: function () { return 8000; },
        animate: true,
        animationDuration: 500
      };
    }
    if (name === "breadthfirst") {
      return {
        name: "breadthfirst",
        directed: true,
        spacingFactor: 1.5,
        animate: true,
        animationDuration: 300
      };
    }
    // fallback
    return { name: "dagre", rankDir: "TB", nodeSep: 50, rankSep: 80 };
  }

  // --- Layout / toolbar controls ---

  layoutSelect.addEventListener("change", function () {
    if (cy) {
      cy.layout(getLayoutConfig(layoutSelect.value)).run();
    }
  });

  btnFit.addEventListener("click", function () {
    if (cy) {
      cy.fit(undefined, 30);
    }
  });

  btnReset.addEventListener("click", function () {
    if (cy) {
      cy.layout(getLayoutConfig(layoutSelect.value)).run();
      cy.fit(undefined, 30);
    }
  });

  // --- Detail panel tabs ---

  var detailTabs = document.querySelectorAll(".detail-tab");
  var tabContents = document.querySelectorAll(".tab-content");

  detailTabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      var target = tab.getAttribute("data-tab");
      detailTabs.forEach(function (t) { t.classList.remove("active"); });
      tabContents.forEach(function (c) { c.classList.remove("active"); });
      tab.classList.add("active");
      var content = document.getElementById("tab-" + target);
      if (content) content.classList.add("active");
    });
  });

  // --- Node tap / detail panel ---

  var selectedNodeId = null;

  function onNodeTap(e) {
    var nodeData = e.target.data("_nodeData");
    if (!nodeData) { console.warn("onNodeTap: no _nodeData"); return; }
    try {
      selectedNodeId = nodeData.node_id;
      populateDetailPanel(nodeData);
      populateLineage(nodeData.node_id);
      updateIsoButtonVisibility(nodeData);
    } catch (err) {
      console.error("onNodeTap error:", err);
    }
    detailPanel.classList.add("visible");
  }

  function populateDetailPanel(node) {
    document.getElementById("detail-name").textContent = node.name || "(unnamed)";

    // Status badge
    var statusEl = document.getElementById("detail-status");
    var status = node.status || "pending";
    statusEl.textContent = status;
    statusEl.className = "status-badge status-" + status;

    // Concept type badge (colored by family)
    var ct = node.concept_type || "custom";
    var ctEl = document.getElementById("detail-concept-type");
    var family = CONCEPT_FAMILY[ct] || "other";
    var fColors = FAMILY_COLORS[family];
    ctEl.textContent = ct.replace(/_/g, " ");
    ctEl.style.background = fColors.bg;
    ctEl.style.color = fColors.text;
    ctEl.style.borderColor = fColors.border;

    // Description
    document.getElementById("detail-description").textContent = node.description || "(none)";

    // Type signature
    document.getElementById("detail-type-sig").textContent = node.type_signature || "(none)";

    // Matched primitive
    document.getElementById("detail-primitive").textContent = node.matched_primitive || "(none)";

    // IO tables
    populateIOTable("detail-inputs", node.inputs || []);
    populateIOTable("detail-outputs", node.outputs || []);

    // Internal metadata
    document.getElementById("detail-depth").textContent = node.depth != null ? String(node.depth) : "—";
    var children = node.children && node.children.length > 0 ? node.children.join(", ") : "(leaf)";
    document.getElementById("detail-children").textContent = children;
    document.getElementById("detail-parent").textContent = node.parent_id || "(root)";
    document.getElementById("detail-rationale").textContent = node.decomposition_rationale || "(none)";

    // Critic notes
    document.getElementById("detail-critic").value = node.critic_notes || "(none)";
  }

  function populateIOTable(tableId, specs) {
    var tbody = document.getElementById(tableId).querySelector("tbody");
    tbody.innerHTML = "";
    if (specs.length === 0) {
      var row = document.createElement("tr");
      var td = document.createElement("td");
      td.setAttribute("colspan", "3");
      td.textContent = "(none)";
      td.style.color = "#9e9e9e";
      row.appendChild(td);
      tbody.appendChild(row);
      return;
    }
    specs.forEach(function (spec) {
      var row = document.createElement("tr");
      var tdName = document.createElement("td");
      tdName.textContent = spec.name || "";
      var tdType = document.createElement("td");
      tdType.textContent = spec.type_desc || "";
      var tdConstraints = document.createElement("td");
      tdConstraints.textContent = spec.constraints || "";
      row.appendChild(tdName);
      row.appendChild(tdType);
      row.appendChild(tdConstraints);
      tbody.appendChild(row);
    });
  }

  // --- Lineage panel ---

  function populateLineage(nodeId) {
    var upList = document.getElementById("lineage-upstream-list");
    var downList = document.getElementById("lineage-downstream-list");
    var hint = document.querySelector(".lineage-hint");
    if (!upList || !downList) return;
    upList.innerHTML = "";
    downList.innerHTML = "";
    if (hint) hint.style.display = "none";

    if (!cy) return;

    // Gather upstream neighbors up to 3 levels
    buildLineageTree(nodeId, "upstream", upList, 3);
    buildLineageTree(nodeId, "downstream", downList, 3);
  }

  function buildLineageTree(startId, direction, container, maxDepth) {
    var visited = {};
    visited[startId] = true;

    function getNeighbors(nid) {
      var cyNode = cy.getElementById(nid);
      if (!cyNode || cyNode.length === 0) return [];
      var edges;
      if (direction === "upstream") {
        edges = cyNode.incomers("edge[edgeType='dataflow']");
      } else {
        edges = cyNode.outgoers("edge[edgeType='dataflow']");
      }
      var results = [];
      edges.forEach(function (e) {
        var neighborId = direction === "upstream" ? e.data("source") : e.data("target");
        var edgeLabel = direction === "upstream"
          ? e.data("outputName") + " \u2192 " + e.data("inputName")
          : e.data("outputName") + " \u2192 " + e.data("inputName");
        results.push({ id: neighborId, edgeLabel: edgeLabel });
      });
      return results;
    }

    function renderLevel(nodeIds, depth) {
      if (depth > maxDepth) return;
      var nextLevel = [];
      nodeIds.forEach(function (entry) {
        if (visited[entry.id]) return;
        visited[entry.id] = true;

        var nd = nodeById[entry.id];
        var ct = nd ? (nd.concept_type || "custom") : "custom";
        var colors = getNodeColors(ct);

        var item = document.createElement("div");
        item.className = "lineage-item" + (depth > 1 ? " lineage-depth-" + depth : "");
        item.addEventListener("click", function () {
          var cyNode = cy.getElementById(entry.id);
          if (cyNode.length) {
            cy.animate({ center: { eles: cyNode } }, { duration: 200 });
            onNodeTap({ target: cyNode });
          }
        });

        var swatch = document.createElement("span");
        swatch.className = "lineage-item-swatch";
        swatch.style.background = colors.bg;
        swatch.style.borderColor = colors.border;

        var name = document.createElement("span");
        name.className = "lineage-item-name";
        name.textContent = nd ? nd.name : entry.id;

        var edgeInfo = document.createElement("span");
        edgeInfo.className = "lineage-item-edge";
        edgeInfo.textContent = entry.edgeLabel;

        item.appendChild(swatch);
        item.appendChild(name);
        item.appendChild(edgeInfo);
        container.appendChild(item);

        // Queue next level
        var neighbors = getNeighbors(entry.id);
        neighbors.forEach(function (n) {
          if (!visited[n.id]) nextLevel.push(n);
        });
      });

      if (nextLevel.length > 0) {
        renderLevel(nextLevel, depth + 1);
      }
    }

    var initial = getNeighbors(startId);
    if (initial.length === 0) {
      var empty = document.createElement("div");
      empty.className = "lineage-hint";
      empty.textContent = "(none)";
      container.appendChild(empty);
      return;
    }
    renderLevel(initial, 1);
  }

  // --- CDG Browser (API mode) ---

  var btnBrowse = document.getElementById("btn-browse");
  var cdgBrowser = document.getElementById("cdg-browser");
  var btnBrowserClose = document.getElementById("btn-browser-close");
  var browserSearch = document.getElementById("browser-search");
  var browserList = document.getElementById("browser-list");
  var apiAvailable = false;

  function fetchCDGList(filters) {
    var params = new URLSearchParams();
    if (filters && filters.q) params.set("q", filters.q);
    if (filters && filters.concept_type) params.set("concept_type", filters.concept_type);
    if (filters && filters.status) params.set("status", filters.status);

    var url = "/api/cdgs";
    var qs = params.toString();
    if (qs) url += "?" + qs;

    return fetch(url)
      .then(function (res) {
        if (!res.ok) throw new Error("API error " + res.status);
        return res.json();
      })
      .then(function (cdgs) {
        apiAvailable = true;
        btnBrowse.style.display = "";
        renderCDGList(cdgs);
        return cdgs;
      })
      .catch(function () {
        // API not available (e.g. file:// mode or static server)
        apiAvailable = false;
        btnBrowse.style.display = "none";
        cdgBrowser.classList.remove("visible");
      });
  }

  function renderCDGList(cdgs) {
    browserList.innerHTML = "";
    if (cdgs.length === 0) {
      browserList.innerHTML = '<div class="browser-empty">No CDGs found</div>';
      return;
    }

    // Group by repo (second path segment, e.g. "advancedvi" from "ageo-atoms/advancedvi/optimize")
    var groups = {};
    cdgs.forEach(function (cdg) {
      var parts = cdg.repo.split("/");
      // 3+ segments: org/repo/cdg → group by parts[1]
      // 2 segments: org/cdg → group by parts[1]
      // 1 segment: group by the name itself
      var key = parts.length >= 2 ? parts[1] : parts[0];
      if (!groups[key]) groups[key] = [];
      groups[key].push(cdg);
    });

    var sortedKeys = Object.keys(groups).sort();
    sortedKeys.forEach(function (ns) {
      var groupEl = document.createElement("div");
      groupEl.className = "browser-group";

      var countLabel = groups[ns].length === 1
        ? "1 CDG"
        : groups[ns].length + " CDGs";
      var header = document.createElement("div");
      header.className = "browser-group-header";
      header.innerHTML = '<span class="browser-group-arrow">&#9654;</span> ' +
        '<span class="browser-group-name">' + ns + '</span>' +
        '<span class="browser-group-count">' + countLabel + '</span>';
      header.addEventListener("click", function () {
        groupEl.classList.toggle("collapsed");
      });

      var items = document.createElement("div");
      items.className = "browser-group-items";

      groups[ns].forEach(function (cdg) {
        var item = document.createElement("div");
        item.className = "browser-item";
        item.addEventListener("click", function () {
          fetchCDG(cdg.repo);
        });

        var title = document.createElement("div");
        title.className = "browser-item-title";
        // Show only the CDG name, not the full repo path
        var displayName = cdg.repo.split("/").pop();
        title.textContent = displayName;

        var meta = document.createElement("div");
        meta.className = "browser-item-meta";

        // Node count
        var nodeCountSpan = document.createElement("span");
        nodeCountSpan.className = "node-count";
        nodeCountSpan.textContent = cdg.node_count + " nodes";
        meta.appendChild(nodeCountSpan);

        // Mini concept-type bar
        if (cdg.concept_types && cdg.concept_types.length > 0) {
          var barContainer = document.createElement("span");
          barContainer.className = "concept-bar";
          var familyCounts = {};
          cdg.concept_types.forEach(function (ct) {
            var fam = CONCEPT_FAMILY[ct] || "other";
            familyCounts[fam] = (familyCounts[fam] || 0) + 1;
          });
          var total = cdg.concept_types.length;
          Object.keys(familyCounts).forEach(function (fam) {
            var seg = document.createElement("span");
            seg.className = "concept-bar-seg";
            seg.style.width = Math.max(4, Math.round(familyCounts[fam] / total * 60)) + "px";
            seg.style.background = FAMILY_COLORS[fam].border;
            seg.title = FAMILY_LABELS[fam] + ": " + familyCounts[fam];
            barContainer.appendChild(seg);
          });
          meta.appendChild(barContainer);
        }

        item.appendChild(title);
        item.appendChild(meta);
        items.appendChild(item);
      });

      groupEl.appendChild(header);
      groupEl.appendChild(items);
      browserList.appendChild(groupEl);
    });
  }

  function fetchCDG(repo) {
    statusText.textContent = "Loading " + repo + "...";
    fetch("/api/cdg?repo=" + encodeURIComponent(repo))
      .then(function (res) {
        if (!res.ok) throw new Error("CDG not found");
        return res.json();
      })
      .then(function (data) {
        validateAndLoad(data);
        cdgBrowser.classList.remove("visible");
      })
      .catch(function (err) {
        statusText.textContent = "Error: " + err.message;
      });
  }

  // Browser panel toggle
  btnBrowse.addEventListener("click", function () {
    cdgBrowser.classList.toggle("visible");
    if (cdgBrowser.classList.contains("visible")) {
      fetchCDGList({ q: browserSearch.value || undefined });
      browserSearch.focus();
    }
  });

  btnBrowserClose.addEventListener("click", function () {
    cdgBrowser.classList.remove("visible");
  });

  var searchTimeout = null;
  browserSearch.addEventListener("input", function () {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(function () {
      fetchCDGList({ q: browserSearch.value || undefined });
    }, 300);
  });

  // --- Compare mode ---

  var btnCompare = document.getElementById("btn-compare");
  var compareBar = document.getElementById("compare-bar");
  var compareContainer = document.getElementById("compare-container");
  var compareLeftSelect = document.getElementById("compare-left-select");
  var compareRightSelect = document.getElementById("compare-right-select");
  var compareScore = document.getElementById("compare-score");
  var btnCompareClose = document.getElementById("btn-compare-close");
  var cyLeft = null;
  var cyRight = null;
  var compareMode = false;
  var compareCDGList = [];

  if (btnCompare) {
    btnCompare.addEventListener("click", function () {
      enterCompareMode();
    });
  }

  if (btnCompareClose) {
    btnCompareClose.addEventListener("click", function () {
      exitCompareMode();
    });
  }

  function enterCompareMode() {
    compareMode = true;
    compareBar.classList.remove("hidden");
    compareContainer.classList.remove("hidden");
    cyContainer.style.display = "none";
    detailPanel.classList.remove("visible");

    // Populate selects from cached CDG list
    fetch("/api/cdgs")
      .then(function (res) { return res.json(); })
      .then(function (cdgs) {
        compareCDGList = cdgs;
        populateCompareSelects(cdgs);
      })
      .catch(function () {
        compareScore.textContent = "API not available";
      });
  }

  function exitCompareMode() {
    compareMode = false;
    compareBar.classList.add("hidden");
    compareContainer.classList.add("hidden");
    cyContainer.style.display = "";
    if (cyLeft) { cyLeft.destroy(); cyLeft = null; }
    if (cyRight) { cyRight.destroy(); cyRight = null; }
    compareScore.textContent = "";
  }

  function populateCompareSelects(cdgs) {
    [compareLeftSelect, compareRightSelect].forEach(function (sel) {
      sel.innerHTML = '<option value="">Select CDG...</option>';
      cdgs.forEach(function (cdg) {
        var opt = document.createElement("option");
        opt.value = cdg.repo;
        opt.textContent = cdg.repo + " (" + cdg.node_count + " nodes)";
        sel.appendChild(opt);
      });
    });
  }

  if (compareLeftSelect) {
    compareLeftSelect.addEventListener("change", function () {
      loadComparePane("left", compareLeftSelect.value);
    });
  }
  if (compareRightSelect) {
    compareRightSelect.addEventListener("change", function () {
      loadComparePane("right", compareRightSelect.value);
    });
  }

  function loadComparePane(side, repo) {
    if (!repo) return;
    var container = document.getElementById("compare-" + side);
    fetch("/api/cdg?repo=" + encodeURIComponent(repo))
      .then(function (res) { return res.json(); })
      .then(function (data) {
        var cyInstance = buildCompareGraph(container, data);
        if (side === "left") {
          if (cyLeft) cyLeft.destroy();
          cyLeft = cyInstance;
        } else {
          if (cyRight) cyRight.destroy();
          cyRight = cyInstance;
        }
        updateJaccardScore();
      });
  }

  function buildCompareGraph(container, data) {
    var elements = [];
    data.nodes.forEach(function (node) {
      var conceptType = node.concept_type || "custom";
      var colors = getNodeColors(conceptType);
      var status = node.status || "pending";
      var shape = STATUS_SHAPES[status] || "ellipse";
      elements.push({
        group: "nodes",
        data: {
          id: node.node_id,
          label: node.name,
          bgColor: colors.bg,
          borderColor: colors.border,
          textColor: colors.text,
          shape: shape,
          size: 40,
          conceptType: conceptType
        }
      });
    });
    data.edges.forEach(function (edge, i) {
      elements.push({
        group: "edges",
        data: {
          id: "df_" + i + "_" + edge.source_id + "_" + edge.target_id,
          source: edge.source_id,
          target: edge.target_id,
          edgeType: "dataflow"
        }
      });
    });

    return cytoscape({
      container: container,
      elements: elements,
      style: getCytoscapeStyle(),
      layout: { name: "dagre", rankDir: "TB", nodeSep: 30, rankSep: 50 },
      wheelSensitivity: 0.3
    });
  }

  function updateJaccardScore() {
    if (!cyLeft || !cyRight) {
      compareScore.textContent = "";
      return;
    }
    // Compute Jaccard on concept_type multisets
    var leftTypes = {};
    var rightTypes = {};
    cyLeft.nodes().forEach(function (n) {
      var ct = n.data("conceptType") || "custom";
      leftTypes[ct] = (leftTypes[ct] || 0) + 1;
    });
    cyRight.nodes().forEach(function (n) {
      var ct = n.data("conceptType") || "custom";
      rightTypes[ct] = (rightTypes[ct] || 0) + 1;
    });

    var allTypes = {};
    Object.keys(leftTypes).forEach(function (k) { allTypes[k] = true; });
    Object.keys(rightTypes).forEach(function (k) { allTypes[k] = true; });

    var intersection = 0;
    var union = 0;
    Object.keys(allTypes).forEach(function (k) {
      var l = leftTypes[k] || 0;
      var r = rightTypes[k] || 0;
      intersection += Math.min(l, r);
      union += Math.max(l, r);
    });

    var jaccard = union > 0 ? (intersection / union) : 0;
    compareScore.textContent = "Jaccard similarity: " + jaccard.toFixed(3);
  }

  // --- Isomorphism search ---

  var btnFindIso = document.getElementById("btn-find-iso");
  var isoModal = document.getElementById("iso-modal");
  var isoMinSim = document.getElementById("iso-min-sim");
  var isoSimValue = document.getElementById("iso-sim-value");
  var isoMaxResults = document.getElementById("iso-max-results");
  var isoCancel = document.getElementById("iso-cancel");
  var isoSearchBtn = document.getElementById("iso-search");
  var isoLoading = document.getElementById("iso-loading");
  var isoEmpty = document.getElementById("iso-empty");
  var isoResults = document.getElementById("iso-results");
  var isoSelectedNode = null; // node data of the selected node when iso button was clicked

  // Show/hide iso button based on whether node is decomposed or child of decomposed
  function updateIsoButtonVisibility(nodeData) {
    if (!btnFindIso || !apiAvailable) return;
    var show = false;
    if (nodeData) {
      // Show if the node is decomposed, or if it has a parent (meaning the parent is decomposed)
      var hasChildren = nodeData.children && nodeData.children.length > 0;
      var hasParent = !!nodeData.parent_id;
      show = hasChildren || hasParent;
    }
    btnFindIso.classList.toggle("visible", show);
  }

  // Slider value display
  if (isoMinSim) {
    isoMinSim.addEventListener("input", function () {
      isoSimValue.textContent = parseFloat(isoMinSim.value).toFixed(2);
    });
  }

  // Open modal
  if (btnFindIso) {
    btnFindIso.addEventListener("click", function () {
      if (!selectedNodeId || !currentData) return;
      isoSelectedNode = nodeById[selectedNodeId];
      if (!isoSelectedNode) return;

      // Configure scope radio: disable "Parent" if node is root-level decomposed
      var parentRadio = document.querySelector('input[name="iso-scope"][value="parent"]');
      if (parentRadio) {
        var hasParent = !!isoSelectedNode.parent_id;
        parentRadio.disabled = !hasParent;
        if (!hasParent) {
          document.querySelector('input[name="iso-scope"][value="this"]').checked = true;
        }
      }

      isoModal.classList.remove("hidden");
    });
  }

  // Close modal
  if (isoCancel) {
    isoCancel.addEventListener("click", function () {
      isoModal.classList.add("hidden");
    });
  }

  // Close on backdrop click
  if (isoModal) {
    var backdrop = isoModal.querySelector(".iso-modal-backdrop");
    if (backdrop) {
      backdrop.addEventListener("click", function () {
        isoModal.classList.add("hidden");
      });
    }
  }

  // Search
  if (isoSearchBtn) {
    isoSearchBtn.addEventListener("click", function () {
      if (!isoSelectedNode || !currentData) return;
      isoModal.classList.add("hidden");

      var scope = document.querySelector('input[name="iso-scope"]:checked').value;
      var layers = [];
      if (document.getElementById("iso-layer-1").checked) layers.push(1);
      if (document.getElementById("iso-layer-2").checked) layers.push(2);
      if (document.getElementById("iso-layer-3").checked) layers.push(3);

      var repo = currentData.metadata && currentData.metadata.repo ? currentData.metadata.repo : "";
      var nodeId = scope === "parent" && isoSelectedNode.parent_id
        ? isoSelectedNode.parent_id
        : isoSelectedNode.node_id;
      var radius = scope === "parent" ? 1 : 0;

      var body = {
        repo: repo,
        node_id: nodeId,
        radius: radius,
        min_jaccard: parseFloat(isoMinSim.value),
        max_results: parseInt(isoMaxResults.value, 10) || 20,
        layers: layers
      };

      // Switch to isomorphisms tab
      activateTab("isomorphisms");

      // Show loading
      isoLoading.classList.remove("hidden");
      isoEmpty.classList.add("hidden");
      isoResults.innerHTML = "";

      fetch("/api/isomorphisms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      })
        .then(function (res) {
          if (!res.ok) throw new Error("API error " + res.status);
          return res.json();
        })
        .then(function (data) {
          isoLoading.classList.add("hidden");
          renderIsoResults(data);
        })
        .catch(function (err) {
          isoLoading.classList.add("hidden");
          isoResults.innerHTML = '<div class="iso-empty"><p>Error: ' + err.message + '</p></div>';
        });
    });
  }

  function activateTab(tabName) {
    var tabs = document.querySelectorAll(".detail-tab");
    var contents = document.querySelectorAll(".tab-content");
    tabs.forEach(function (t) {
      t.classList.toggle("active", t.getAttribute("data-tab") === tabName);
    });
    contents.forEach(function (c) {
      c.classList.toggle("active", c.id === "tab-" + tabName);
    });
  }

  function renderIsoResults(data) {
    isoResults.innerHTML = "";

    if (!data.results || data.results.length === 0) {
      isoEmpty.classList.remove("hidden");
      isoEmpty.innerHTML = "<p>No similar subgraphs found</p>";
      return;
    }

    isoEmpty.classList.add("hidden");

    // Query info
    var info = document.createElement("div");
    info.className = "iso-query-info";
    info.textContent = "Query: " + data.query_node.name +
      " (" + data.query_node.concept_type + ", " + data.query_node.n_children + " children)" +
      " — " + data.results.length + " result" + (data.results.length === 1 ? "" : "s");
    isoResults.appendChild(info);

    data.results.forEach(function (result) {
      var row = document.createElement("div");
      row.className = "iso-result-row";
      row.addEventListener("click", function () {
        openIsoInCompare(result.repo);
      });

      // Top line: score bar + name + score value
      var top = document.createElement("div");
      top.className = "iso-result-top";

      var scoreBar = document.createElement("div");
      scoreBar.className = "iso-score-bar";
      var scoreFill = document.createElement("div");
      scoreFill.className = "iso-score-fill";
      scoreFill.style.width = (result.score * 100) + "%";
      scoreFill.style.background = scoreColor(result.score);
      scoreBar.appendChild(scoreFill);
      top.appendChild(scoreBar);

      var name = document.createElement("span");
      name.className = "iso-result-name";
      name.textContent = result.name;
      name.title = result.fqn;
      top.appendChild(name);

      var scoreText = document.createElement("span");
      scoreText.className = "iso-result-score";
      scoreText.textContent = result.score.toFixed(2);
      top.appendChild(scoreText);

      row.appendChild(top);

      // Meta line: layer badge, repo, concept bar
      var meta = document.createElement("div");
      meta.className = "iso-result-meta";

      var layerBadge = document.createElement("span");
      var layerNames = { 1: "topo", 2: "struct", 3: "jaccard" };
      var layerClasses = { 1: "iso-layer-topo", 2: "iso-layer-struct", 3: "iso-layer-jaccard" };
      layerBadge.className = "iso-layer-badge " + (layerClasses[result.layer] || "");
      layerBadge.textContent = layerNames[result.layer] || "?";
      meta.appendChild(layerBadge);

      var repo = document.createElement("span");
      repo.textContent = result.repo;
      meta.appendChild(repo);

      // Mini concept bar from children_summary
      if (result.children_summary && result.children_summary.length > 0) {
        var conceptBar = document.createElement("span");
        conceptBar.className = "iso-result-concepts";
        var familyCounts = {};
        result.children_summary.forEach(function (ct) {
          var fam = CONCEPT_FAMILY[ct] || "other";
          familyCounts[fam] = (familyCounts[fam] || 0) + 1;
        });
        var total = result.children_summary.length;
        Object.keys(familyCounts).forEach(function (fam) {
          var seg = document.createElement("span");
          seg.className = "iso-concept-seg";
          seg.style.width = Math.max(4, Math.round(familyCounts[fam] / total * 40)) + "px";
          seg.style.background = (FAMILY_COLORS[fam] || FAMILY_COLORS.other).border;
          seg.title = (FAMILY_LABELS[fam] || fam) + ": " + familyCounts[fam];
          conceptBar.appendChild(seg);
        });
        meta.appendChild(conceptBar);
      }

      row.appendChild(meta);
      isoResults.appendChild(row);
    });
  }

  function scoreColor(score) {
    if (score >= 0.8) return "#43a047";
    if (score >= 0.5) return "#fb8c00";
    return "#e53935";
  }

  function openIsoInCompare(matchRepo) {
    var currentRepo = currentData && currentData.metadata ? currentData.metadata.repo : "";
    if (!currentRepo || !matchRepo) return;

    enterCompareMode();

    // Wait for selects to be populated, then set values
    var waitForSelects = setInterval(function () {
      if (compareLeftSelect.options.length > 1) {
        clearInterval(waitForSelects);
        compareLeftSelect.value = currentRepo;
        compareRightSelect.value = matchRepo;
        loadComparePane("left", currentRepo);
        loadComparePane("right", matchRepo);
      }
    }, 100);

    // Safety timeout
    setTimeout(function () { clearInterval(waitForSelects); }, 5000);
  }

  // --- Animated flow direction ---

  var flowOffset = 0;
  function animateFlow() {
    flowOffset = (flowOffset + 0.3) % 18;
    if (cy) {
      cy.edges("[edgeType='dataflow']").not(".collapsed-hidden").style("line-dash-offset", -flowOffset);
    }
    requestAnimationFrame(animateFlow);
  }
  requestAnimationFrame(animateFlow);

  // --- Init ---

  fetchCDGList();
  tryLoadDefault();
})();
