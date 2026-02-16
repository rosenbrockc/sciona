(function () {
  "use strict";

  // --- Constants ---

  var STATUS_COLORS = {
    pending:    { bg: "#e0e0e0", border: "#9e9e9e", text: "#616161" },
    decomposed: { bg: "#bbdefb", border: "#42a5f5", text: "#1565c0" },
    atomic:     { bg: "#c8e6c9", border: "#66bb6a", text: "#2e7d32" },
    rejected:   { bg: "#ffcdd2", border: "#ef5350", text: "#c62828" },
    high_risk:  { bg: "#ffe0b2", border: "#ffa726", text: "#e65100" }
  };

  var CONCEPT_SHAPES = {
    divide_and_conquer: "hexagon",
    searching:          "diamond",
    sorting:            "round-rectangle",
    greedy:             "triangle",
    dynamic_programming: "vee",
    graph_traversal:    "pentagon",
    graph_optimization: "octagon",
    string_matching:    "rhomboid",
    geometry:           "star",
    arithmetic:         "round-diamond",
    number_theory:      "round-triangle",
    combinatorics:      "round-pentagon",
    algebra:            "round-hexagon",
    analysis:           "round-octagon",
    set_theory:         "concave-hexagon",
    custom:             "ellipse"
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

  var cy = null;

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
      // edges can be empty but the key must exist
      data.edges = [];
    }
    dropZone.classList.add("hidden");
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

  // --- Graph construction ---

  function buildGraph(data) {
    if (cy) {
      cy.destroy();
      cy = null;
    }

    // Update metadata bar
    var meta = data.metadata || {};
    metaGoal.textContent = "Goal: " + (meta.goal || "—");
    metaParadigm.textContent = "Paradigm: " + (meta.paradigm || "—");
    metaNodes.textContent = "Nodes: " + data.nodes.length;
    metaEdges.textContent = "Edges: " + data.edges.length;
    metaThread.textContent = "Thread: " + (meta.thread_id ? meta.thread_id.substring(0, 12) : "—");

    // Build node elements
    var elements = [];
    var dataFlowPairs = {};

    // Index data-flow edges to suppress duplicate hierarchy edges
    data.edges.forEach(function (edge) {
      var key = edge.source_id + "->" + edge.target_id;
      dataFlowPairs[key] = true;
    });

    data.nodes.forEach(function (node) {
      var status = node.status || "pending";
      var colors = STATUS_COLORS[status] || STATUS_COLORS.pending;
      var shape = CONCEPT_SHAPES[node.concept_type] || "ellipse";
      var childCount = (node.children && node.children.length) || 0;
      var size = Math.min(80, 40 + childCount * 8);

      elements.push({
        group: "nodes",
        data: {
          id: node.node_id,
          label: node.name,
          _nodeData: node,
          bgColor: colors.bg,
          borderColor: colors.border,
          shape: shape,
          size: size
        }
      });

      // Hierarchy edge from parent (dotted gray), suppressed if data-flow already covers it
      if (node.parent_id) {
        var pairKey = node.parent_id + "->" + node.node_id;
        var reversePairKey = node.node_id + "->" + node.parent_id;
        if (!dataFlowPairs[pairKey] && !dataFlowPairs[reversePairKey]) {
          elements.push({
            group: "edges",
            data: {
              id: "hier_" + node.parent_id + "_" + node.node_id,
              source: node.parent_id,
              target: node.node_id,
              edgeType: "hierarchy"
            }
          });
        }
      }
    });

    // Data-flow edges
    data.edges.forEach(function (edge, i) {
      var edgeLabel = edge.output_name + " \u2192 " + edge.input_name;
      var classes = edge.requires_glue ? "glue-edge" : "";
      elements.push({
        group: "edges",
        data: {
          id: "df_" + i + "_" + edge.source_id + "_" + edge.target_id,
          source: edge.source_id,
          target: edge.target_id,
          label: edgeLabel,
          edgeType: "dataflow",
          requiresGlue: !!edge.requires_glue
        },
        classes: classes
      });
    });

    // Initialize Cytoscape
    cy = cytoscape({
      container: cyContainer,
      elements: elements,
      style: getCytoscapeStyle(),
      layout: getLayoutConfig(layoutSelect.value),
      wheelSensitivity: 0.3
    });

    // Interactions
    cy.on("tap", "node", onNodeTap);
    cy.on("tap", function (e) {
      if (e.target === cy) {
        detailPanel.classList.remove("visible");
      }
    });

    statusText.textContent = data.nodes.length + " nodes, " + data.edges.length + " data-flow edges";
  }

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
          "background-color": "data(bgColor)",
          "border-color": "data(borderColor)",
          "border-width": 2,
          "shape": "data(shape)",
          "width": "data(size)",
          "height": "data(size)"
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
      // Data-flow edges
      {
        selector: "edge[edgeType='dataflow']",
        style: {
          "width": 2,
          "line-style": "solid",
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

  // --- Node tap / detail panel ---

  function onNodeTap(e) {
    var nodeData = e.target.data("_nodeData");
    if (!nodeData) return;
    populateDetailPanel(nodeData);
    detailPanel.classList.add("visible");
  }

  function populateDetailPanel(node) {
    document.getElementById("detail-name").textContent = node.name || "(unnamed)";

    // Status badge
    var statusEl = document.getElementById("detail-status");
    var status = node.status || "pending";
    statusEl.textContent = status;
    statusEl.className = "status-badge status-" + status;

    // Concept type
    var ct = node.concept_type || "";
    document.getElementById("detail-concept-type").textContent = ct.replace(/_/g, " ");

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

  // --- Init ---

  tryLoadDefault();
})();
