(function () {
  "use strict";

  var CONCEPT_FAMILY = {
    sorting: "math", searching: "math", divide_and_conquer: "math",
    greedy: "math", dynamic_programming: "math", combinatorics: "math",
    algebra: "math", analysis: "math", arithmetic: "math",
    number_theory: "math", geometry: "math", set_theory: "math",
    sampler: "prob", log_prob: "prob", posterior_update: "prob",
    variational_inference: "prob", prior_init: "prob",
    prior_distribution: "prob", likelihood_evaluation: "prob",
    probabilistic_oracle: "prob", oracle_gradient: "prob",
    mcmc_kernel: "prob", mcmc_proposal: "prob", vi_elbo: "prob",
    conjugate_update: "prob",
    signal_filter: "signal", signal_transform: "signal",
    graph_signal_processing: "signal", sequential_filter: "signal",
    smc_reweight: "signal",
    state_init: "orch", data_assembly: "orch",
    conditional_routing: "orch", data_extraction: "orch",
    visualization: "pres", observability: "pres",
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

  var STATUS_SHAPES = {
    atomic: "ellipse",
    decomposed: "round-rectangle",
    external: "diamond",
    pending: "ellipse",
    rejected: "cut-rectangle",
    high_risk: "triangle"
  };

  var btnOpen = document.getElementById("btn-open");
  var btnDashboard = document.getElementById("btn-dashboard");
  var fileInput = document.getElementById("file-input");

  function getNodeColors(conceptType) {
    var family = CONCEPT_FAMILY[conceptType] || "other";
    return FAMILY_COLORS[family] || FAMILY_COLORS.other;
  }

  function handleFile(file, graphControls) {
    var reader = new FileReader();
    reader.onload = function (e) {
      try {
        graphControls.validateAndLoad(JSON.parse(e.target.result));
      } catch (err) {
        var statusText = document.getElementById("status-text");
        if (statusText) statusText.textContent = "Error: invalid JSON — " + err.message;
      }
    };
    reader.readAsText(file);
  }

  var detailControls = null;
  var isoControls = null;
  detailControls = window.initVisualizerDetailPanel({
    conceptFamily: CONCEPT_FAMILY,
    familyColors: FAMILY_COLORS,
    getCy: function () { return graphControls.getCy(); },
    getNodeById: function (nodeId) { return graphControls.getNodeById(nodeId); },
    getNodeColors: getNodeColors,
    focusNode: function (nodeId) { graphControls.focusNode(nodeId); }
  });

  var graphControls = window.initVisualizerGraph({
    familyColors: FAMILY_COLORS,
    familyLabels: FAMILY_LABELS,
    getNodeColors: getNodeColors,
    statusShapes: STATUS_SHAPES,
    onNodeSelected: function (nodeData) {
      detailControls.handleNodeSelected(nodeData);
      if (isoControls) isoControls.updateButtonVisibility(nodeData);
    },
    onCanvasTapped: function () {
      detailControls.hide();
    }
  });

  var browserControls = window.initVisualizerBrowser({
    conceptFamily: CONCEPT_FAMILY,
    familyColors: FAMILY_COLORS,
    familyLabels: FAMILY_LABELS,
    setStatus: function (text) {
      var statusText = document.getElementById("status-text");
      if (statusText) statusText.textContent = text;
    },
    validateAndLoad: graphControls.validateAndLoad
  });

  var compareControls = window.initVisualizerCompare({
    cyContainer: document.getElementById("cy-container"),
    detailPanel: detailControls.getPanel(),
    getNodeColors: getNodeColors,
    getCytoscapeStyle: graphControls.getCytoscapeStyle,
    statusShapes: STATUS_SHAPES
  });

  isoControls = window.initVisualizerIsomorphism({
    getSelectedNodeId: function () {
      return detailControls.getSelectedNodeId();
    },
    getCurrentData: function () {
      return graphControls.getCurrentData();
    },
    getNodeById: function (nodeId) {
      return graphControls.getNodeById(nodeId);
    },
    isApiAvailable: function () {
      return browserControls && browserControls.isApiAvailable();
    },
    activateTab: detailControls.activateTab,
    openInCompare: function (currentRepo, matchRepo) {
      compareControls.openInCompare(currentRepo, matchRepo);
    },
    conceptFamily: CONCEPT_FAMILY,
    familyColors: FAMILY_COLORS,
    familyLabels: FAMILY_LABELS
  });

  document.body.addEventListener("drop", function (e) {
    e.preventDefault();
    e.stopPropagation();
    var dropZone = document.getElementById("drop-zone");
    if (dropZone) dropZone.classList.remove("drag-active");
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0], graphControls);
    }
  });

  if (btnOpen) {
    btnOpen.addEventListener("click", function () {
      fileInput.click();
    });
  }

  if (btnDashboard) {
    btnDashboard.addEventListener("click", function () {
      window.open("/dashboard.html", "_blank");
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0], graphControls);
        fileInput.value = "";
      }
    });
  }

  var flowOffset = 0;
  function animateFlow() {
    flowOffset = (flowOffset + 0.3) % 18;
    var cy = graphControls.getCy();
    if (cy) {
      cy.edges("[edgeType='dataflow']").not(".collapsed-hidden").style("line-dash-offset", -flowOffset);
    }
    requestAnimationFrame(animateFlow);
  }

  requestAnimationFrame(animateFlow);
  browserControls.fetchCDGList();
  graphControls.tryLoadDefault();
})();
