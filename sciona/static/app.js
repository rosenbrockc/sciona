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

  var TUTORIAL_A_CDG = {
    nodes: [
      {
        node_id: "ingest",
        name: "ECG Ingestion",
        description: "Loads the raw ECG voltage signals.",
        concept_type: "state_init",
        status: "atomic",
        matched_primitive: "biosppy.signals.ecg.load",
        inputs: [],
        outputs: [{ name: "sig", type_desc: "np.ndarray", constraints: "time domain" }],
        depth: 1
      },
      {
        node_id: "fft1",
        name: "Forward FFT",
        description: "Transforms signal into frequency domain for filtering.",
        concept_type: "signal_transform",
        status: "atomic",
        matched_primitive: "fft",
        inputs: [{ name: "sig", type_desc: "np.ndarray", constraints: "time domain" }],
        outputs: [{ name: "spectrum", type_desc: "np.ndarray", constraints: "freq domain" }],
        depth: 1
      },
      {
        node_id: "fft2",
        name: "Second Forward FFT (Invalid)",
        description: "Applies a second forward FFT, causing domain mismatch (expects time domain, gets freq domain).",
        concept_type: "signal_transform",
        status: "atomic",
        matched_primitive: "fft",
        inputs: [{ name: "sig", type_desc: "np.ndarray", constraints: "freq domain" }],
        outputs: [{ name: "spectrum2", type_desc: "np.ndarray", constraints: "freq domain" }],
        depth: 1
      }
    ],
    edges: [
      {
        source_id: "ingest",
        target_id: "fft1",
        output_name: "sig",
        input_name: "sig",
        source_type: "np.ndarray",
        target_type: "np.ndarray"
      },
      {
        source_id: "fft1",
        target_id: "fft2",
        output_name: "spectrum",
        input_name: "sig",
        source_type: "np.ndarray",
        target_type: "np.ndarray"
      }
    ],
    metadata: {
      goal: "Demonstrate GhostSim mismatch detection on ECG pipelines",
      paradigm: "filtering",
      repo: "biosppy/ecg_mismatch"
    }
  };

  var TUTORIAL_B_CDG = {
    nodes: [
      {
        node_id: "data_prep",
        name: "data_assembly",
        description: "Prepares features and labels.",
        concept_type: "data_assembly",
        status: "atomic",
        matched_primitive: "pandas.read_csv",
        inputs: [],
        outputs: [{ name: "X", type_desc: "ndarray" }],
        depth: 1
      },
      {
        node_id: "fit_est",
        name: "fit estimator",
        description: "model_training",
        concept_type: "ml_model_selection",
        status: "atomic",
        matched_primitive: "sklearn.linear_model.LogisticRegression.fit",
        inputs: [{ name: "X", type_desc: "ndarray" }],
        outputs: [{ name: "model", type_desc: "estimator" }],
        depth: 1
      },
      {
        node_id: "score_val",
        name: "score validation split",
        description: "prediction_ensemble",
        concept_type: "ml_model_selection",
        status: "atomic",
        matched_primitive: "sklearn.metrics.accuracy_score",
        inputs: [{ name: "model", type_desc: "estimator" }],
        outputs: [{ name: "score", type_desc: "float" }],
        depth: 1
      },
      {
        node_id: "kfold_ensemble",
        name: "k-fold cross validated ensemble",
        description: "Perform ensembling using K-fold CV.",
        concept_type: "ml_model_selection",
        status: "atomic",
        matched_primitive: null,
        inputs: [],
        outputs: [],
        depth: 1
      },
      {
        node_id: "stacking_meta",
        name: "stacking meta learner",
        description: "Use stacking ensemble classifier.",
        concept_type: "ml_model_selection",
        status: "atomic",
        matched_primitive: null,
        inputs: [],
        outputs: [],
        depth: 1
      }
    ],
    edges: [
      {
        source_id: "data_prep",
        target_id: "fit_est",
        output_name: "X",
        input_name: "X",
        source_type: "ndarray",
        target_type: "ndarray"
      },
      {
        source_id: "fit_est",
        target_id: "score_val",
        output_name: "model",
        input_name: "model",
        source_type: "estimator",
        target_type: "estimator"
      }
    ],
    metadata: {
      goal: "Demonstrate Delta Planner ensembling quick-fixes on Tabular ML CDGs",
      paradigm: "ml_model_selection",
      repo: "sklearn/tabular_ml"
    }
  };

  var TUTORIAL_C_CDG = {
    nodes: [
      {
        node_id: "data_load",
        name: "High-Dimensional Input",
        description: "Loads the raw high-dimensional dataset.",
        concept_type: "data_assembly",
        status: "atomic",
        matched_primitive: "pandas.read_csv",
        inputs: [],
        outputs: [{ name: "data", type_desc: "np.ndarray" }],
        depth: 1
      },
      {
        node_id: "pca_projection",
        name: "PCA Pre-reduction",
        description: "Reduces dimension to 50 using PCA to speed up downstream projection.",
        concept_type: "signal_transform",
        status: "atomic",
        matched_primitive: "sklearn.decomposition.PCA",
        inputs: [{ name: "data", type_desc: "np.ndarray" }],
        outputs: [{ name: "reduced", type_desc: "np.ndarray" }],
        depth: 1
      },
      {
        node_id: "umap_layout",
        name: "UMAP Projection",
        description: "Computes UMAP 2D coordinates.",
        concept_type: "signal_transform",
        status: "atomic",
        matched_primitive: "umap.UMAP",
        inputs: [{ name: "reduced", type_desc: "np.ndarray" }],
        outputs: [{ name: "projection", type_desc: "np.ndarray" }],
        depth: 1
      }
    ],
    edges: [
      {
        source_id: "data_load",
        target_id: "pca_projection",
        output_name: "data",
        input_name: "data",
        source_type: "np.ndarray",
        target_type: "np.ndarray"
      },
      {
        source_id: "pca_projection",
        target_id: "umap_layout",
        output_name: "reduced",
        input_name: "reduced",
        source_type: "np.ndarray",
        target_type: "np.ndarray"
      }
    ],
    metadata: {
      goal: "Structured composition and layout of UMAP scientific computing pipeline",
      paradigm: "dimensionality_reduction",
      repo: "umap/scientific_computing"
    }
  };

  var detailControls = null;
  var isoControls = null;
  var runnerControls = null;

  detailControls = window.initVisualizerDetailPanel({
    conceptFamily: CONCEPT_FAMILY,
    familyColors: FAMILY_COLORS,
    getCy: function () { return graphControls.getCy(); },
    getNodeById: function (nodeId) { return graphControls.getNodeById(nodeId); },
    getNodeColors: getNodeColors,
    focusNode: function (nodeId) { graphControls.focusNode(nodeId); },
    getRunId: function () { return runnerControls ? runnerControls.getActiveRunId() : null; },
    isApiAvailable: function () { return browserControls && browserControls.isApiAvailable(); },
    getCurrentData: function () { return graphControls ? graphControls.getCurrentData() : null; },
    validateAndLoad: function (data) { if (graphControls) graphControls.validateAndLoad(data); }
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
    },
    onCDGLoaded: function () {
      if (detailControls && detailControls.fetchQuickFixes) {
        detailControls.fetchQuickFixes(null);
      }
    },
    isApiAvailable: function () { return browserControls && browserControls.isApiAvailable(); }
  });

  runnerControls = window.initVisualizerRunner({
    getCy: function () { return graphControls.getCy(); },
    getCurrentData: function () { return graphControls.getCurrentData(); },
    isApiAvailable: function () { return browserControls && browserControls.isApiAvailable(); },
    detailControls: detailControls
  });

  // Intercept validateAndLoad to trigger runner panel repo session sync
  var originalValidateAndLoad = graphControls.validateAndLoad;
  graphControls.validateAndLoad = function (data) {
    originalValidateAndLoad(data);
    if (data && data.metadata && data.metadata.repo && runnerControls) {
      runnerControls.setRepo(data.metadata.repo);
    }
  };

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

  var btnTutorials = document.getElementById("btn-tutorials");
  var btnTutorialsClose = document.getElementById("btn-tutorials-close");
  var tutorialsModal = document.getElementById("tutorials-modal");

  if (btnTutorials && tutorialsModal) {
    btnTutorials.addEventListener("click", function () {
      tutorialsModal.classList.remove("hidden");
    });
  }

  if (btnTutorialsClose && tutorialsModal) {
    btnTutorialsClose.addEventListener("click", function () {
      tutorialsModal.classList.add("hidden");
    });
  }

  // Tutorial tab switching
  var tutorialTabButtons = document.querySelectorAll("#tutorials-tabs button");
  tutorialTabButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      tutorialTabButtons.forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      
      var selectedTut = btn.getAttribute("data-tutorial");
      var panes = document.querySelectorAll(".tutorial-pane");
      panes.forEach(function (pane) {
        if (pane.id === "tut-" + selectedTut) {
          pane.classList.remove("hidden");
        } else {
          pane.classList.add("hidden");
        }
      });
    });
  });

  // Load tutorial buttons
  var btnLoadTutA = document.getElementById("btn-load-tutorial-a");
  var btnLoadTutB = document.getElementById("btn-load-tutorial-b");
  var btnLoadTutC = document.getElementById("btn-load-tutorial-c");

  if (btnLoadTutA) {
    btnLoadTutA.addEventListener("click", function () {
      graphControls.validateAndLoad(TUTORIAL_A_CDG);
      if (tutorialsModal) tutorialsModal.classList.add("hidden");
    });
  }
  if (btnLoadTutB) {
    btnLoadTutB.addEventListener("click", function () {
      graphControls.validateAndLoad(TUTORIAL_B_CDG);
      if (tutorialsModal) tutorialsModal.classList.add("hidden");
    });
  }
  if (btnLoadTutC) {
    btnLoadTutC.addEventListener("click", function () {
      graphControls.validateAndLoad(TUTORIAL_C_CDG);
      if (tutorialsModal) tutorialsModal.classList.add("hidden");
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

  // Startup URL-based loader
  var queryRepo = null;
  var search = window.location ? window.location.search : "";
  var match = RegExp("[?&]repo=([^&]*)").exec(search);
  if (match) {
    queryRepo = decodeURIComponent(match[1].replace(/\+/g, " "));
  }

  if (queryRepo) {
    var statusText = document.getElementById("status-text");
    if (statusText) statusText.textContent = "Loading " + queryRepo + "...";
    fetch("/api/cdg?repo=" + encodeURIComponent(queryRepo))
      .then(function (res) {
        if (!res.ok) throw new Error("CDG not found");
        return res.json();
      })
      .then(function (data) {
        graphControls.validateAndLoad(data);
      })
      .catch(function (err) {
        var statusText = document.getElementById("status-text");
        if (statusText) statusText.textContent = "Error: " + err.message;
        graphControls.tryLoadDefault();
      });
  } else {
    graphControls.tryLoadDefault();
  }
})();
