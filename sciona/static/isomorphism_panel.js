(function (global) {
  "use strict";

  global.initVisualizerIsomorphism = function initVisualizerIsomorphism(options) {
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
    var isoSelectedNode = null;

    function updateButtonVisibility(nodeData) {
      if (!btnFindIso) return;
      if (!options.isApiAvailable()) {
        btnFindIso.classList.remove("visible");
        return;
      }
      var show = false;
      if (nodeData) {
        var hasChildren = nodeData.children && nodeData.children.length > 0;
        var hasParent = !!nodeData.parent_id;
        show = hasChildren || hasParent;
      }
      btnFindIso.classList.toggle("visible", show);
    }

    function activateTab(tabName) {
      options.activateTab(tabName);
    }

    function renderIsoResults(data) {
      if (!isoResults || !isoEmpty) return;
      isoResults.innerHTML = "";

      if (!data.results || data.results.length === 0) {
        isoEmpty.classList.remove("hidden");
        isoEmpty.innerHTML = "<p>No similar subgraphs found</p>";
        return;
      }

      isoEmpty.classList.add("hidden");

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
          var currentData = options.getCurrentData();
          var currentRepo = currentData && currentData.metadata ? currentData.metadata.repo : "";
          options.openInCompare(currentRepo, result.repo);
        });

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

        if (result.children_summary && result.children_summary.length > 0) {
          var conceptBar = document.createElement("span");
          conceptBar.className = "iso-result-concepts";
          var familyCounts = {};
          result.children_summary.forEach(function (ct) {
            var fam = options.conceptFamily[ct] || "other";
            familyCounts[fam] = (familyCounts[fam] || 0) + 1;
          });
          var total = result.children_summary.length;
          Object.keys(familyCounts).forEach(function (fam) {
            var seg = document.createElement("span");
            seg.className = "iso-concept-seg";
            seg.style.width = Math.max(4, Math.round(familyCounts[fam] / total * 40)) + "px";
            seg.style.background = (options.familyColors[fam] || options.familyColors.other).border;
            seg.title = (options.familyLabels[fam] || fam) + ": " + familyCounts[fam];
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

    if (isoMinSim && isoSimValue) {
      isoMinSim.addEventListener("input", function () {
        isoSimValue.textContent = parseFloat(isoMinSim.value).toFixed(2);
      });
    }

    if (btnFindIso) {
      btnFindIso.addEventListener("click", function () {
        var selectedNodeId = options.getSelectedNodeId();
        if (!selectedNodeId || !options.getCurrentData()) return;
        isoSelectedNode = options.getNodeById(selectedNodeId);
        if (!isoSelectedNode) return;

        var parentRadio = document.querySelector('input[name="iso-scope"][value="parent"]');
        if (parentRadio) {
          var hasParent = !!isoSelectedNode.parent_id;
          parentRadio.disabled = !hasParent;
          if (!hasParent) {
            document.querySelector('input[name="iso-scope"][value="this"]').checked = true;
          }
        }

        if (isoModal) isoModal.classList.remove("hidden");
      });
    }

    if (isoCancel && isoModal) {
      isoCancel.addEventListener("click", function () {
        isoModal.classList.add("hidden");
      });
    }

    if (isoModal) {
      var backdrop = isoModal.querySelector(".iso-modal-backdrop");
      if (backdrop) {
        backdrop.addEventListener("click", function () {
          isoModal.classList.add("hidden");
        });
      }
    }

    if (isoSearchBtn) {
      isoSearchBtn.addEventListener("click", function () {
        var currentData = options.getCurrentData();
        if (!isoSelectedNode || !currentData) return;
        if (isoModal) isoModal.classList.add("hidden");

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

        activateTab("isomorphisms");
        if (isoLoading) isoLoading.classList.remove("hidden");
        if (isoEmpty) isoEmpty.classList.add("hidden");
        if (isoResults) isoResults.innerHTML = "";

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
            if (isoLoading) isoLoading.classList.add("hidden");
            renderIsoResults(data);
          })
          .catch(function (err) {
            if (isoLoading) isoLoading.classList.add("hidden");
            if (isoResults) {
              isoResults.innerHTML = '<div class="iso-empty"><p>Error: ' + err.message + '</p></div>';
            }
          });
      });
    }

    return {
      updateButtonVisibility: updateButtonVisibility
    };
  };
})(window);
