(function (global) {
  "use strict";

  global.initVisualizerDetailPanel = function initVisualizerDetailPanel(options) {
    var detailPanel = document.getElementById("detail-panel");
    var detailTabs = document.querySelectorAll(".detail-tab");
    var tabContents = document.querySelectorAll(".tab-content");
    var selectedNodeId = null;

    function activateTab(target) {
      detailTabs.forEach(function (t) {
        t.classList.toggle("active", t.getAttribute("data-tab") === target);
      });
      tabContents.forEach(function (c) {
        c.classList.toggle("active", c.id === "tab-" + target);
      });
    }

    function populateIOTable(tableId, specs) {
      var table = document.getElementById(tableId);
      if (!table) return;
      var tbody = table.querySelector("tbody");
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
        var tdType = document.createElement("td");
        var tdConstraints = document.createElement("td");
        tdName.textContent = spec.name || "";
        tdType.textContent = spec.type_desc || "";
        tdConstraints.textContent = spec.constraints || "";
        row.appendChild(tdName);
        row.appendChild(tdType);
        row.appendChild(tdConstraints);
        tbody.appendChild(row);
      });
    }

    function populateDetailPanel(node) {
      document.getElementById("detail-name").textContent = node.name || "(unnamed)";

      var statusEl = document.getElementById("detail-status");
      var status = node.status || "pending";
      statusEl.textContent = status;
      statusEl.className = "status-badge status-" + status;

      var ct = node.concept_type || "custom";
      var ctEl = document.getElementById("detail-concept-type");
      var family = options.conceptFamily[ct] || "other";
      var colors = options.familyColors[family];
      ctEl.textContent = ct.replace(/_/g, " ");
      ctEl.style.background = colors.bg;
      ctEl.style.color = colors.text;
      ctEl.style.borderColor = colors.border;

      document.getElementById("detail-description").textContent = node.description || "(none)";
      document.getElementById("detail-type-sig").textContent = node.type_signature || "(none)";
      document.getElementById("detail-primitive").textContent = node.matched_primitive || "(none)";

      populateIOTable("detail-inputs", node.inputs || []);
      populateIOTable("detail-outputs", node.outputs || []);

      document.getElementById("detail-depth").textContent = node.depth != null ? String(node.depth) : "—";
      document.getElementById("detail-children").textContent = node.children && node.children.length > 0 ? node.children.join(", ") : "(leaf)";
      document.getElementById("detail-parent").textContent = node.parent_id || "(root)";
      document.getElementById("detail-rationale").textContent = node.decomposition_rationale || "(none)";
      document.getElementById("detail-critic").value = node.critic_notes || "(none)";
    }

    function buildLineageTree(startId, direction, container, maxDepth) {
      var cy = options.getCy();
      var visited = {};
      visited[startId] = true;

      function getNeighbors(nid) {
        var cyNode = cy.getElementById(nid);
        if (!cyNode || cyNode.length === 0) return [];
        var edges = direction === "upstream"
          ? cyNode.incomers("edge[edgeType='dataflow']")
          : cyNode.outgoers("edge[edgeType='dataflow']");
        var results = [];
        edges.forEach(function (e) {
          results.push({
            id: direction === "upstream" ? e.data("source") : e.data("target"),
            edgeLabel: e.data("outputName") + " \u2192 " + e.data("inputName")
          });
        });
        return results;
      }

      function renderLevel(nodeIds, depth) {
        if (depth > maxDepth) return;
        var nextLevel = [];
        nodeIds.forEach(function (entry) {
          if (visited[entry.id]) return;
          visited[entry.id] = true;

          var nd = options.getNodeById(entry.id);
          var colors = options.getNodeColors(nd ? (nd.concept_type || "custom") : "custom");
          var item = document.createElement("div");
          item.className = "lineage-item" + (depth > 1 ? " lineage-depth-" + depth : "");
          item.addEventListener("click", function () {
            options.focusNode(entry.id);
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

          getNeighbors(entry.id).forEach(function (n) {
            if (!visited[n.id]) nextLevel.push(n);
          });
        });

        if (nextLevel.length > 0) renderLevel(nextLevel, depth + 1);
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

    function populateLineage(nodeId) {
      var upList = document.getElementById("lineage-upstream-list");
      var downList = document.getElementById("lineage-downstream-list");
      var hint = document.querySelector(".lineage-hint");
      if (!upList || !downList) return;
      upList.innerHTML = "";
      downList.innerHTML = "";
      if (hint) hint.style.display = "none";
      if (!options.getCy()) return;
      buildLineageTree(nodeId, "upstream", upList, 3);
      buildLineageTree(nodeId, "downstream", downList, 3);
    }

    function handleNodeSelected(node) {
      if (!node) return;
      selectedNodeId = node.node_id;
      populateDetailPanel(node);
      populateLineage(node.node_id);
      detailPanel.classList.add("visible");
    }

    function hide() {
      if (detailPanel) detailPanel.classList.remove("visible");
    }

    detailTabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        activateTab(tab.getAttribute("data-tab"));
      });
    });

    return {
      activateTab: activateTab,
      handleNodeSelected: handleNodeSelected,
      getSelectedNodeId: function () { return selectedNodeId; },
      hide: hide,
      getPanel: function () { return detailPanel; }
    };
  };
})(window);
