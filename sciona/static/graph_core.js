(function (global) {
  "use strict";

  global.initVisualizerGraph = function initVisualizerGraph(options) {
    var metaGoal = document.getElementById("meta-goal");
    var metaParadigm = document.getElementById("meta-paradigm");
    var metaNodes = document.getElementById("meta-nodes");
    var metaEdges = document.getElementById("meta-edges");
    var metaThread = document.getElementById("meta-thread");
    var statusText = document.getElementById("status-text");
    var btnFit = document.getElementById("btn-fit");
    var btnReset = document.getElementById("btn-reset");
    var layoutSelect = document.getElementById("layout-select");
    var cyContainer = document.getElementById("cy-container");
    var dropZone = document.getElementById("drop-zone");
    var graphSearch = document.getElementById("graph-search");
    var legendPanel = document.getElementById("legend-panel");
    var btnLegend = document.getElementById("btn-legend");
    var breadcrumbBar = document.getElementById("breadcrumb-bar");
    var breadcrumbContent = document.getElementById("breadcrumb-content");

    var cy = null;
    var currentData = null;
    var expandedNodes = {};
    var nodeById = {};
    var childrenOf = {};
    var parentOf = {};
    var breadcrumbPath = [];
    var searchMatches = [];
    var searchIndex = -1;

    function setStatus(text) {
      if (statusText) statusText.textContent = text;
    }

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
      if (!breadcrumbBar || !breadcrumbContent) return;
      if (breadcrumbPath.length === 0) {
        breadcrumbBar.classList.add("hidden");
        return;
      }
      breadcrumbBar.classList.remove("hidden");
      breadcrumbContent.innerHTML = "";

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

    function getCytoscapeStyle() {
      return [
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
        {
          selector: "node:selected",
          style: {
            "border-width": 4,
            "overlay-opacity": 0.15,
            "overlay-color": "#42a5f5"
          }
        },
        {
          selector: ".dimmed",
          style: {
            "opacity": 0.15
          }
        },
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
        {
          selector: ".hover-focus",
          style: {
            "border-width": 4,
            "opacity": 1,
            "z-index": 20
          }
        },
        {
          selector: "edge.edge-tooltip",
          style: {
            "width": 3,
            "z-index": 15
          }
        },
        {
          selector: ".search-dimmed",
          style: {
            "opacity": 0.12
          }
        },
        {
          selector: ".search-match",
          style: {
            "border-width": 4,
            "border-color": "#ff6f00",
            "opacity": 1,
            "z-index": 10
          }
        },
        {
          selector: ".collapsed-hidden",
          style: {
            "display": "none"
          }
        },
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
      return { name: "dagre", rankDir: "TB", nodeSep: 50, rankSep: 80 };
    }

    function rebuildVisibleGraph() {
      if (!cy || !currentData) return;

      var visibleNodeIds = {};
      currentData.nodes.forEach(function (node) {
        if (isNodeVisible(node.node_id)) {
          visibleNodeIds[node.node_id] = true;
        }
      });

      cy.nodes().forEach(function (n) {
        if (visibleNodeIds[n.id()]) {
          n.removeClass("collapsed-hidden");
        } else {
          n.addClass("collapsed-hidden");
        }
        var nd = n.data("_nodeData");
        if (nd && isDecomposed(n.id())) {
          if (expandedNodes[n.id()]) {
            n.data("label", nd.name + " [-]");
          } else {
            n.data("label", nd.name + " [" + childrenOf[n.id()].length + "]");
          }
        }
      });

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

      var visible = cy.elements().not(".collapsed-hidden");
      if (visible.length > 0) {
        visible.layout(getLayoutConfig(layoutSelect.value)).run();
      }

      setStatus(Object.keys(visibleNodeIds).length + " of " + currentData.nodes.length + " nodes visible");
    }

    function collapseAll() {
      expandedNodes = {};
      breadcrumbPath = [];
      rebuildVisibleGraph();
    }

    function collapseTo(nodeId, depth) {
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
        delete expandedNodes[nodeId];
        getDescendants(nodeId).forEach(function (id) { delete expandedNodes[id]; });
        var idx = breadcrumbPath.indexOf(nodeId);
        if (idx >= 0) breadcrumbPath = breadcrumbPath.slice(0, idx);
      } else {
        expandedNodes[nodeId] = true;
        breadcrumbPath = [];
        var cur = nodeId;
        while (cur) {
          if (expandedNodes[cur]) breadcrumbPath.unshift(cur);
          cur = parentOf[cur];
        }
      }
      rebuildVisibleGraph();
    }

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

    function parseSearchQuery(raw) {
      var structured = {};
      var freeText = [];
      raw.trim().split(/\s+/).forEach(function (tok) {
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
        if (depthStr.charAt(0) === ">" && nodeDepth <= parseInt(depthStr.substring(1), 10)) return false;
        if (depthStr.charAt(0) === "<" && nodeDepth >= parseInt(depthStr.substring(1), 10)) return false;
        if (depthStr.charAt(0) !== ">" && depthStr.charAt(0) !== "<" && nodeDepth !== parseInt(depthStr, 10)) return false;
      }
      if (query.freeText) {
        var haystack = ((nodeData.name || "") + " " + (nodeData.description || "")).toLowerCase();
        if (haystack.indexOf(query.freeText) === -1) return false;
      }
      return true;
    }

    function panToSearchMatch() {
      if (!cy || searchMatches.length === 0) return;
      var node = cy.getElementById(searchMatches[searchIndex]);
      if (node.length) {
        cy.animate({
          center: { eles: node },
          zoom: Math.max(cy.zoom(), 1.2)
        }, { duration: 200 });
      }
    }

    function runSearch(raw) {
      if (!cy) return;

      searchMatches = [];
      searchIndex = -1;

      if (!raw || !raw.trim()) {
        cy.elements().removeClass("search-match search-dimmed");
        setStatus(currentData ? currentData.nodes.length + " nodes, " + currentData.edges.length + " data-flow edges" : "No data loaded");
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
        setStatus("No matches");
        return;
      }

      var matchSelector = matchIds.map(function (id) { return 'node[id="' + id + '"]'; }).join(", ");
      var matchNodes = cy.nodes(matchSelector);

      cy.elements().addClass("search-dimmed");
      matchNodes.removeClass("search-dimmed").addClass("search-match");
      matchNodes.edgesWith(matchNodes).removeClass("search-dimmed");

      searchMatches = matchIds;
      searchIndex = 0;
      setStatus(matchIds.length + " match" + (matchIds.length === 1 ? "" : "es"));
      panToSearchMatch();
    }

    function cycleSearchMatch() {
      if (searchMatches.length <= 1) return;
      searchIndex = (searchIndex + 1) % searchMatches.length;
      setStatus("Match " + (searchIndex + 1) + " of " + searchMatches.length);
      panToSearchMatch();
    }

    function buildLegend() {
      var container = document.getElementById("legend-content");
      if (!container) return;
      container.innerHTML = "";

      var colorTitle = document.createElement("div");
      colorTitle.className = "legend-group-title";
      colorTitle.textContent = "Color = Concept Type Family";
      container.appendChild(colorTitle);

      Object.keys(options.familyColors).forEach(function (key) {
        var row = document.createElement("div");
        row.className = "legend-row";
        var swatch = document.createElement("span");
        swatch.className = "legend-swatch";
        swatch.style.background = options.familyColors[key].bg;
        swatch.style.borderColor = options.familyColors[key].border;
        var label = document.createElement("span");
        label.className = "legend-label";
        label.textContent = options.familyLabels[key] || key;
        row.appendChild(swatch);
        row.appendChild(label);
        container.appendChild(row);
      });

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

    function buildGraph(data) {
      if (cy) {
        cy.destroy();
        cy = null;
      }

      expandedNodes = {};
      breadcrumbPath = [];
      buildIndexes(data);
      searchMatches = [];
      searchIndex = -1;
      if (graphSearch) graphSearch.value = "";

      var meta = data.metadata || {};
      if (metaGoal) metaGoal.textContent = "Goal: " + (meta.goal || "—");
      if (metaParadigm) metaParadigm.textContent = "Paradigm: " + (meta.paradigm || "—");
      if (metaNodes) metaNodes.textContent = "Nodes: " + data.nodes.length;
      if (metaEdges) metaEdges.textContent = "Edges: " + data.edges.length;
      if (metaThread) metaThread.textContent = "Thread: " + (meta.thread_id ? meta.thread_id.substring(0, 12) : "—");

      var elements = [];
      var dataFlowPairs = {};
      data.edges.forEach(function (edge) {
        dataFlowPairs[edge.source_id + "->" + edge.target_id] = true;
      });

      data.nodes.forEach(function (node) {
        var status = node.status || "pending";
        var conceptType = node.concept_type || "custom";
        var colors = options.getNodeColors(conceptType);
        var shape = options.statusShapes[status] || "ellipse";
        var childCount = (node.children && node.children.length) || 0;
        var size = Math.min(80, 40 + childCount * 8);
        var visible = isNodeVisible(node.node_id);
        var label = isDecomposed(node.node_id)
          ? node.name + " [" + childrenOf[node.node_id].length + "]"
          : node.name;

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
              },
              classes: visible && isNodeVisible(node.parent_id) ? "" : "collapsed-hidden"
            });
          }
        }
      });

      data.edges.forEach(function (edge, i) {
        var classes = edge.requires_glue ? "glue-edge" : "";
        if (!isNodeVisible(edge.source_id) || !isNodeVisible(edge.target_id)) {
          classes = classes ? classes + " collapsed-hidden" : "collapsed-hidden";
        }
        elements.push({
          group: "edges",
          data: {
            id: "df_" + i + "_" + edge.source_id + "_" + edge.target_id,
            source: edge.source_id,
            target: edge.target_id,
            label: edge.output_name + " \u2192 " + edge.input_name,
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

      cy = cytoscape({
        container: cyContainer,
        elements: elements,
        style: getCytoscapeStyle(),
        layout: { name: "preset" },
        wheelSensitivity: 0.3,
        maxZoom: 3
      });

      var visible = cy.elements().not(".collapsed-hidden");
      if (visible.length > 0) {
        visible.layout(getLayoutConfig(layoutSelect.value)).run();
      }

      cy.on("tap", function (e) {
        if (e.target === cy) {
          if (options.onCanvasTapped) options.onCanvasTapped();
        } else if (e.target.isNode()) {
          if (options.onNodeSelected) options.onNodeSelected(e.target.data("_nodeData"));
        }
      });

      cy.on("dbltap", "node", function (e) {
        if (isDecomposed(e.target.id())) {
          toggleExpand(e.target.id());
        }
      });

      cy.on("mouseover", "node", onNodeMouseOver);
      cy.on("mouseout", "node", onNodeMouseOut);
      cy.on("mouseover", "edge", onEdgeMouseOver);
      cy.on("mouseout", "edge", onEdgeMouseOut);

      setStatus(data.nodes.filter(function (n) { return isNodeVisible(n.node_id); }).length + " of " + data.nodes.length + " nodes visible (double-click to expand)");
      updateBreadcrumb();
    }

    function validateAndLoad(data) {
      if (!data.nodes || !Array.isArray(data.nodes)) {
        setStatus("Error: JSON must contain a 'nodes' array");
        return;
      }
      if (!data.edges || !Array.isArray(data.edges)) {
        data.edges = [];
      }
      if (dropZone) dropZone.classList.add("hidden");
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
        .catch(function () {});
    }

    function focusNode(nodeId) {
      if (!cy) return;
      var cyNode = cy.getElementById(nodeId);
      if (cyNode.length) {
        cy.animate({ center: { eles: cyNode } }, { duration: 200 });
        if (options.onNodeSelected) options.onNodeSelected(cyNode.data("_nodeData"));
      }
    }

    document.body.addEventListener("dragover", function (e) {
      e.preventDefault();
      e.stopPropagation();
      if (dropZone) dropZone.classList.add("drag-active");
    });

    document.body.addEventListener("dragleave", function (e) {
      e.preventDefault();
      e.stopPropagation();
      if (dropZone) dropZone.classList.remove("drag-active");
    });

    buildLegend();

    if (btnLegend && legendPanel) {
      btnLegend.addEventListener("click", function () {
        legendPanel.classList.toggle("visible");
      });
    }

    if (layoutSelect) {
      layoutSelect.addEventListener("change", function () {
        if (cy) cy.layout(getLayoutConfig(layoutSelect.value)).run();
      });
    }

    if (btnFit) {
      btnFit.addEventListener("click", function () {
        if (cy) cy.fit(undefined, 30);
      });
    }

    if (btnReset) {
      btnReset.addEventListener("click", function () {
        if (cy) {
          cy.layout(getLayoutConfig(layoutSelect.value)).run();
          cy.fit(undefined, 30);
        }
      });
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

    return {
      validateAndLoad: validateAndLoad,
      tryLoadDefault: tryLoadDefault,
      getCurrentData: function () { return currentData; },
      getNodeById: function (nodeId) { return nodeById[nodeId]; },
      getCy: function () { return cy; },
      getCytoscapeStyle: getCytoscapeStyle,
      focusNode: focusNode
    };
  };
})(window);
