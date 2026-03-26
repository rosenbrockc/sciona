(function (global) {
  "use strict";

  global.initVisualizerBrowser = function initVisualizerBrowser(options) {
    var btnBrowse = document.getElementById("btn-browse");
    var cdgBrowser = document.getElementById("cdg-browser");
    var btnBrowserClose = document.getElementById("btn-browser-close");
    var browserSearch = document.getElementById("browser-search");
    var browserList = document.getElementById("browser-list");
    var apiAvailable = false;
    var searchTimeout = null;

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
          if (btnBrowse) btnBrowse.style.display = "";
          renderCDGList(cdgs);
          return cdgs;
        })
        .catch(function () {
          apiAvailable = false;
          if (btnBrowse) btnBrowse.style.display = "none";
          if (cdgBrowser) cdgBrowser.classList.remove("visible");
        });
    }

    function renderCDGList(cdgs) {
      if (!browserList) return;
      browserList.innerHTML = "";
      if (cdgs.length === 0) {
        browserList.innerHTML = '<div class="browser-empty">No CDGs found</div>';
        return;
      }

      var groups = {};
      cdgs.forEach(function (cdg) {
        var parts = cdg.repo.split("/");
        var key = parts.length >= 2 ? parts[1] : parts[0];
        if (!groups[key]) groups[key] = [];
        groups[key].push(cdg);
      });

      Object.keys(groups).sort().forEach(function (ns) {
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
          title.textContent = cdg.repo.split("/").pop();

          var meta = document.createElement("div");
          meta.className = "browser-item-meta";

          var nodeCountSpan = document.createElement("span");
          nodeCountSpan.className = "node-count";
          nodeCountSpan.textContent = cdg.node_count + " nodes";
          meta.appendChild(nodeCountSpan);

          if (cdg.concept_types && cdg.concept_types.length > 0) {
            var barContainer = document.createElement("span");
            barContainer.className = "concept-bar";
            var familyCounts = {};
            cdg.concept_types.forEach(function (ct) {
              var fam = options.conceptFamily[ct] || "other";
              familyCounts[fam] = (familyCounts[fam] || 0) + 1;
            });
            var total = cdg.concept_types.length;
            Object.keys(familyCounts).forEach(function (fam) {
              var seg = document.createElement("span");
              seg.className = "concept-bar-seg";
              seg.style.width = Math.max(4, Math.round(familyCounts[fam] / total * 60)) + "px";
              seg.style.background = options.familyColors[fam].border;
              seg.title = options.familyLabels[fam] + ": " + familyCounts[fam];
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
      options.setStatus("Loading " + repo + "...");
      fetch("/api/cdg?repo=" + encodeURIComponent(repo))
        .then(function (res) {
          if (!res.ok) throw new Error("CDG not found");
          return res.json();
        })
        .then(function (data) {
          options.validateAndLoad(data);
          if (cdgBrowser) cdgBrowser.classList.remove("visible");
        })
        .catch(function (err) {
          options.setStatus("Error: " + err.message);
        });
    }

    if (btnBrowse && cdgBrowser) {
      btnBrowse.addEventListener("click", function () {
        cdgBrowser.classList.toggle("visible");
        if (cdgBrowser.classList.contains("visible")) {
          fetchCDGList({ q: browserSearch && browserSearch.value ? browserSearch.value : undefined });
          if (browserSearch) browserSearch.focus();
        }
      });
    }

    if (btnBrowserClose && cdgBrowser) {
      btnBrowserClose.addEventListener("click", function () {
        cdgBrowser.classList.remove("visible");
      });
    }

    if (browserSearch) {
      browserSearch.addEventListener("input", function () {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(function () {
          fetchCDGList({ q: browserSearch.value || undefined });
        }, 300);
      });
    }

    return {
      fetchCDGList: fetchCDGList,
      isApiAvailable: function () {
        return apiAvailable;
      }
    };
  };
})(window);
