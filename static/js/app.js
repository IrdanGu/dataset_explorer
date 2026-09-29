(function () {
    "use strict";

    const $ = (s, p) => (p || document).querySelector(s);
    const $$ = (s, p) => [...(p || document).querySelectorAll(s)];

    let summary = null;
    let chartInstances = {};
    let transformHistory = [];
    let previewOffset = 0;
    const PAGE_SIZE = 50;

    // ── Upload ──────────────────────────────────────────────────────────

    const dropZone = $("#drop-zone");
    const fileInput = $("#file-input");

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) uploadFile(fileInput.files[0]);
    });

    async function uploadFile(file) {
        if (!file.name.toLowerCase().endsWith(".csv")) {
            toast("Only CSV files are supported", "error");
            return;
        }
        const fd = new FormData();
        fd.append("file", file);
        try {
            const res = await api("/api/upload", { method: "POST", body: fd });
            summary = res;
            transformHistory = [];
            renderDashboard();
            toast("Dataset loaded: " + res.file_name, "success");
        } catch (e) {
            toast(e.message, "error");
        }
    }

    // ── API helper ──────────────────────────────────────────────────────

    async function api(url, opts) {
        const res = await fetch(url, opts);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");
        return data;
    }

    // ── Toast ───────────────────────────────────────────────────────────

    function toast(msg, type) {
        const el = $("#toast");
        el.textContent = msg;
        el.className = "toast " + type;
        el.classList.remove("hidden");
        clearTimeout(el._t);
        el._t = setTimeout(() => el.classList.add("hidden"), 3500);
    }

    // ── Tabs ────────────────────────────────────────────────────────────

    $$(".tab").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".tab").forEach((b) => b.classList.remove("active"));
            $$(".tab-content").forEach((c) => c.classList.remove("active"));
            btn.classList.add("active");
            const target = $("#tab-" + btn.dataset.tab);
            if (target) target.classList.add("active");
            onTabSwitch(btn.dataset.tab);
        });
    });

    function onTabSwitch(tab) {
        if (tab === "statistics") loadStatistics();
        if (tab === "distributions") loadDistribution();
        if (tab === "correlations") loadCorrelation();
        if (tab === "data") loadPreview();
    }

    // ── Dashboard render ────────────────────────────────────────────────

    function renderDashboard() {
        $("#upload-section").classList.add("hidden");
        $("#dashboard").classList.remove("hidden");
        renderSummary();
        renderOverview();
        populateSelects();
    }

    function renderSummary() {
        const s = summary;
        $("#summary-bar").innerHTML = [
            mkSummaryItem(s.file_name, "File"),
            mkSummaryItem(s.row_count.toLocaleString(), "Rows"),
            mkSummaryItem(s.col_count, "Columns"),
            mkSummaryItem(s.numeric_columns.length, "Numeric"),
            mkSummaryItem(s.categorical_columns.length, "Categorical"),
            mkSummaryItem(s.total_missing, "Missing Values"),
        ].join("");
    }

    function mkSummaryItem(val, label) {
        return `<div class="summary-item"><div class="value">${val}</div><div class="label">${label}</div></div>`;
    }

    // ── Overview tab ────────────────────────────────────────────────────

    function renderOverview() {
        renderTypeChart();
        renderMissingChart();
        renderColSummaryTable();
    }

    function renderTypeChart() {
        destroyChart("chart-types");
        const counts = {};
        summary.columns.forEach((c) => {
            counts[c.dtype] = (counts[c.dtype] || 0) + 1;
        });
        chartInstances["chart-types"] = new Chart($("#chart-types"), {
            type: "doughnut",
            data: {
                labels: Object.keys(counts),
                datasets: [{
                    data: Object.values(counts),
                    backgroundColor: ["#6366f1", "#f59e0b", "#22c55e", "#ef4444"],
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { position: "bottom", labels: { color: "#8b90a0" } } },
            },
        });
    }

    function renderMissingChart() {
        destroyChart("chart-missing");
        const cols = summary.columns.filter((c) => c.missing > 0);
        if (!cols.length) {
            $("#chart-missing").parentElement.querySelector("h3").textContent = "Missing Values — None!";
            return;
        }
        chartInstances["chart-missing"] = new Chart($("#chart-missing"), {
            type: "bar",
            data: {
                labels: cols.map((c) => c.name),
                datasets: [{
                    label: "Missing count",
                    data: cols.map((c) => c.missing),
                    backgroundColor: "#ef4444aa",
                    borderColor: "#ef4444",
                    borderWidth: 1,
                }],
            },
            options: barOpts("Missing count"),
        });
    }

    function renderColSummaryTable() {
        const rows = summary.columns.map((c) =>
            `<tr><td>${c.name}</td><td><span class="badge badge-${c.dtype}">${c.dtype}</span></td><td>${c.missing}</td><td>${c.missing_pct}%</td></tr>`
        ).join("");
        $("#col-summary-table").innerHTML =
            `<table><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Missing %</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    // ── Statistics tab ──────────────────────────────────────────────────

    function populateSelects() {
        const numOpts = summary.numeric_columns.map((n) => `<option value="${n}">${n}</option>`).join("");
        const allOpts = summary.columns.map((c) => `<option value="${c.name}">${c.name}</option>`).join("");
        const catOpts = summary.categorical_columns.map((n) => `<option value="${n}">${n}</option>`).join("");

        $("#stats-col-select").innerHTML = allOpts;
        $("#dist-col-select").innerHTML = allOpts;
        $("#tf-col-select").innerHTML = allOpts;

        $("#stats-col-select").addEventListener("change", loadStatistics);
        $("#dist-col-select").addEventListener("change", loadDistribution);
        $("#tf-action").addEventListener("change", renderExtraOpts);
        renderExtraOpts();
    }

    async function loadStatistics() {
        const col = $("#stats-col-select").value;
        if (!col) return;
        try {
            const s = await api("/api/statistics?column=" + encodeURIComponent(col));
            renderStatsDetail(s);
        } catch (e) {
            toast(e.message, "error");
        }
    }

    function renderStatsDetail(s) {
        let html = '<div class="stats-grid">';
        const fields = s.dtype === "numeric"
            ? [
                ["Count", s.count], ["Missing", s.missing + ` (${s.missing_pct}%)`],
                ["Unique", s.unique], ["Mean", fmt(s.mean)], ["Median", fmt(s.median)],
                ["Mode", fmt(s.mode)], ["Std Dev", fmt(s.std)], ["Variance", fmt(s.variance)],
                ["Min", fmt(s.min)], ["Max", fmt(s.max)],
                ["Q25", fmt(s.q25)], ["Q50", fmt(s.q50)], ["Q75", fmt(s.q75)],
                ["IQR", fmt(s.iqr)], ["Skewness", fmt(s.skewness)], ["Kurtosis", fmt(s.kurtosis)],
            ]
            : [
                ["Count", s.count], ["Missing", s.missing + ` (${s.missing_pct}%)`],
                ["Unique", s.unique], ["Mode", s.mode],
            ];
        fields.forEach(([label, val]) => {
            html += `<div class="stat-card"><div class="stat-label">${label}</div><div class="stat-value">${val}</div></div>`;
        });
        html += "</div>";

        if (s.dtype !== "numeric" && s.top_values) {
            html += '<div style="margin-top:1rem"><strong>Top Values</strong>';
            html += '<table><thead><tr><th>Value</th><th>Count</th></tr></thead><tbody>';
            Object.entries(s.top_values).forEach(([k, v]) => {
                html += `<tr><td>${k}</td><td>${v}</td></tr>`;
            });
            html += "</tbody></table></div>";
        }
        $("#stats-detail").innerHTML = html;
    }

    // ── Distributions tab ───────────────────────────────────────────────

    async function loadDistribution() {
        const col = $("#dist-col-select").value;
        if (!col) return;
        try {
            const s = await api("/api/distribution?column=" + encodeURIComponent(col));
            renderDistChart(s);
        } catch (e) {
            toast(e.message, "error");
        }
    }

    function renderDistChart(s) {
        destroyChart("chart-distribution");
        const canvas = $("#chart-distribution");

        if (s.dtype === "numeric" && s.histogram) {
            const bins = s.histogram;
            chartInstances["chart-distribution"] = new Chart(canvas, {
                type: "bar",
                data: {
                    labels: bins.map((b) => `${fmt(b.lo)} – ${fmt(b.hi)}`),
                    datasets: [{
                        label: s.name,
                        data: bins.map((b) => b.count),
                        backgroundColor: "#6366f1aa",
                        borderColor: "#6366f1",
                        borderWidth: 1,
                    }],
                },
                options: barOpts(s.name + " distribution"),
            });
        } else if (s.top_values) {
            const entries = Object.entries(s.top_values).slice(0, 20);
            chartInstances["chart-distribution"] = new Chart(canvas, {
                type: "bar",
                data: {
                    labels: entries.map(([k]) => k),
                    datasets: [{
                        label: s.name,
                        data: entries.map(([, v]) => v),
                        backgroundColor: "#f59e0baa",
                        borderColor: "#f59e0b",
                        borderWidth: 1,
                    }],
                },
                options: barOpts(s.name + " value counts"),
            });
        }
    }

    // ── Correlations tab ────────────────────────────────────────────────

    async function loadCorrelation() {
        try {
            const data = await api("/api/correlation");
            renderCorrMatrix(data);
        } catch (e) {
            toast(e.message, "error");
        }
    }

    function renderCorrMatrix(data) {
        const { columns, matrix } = data;
        if (!columns.length) {
            $("#corr-matrix-wrap").innerHTML = '<p class="hint">No numeric columns for correlation.</p>';
            return;
        }
        let html = '<div class="corr-table"><table><thead><tr><th></th>';
        columns.forEach((c) => (html += `<th>${c}</th>`));
        html += "</tr></thead><tbody>";
        matrix.forEach((row, i) => {
            html += `<tr><th>${columns[i]}</th>`;
            row.forEach((v) => {
                const color = corrColor(v);
                html += `<td class="corr-cell" style="background:${color}">${v !== null ? v.toFixed(2) : "–"}</td>`;
            });
            html += "</tr>";
        });
        html += "</tbody></table></div>";
        $("#corr-matrix-wrap").innerHTML = html;
    }

    function corrColor(v) {
        if (v === null) return "transparent";
        if (v > 0) return `rgba(99, 102, 241, ${Math.abs(v) * 0.6})`;
        return `rgba(239, 68, 68, ${Math.abs(v) * 0.6})`;
    }

    // ── Transforms tab ──────────────────────────────────────────────────

    function renderExtraOpts() {
        const action = $("#tf-action").value;
        let html = "";
        if (action === "log") {
            html = `<div class="control-row"><label>Base:
                <select id="tf-log-base">
                    <option value="ln">ln (natural)</option>
                    <option value="log2">log2</option>
                    <option value="log10">log10</option>
                    <option value="log1p">log1p (ln(1+x))</option>
                </select></label></div>`;
        } else if (action === "fill_missing") {
            html = `<div class="control-row"><label>Strategy:
                <select id="tf-fill-strategy">
                    <option value="mean">Mean</option>
                    <option value="median">Median</option>
                    <option value="mode">Mode</option>
                    <option value="zero">Zero</option>
                    <option value="ffill">Forward Fill</option>
                </select></label></div>`;
        }
        $("#tf-extra-opts").innerHTML = html;
    }

    $("#tf-apply").addEventListener("click", async () => {
        const action = $("#tf-action").value;
        const column = $("#tf-col-select").value;
        const body = { action, column };

        if (action === "log") body.base = ($("#tf-log-base") || {}).value || "ln";
        if (action === "fill_missing") body.strategy = ($("#tf-fill-strategy") || {}).value || "mean";

        try {
            const res = await api("/api/transform", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            summary = res;
            const desc = `${action}(${column}${body.base ? ", " + body.base : ""}${body.strategy ? ", " + body.strategy : ""})`;
            transformHistory.push(desc);
            renderTransformHistory();
            renderSummary();
            populateSelects();
            toast("Transform applied: " + desc, "success");
        } catch (e) {
            toast(e.message, "error");
        }
    });

    $("#tf-reset").addEventListener("click", async () => {
        try {
            const res = await api("/api/transform", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "reset" }),
            });
            summary = res;
            transformHistory.push("RESET to original");
            renderTransformHistory();
            renderSummary();
            populateSelects();
            toast("Dataset reset to original", "success");
        } catch (e) {
            toast(e.message, "error");
        }
    });

    function renderTransformHistory() {
        $("#tf-history").innerHTML = transformHistory
            .map((t) => `<li>${t}</li>`)
            .join("");
    }

    // ── Data preview tab ────────────────────────────────────────────────

    async function loadPreview() {
        try {
            const data = await api(`/api/preview?offset=${previewOffset}&limit=${PAGE_SIZE}`);
            renderDataTable(data);
        } catch (e) {
            toast(e.message, "error");
        }
    }

    function renderDataTable(data) {
        const { columns, rows, total, offset } = data;
        $("#data-info").textContent = `Showing ${offset + 1}–${offset + rows.length} of ${total} rows`;

        let html = "<table><thead><tr>";
        columns.forEach((c) => (html += `<th>${c}</th>`));
        html += "</tr></thead><tbody>";
        rows.forEach((row) => {
            html += "<tr>";
            columns.forEach((c) => {
                const v = row[c];
                html += `<td>${v !== null && v !== undefined ? v : '<span style="color:var(--danger)">null</span>'}</td>`;
            });
            html += "</tr>";
        });
        html += "</tbody></table>";
        $("#data-table").innerHTML = html;

        const totalPages = Math.ceil(total / PAGE_SIZE);
        const curPage = Math.floor(offset / PAGE_SIZE);
        let pag = "";
        pag += `<button ${curPage === 0 ? "disabled" : ""} data-page="${curPage - 1}">← Prev</button>`;
        pag += `<span class="hint" style="margin:0 .5rem">Page ${curPage + 1} / ${totalPages}</span>`;
        pag += `<button ${curPage >= totalPages - 1 ? "disabled" : ""} data-page="${curPage + 1}">Next →</button>`;
        $("#pagination").innerHTML = pag;

        $$("#pagination button").forEach((btn) => {
            btn.addEventListener("click", () => {
                previewOffset = parseInt(btn.dataset.page) * PAGE_SIZE;
                loadPreview();
            });
        });
    }

    $("#export-btn").addEventListener("click", () => {
        window.location.href = "/api/export";
    });

    // ── Chart helpers ───────────────────────────────────────────────────

    function destroyChart(id) {
        if (chartInstances[id]) {
            chartInstances[id].destroy();
            delete chartInstances[id];
        }
    }

    function barOpts(title) {
        return {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: !!title, text: title, color: "#8b90a0" },
            },
            scales: {
                x: { ticks: { color: "#8b90a0", maxRotation: 45 }, grid: { color: "#2a2e3d" } },
                y: { ticks: { color: "#8b90a0" }, grid: { color: "#2a2e3d" } },
            },
        };
    }

    function fmt(v) {
        if (v === null || v === undefined) return "–";
        if (typeof v === "number") return Number.isInteger(v) ? v.toString() : v.toFixed(4);
        return v;
    }
})();
