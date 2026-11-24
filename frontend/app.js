/* 罕见事件分析前端：原生 JS，无外部依赖。
   图表为自绘 SVG，遵循 dataviz 规范：固定槽位颜色、单轴、细标记、
   悬停十字线/提示、表格孪生视图、文本不穿系列色。 */

const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x))
  ? "—" : x.toExponential(d);
const fmtFixed = (x, d = 3) => (x === null || x === undefined) ? "—" : Number(x).toFixed(d);
const STATUS_LABEL = {
  queued: "排队", running: "运行中", interrupted: "已中断（可恢复）",
  done: "已完成", cancelled: "已取消", failed: "失败",
};

let presets = {};
let currentTaskId = null;
let pollTimer = null;
const tooltip = document.createElement("div");
tooltip.className = "tooltip";
document.body.appendChild(tooltip);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.status);
  return res.json();
}

// ------------------------------------------------------------ 初始化
async function init() {
  try {
    presets = await api("/api/presets");
    $("conn-badge").textContent = "后端已连接";
    $("conn-badge").classList.add("ok");
  } catch (e) {
    $("conn-badge").textContent = "后端未连接";
    $("conn-badge").classList.add("err");
    return;
  }
  const sel = $("preset-select");
  sel.innerHTML = Object.entries(presets)
    .map(([id, p]) => `<option value="${id}">${p.label}</option>`).join("");
  sel.addEventListener("change", () => fillPreset(sel.value));
  $("btn-run").addEventListener("click", startTask);
  $("btn-cancel").addEventListener("click", cancelTask);
  $("btn-exact").addEventListener("click", runExact);
  fillPreset(sel.value);
  refreshTaskTable();
  setInterval(refreshTaskTable, 4000);
}

function fillPreset(id) {
  const p = presets[id];
  const c = p.config;
  $("in-capacity").value = c.capacity;
  $("in-horizon").value = c.horizon;
  $("in-n").value = p.default_n;
  $("in-pilot").value = p.pilot_n;
  $("in-shard").value = 2000;
  $("in-tilt").value = p.forced_tilt || "";
  $("in-seed").value = 20261125;
  $("scenario-desc").textContent = c.description || "";
  $("model-json").textContent = JSON.stringify(
    { arrival: c.arrival, service: c.service }, null, 2);
  $("cfg-disclaimer").textContent =
    "虚构模型教学模拟，不代表真实系统，不构成容量保证。";
}

function readConfig() {
  const presetId = $("preset-select").value;
  const base = presets[presetId].config;
  const overrides = {
    capacity: Number($("in-capacity").value),
    horizon: Number($("in-horizon").value),
  };
  const manual = $("in-tilt").value;
  return {
    scenario: { preset_id: presetId, overrides },
    n_total: Number($("in-n").value),
    pilot_n: Number($("in-pilot").value),
    shard_size: Number($("in-shard").value),
    seed: Number($("in-seed").value),
    manual_tilt: manual ? Number(manual) : undefined,
  };
}

// ------------------------------------------------------------ 任务控制
async function startTask() {
  const cfg = readConfig();
  const doc = await api("/api/tasks", { method: "POST", body: cfg });
  currentTaskId = doc.id;
  $("btn-cancel").disabled = false;
  showStatusCard(doc);
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollTask, 1200);
  pollTask();
}

async function cancelTask() {
  if (!currentTaskId) return;
  await api(`/api/tasks/${currentTaskId}/cancel`, { method: "POST" });
  $("btn-cancel").disabled = true;
}

async function pollTask() {
  if (!currentTaskId) return;
  let doc;
  try {
    doc = await api(`/api/tasks/${currentTaskId}`);
  } catch (e) { return; }
  renderStatus(doc);
  if (["done", "cancelled", "failed"].includes(doc.status)) {
    clearInterval(pollTimer);
    $("btn-cancel").disabled = true;
    if (doc.status === "done") renderResults(doc);
    refreshTaskTable();
  }
}

// ------------------------------------------------------------ 状态渲染
function showStatusCard(doc) {
  $("status-card").classList.remove("hidden");
  $("result-card").classList.add("hidden");
  $("trace-card").classList.add("hidden");
  renderStatus(doc);
}

function renderStatus(doc) {
  const pill = $("task-status");
  pill.textContent = STATUS_LABEL[doc.status] || doc.status;
  pill.className = `status-pill ${doc.status}`;
  if (doc.error) $("task-progress").textContent = "错误：" + doc.error;

  const n = doc.config.n_total;
  const rows = Object.values(doc.methods || {});
  const done = rows.reduce((s, m) => s + (m.acc ? m.acc.n : 0), 0);
  const denom = n * 2;
  const pct = Math.min(100, 100 * done / denom);
  $("progress-fill").style.width = pct.toFixed(1) + "%";
  const parts = rows.map((m) => `${m.method === "is" ? "IS" : "普通"} ${m.acc ? m.acc.n : 0}/${n}`);
  $("task-progress").textContent = parts.join(" · ") || "试运行中…";

  if (doc.pilot && doc.pilot.grid && doc.pilot.grid.length) {
    $("pilot-box").classList.remove("hidden");
    $("pilot-table").innerHTML = tableHtml(
      ["偏置 r", "命中数", "p̂", "标准误", "相对误差", "ESS 占比"],
      doc.pilot.grid.map((r) => [
        r.tilt.toFixed(2), String(r.hits), fmt(r.estimate), fmt(r.se),
        r.rel_error === null ? "NA" : r.rel_error.toFixed(3),
        r.ess_all_ratio.toExponential(2),
      ]),
      [false, true, true, true, true, true]);
    const note = doc.pilot.note ? `（${doc.pilot.note}）` : "";
    $("pilot-choice").innerHTML =
      `正式阶段固定偏置倍率 <code>r=${doc.pilot.chosen_tilt}</code>，试运行与正式估计使用独立随机流。${note}`
      + (doc.pilot.inconclusive ? " ⚠ 所有候选均零命中，选择不具信息量。" : "");
  }
}

// ------------------------------------------------------------ 结果渲染
function renderResults(doc) {
  $("result-card").classList.remove("hidden");
  const R = doc.results;
  const crude = R.methods.crude, ist = R.methods.is;

  // 可靠性横幅
  const banner = $("reliability-banner");
  banner.innerHTML = "";
  for (const [key, m] of [["普通模拟", crude], ["重要抽样", ist]]) {
    const rel = m.reliability;
    const div = document.createElement("div");
    div.className = "rel-item " + (rel.reliable ? "good" : "bad");
    div.innerHTML = `<span class="rel-icon">${rel.reliable ? "✓" : "⚠"}</span>
      <div><strong>${key}</strong>：${rel.reliable ? "估计可靠" : "存在可靠性问题"}
      ${rel.reasons.length ? "<br>" + rel.reasons.map(escapeHtml).join("<br>") : ""}</div>`;
    banner.appendChild(div);
  }

  // 统计块
  const tile = (title, dot, m) => {
    const e = m.estimates;
    const zero = e.hits === 0;
    return `<div class="stat-tile">
      <div class="label"><span class="dot ${dot}"></span>${title}</div>
      <div class="value">${zero ? "0（零命中）" : fmt(e.estimate)}</div>
      <div class="sub">95% CI：${zero ? "不可用" : `[${fmt(e.ci_low)}, ${fmt(e.ci_high)}]`}</div>
      <div class="sub">命中 ${e.hits}/${e.n} · ESS ${e.ess_all.toFixed(1)}
        (${(e.ess_all_ratio * 100).toFixed(2)}%)</div>
    </div>`;
  };
  $("stat-grid").innerHTML =
    tile("普通模拟 Crude MC", "crude", crude)
    + tile("重要抽样 IS（r=" + ist.tilt + "）", "is", ist);

  // 收敛曲线
  drawConvergence(R.methods);
  drawConvergenceTable(R.methods);

  // 费用对照表
  drawCompareTable(R, crude, ist);

  // 精确核对
  const exactBox = $("exact-box");
  if (R.exact && R.exact.probability !== undefined) {
    const rows = [crude, ist].map((m) => {
      const c = m.exact_check;
      return c ? `<tr><td>${m.method === "is" ? "重要抽样" : "普通模拟"}</td>
        <td class="num">${fmt(c.exact, 4)}</td>
        <td class="num">${fmt(c.estimate, 4)}</td>
        <td class="num">${fmt(c.absolute_error)}</td>
        <td class="num">${c.z_deviation === null ? "—" : c.z_deviation.toFixed(2)}</td>
        <td>${c.ci_covers_exact ? "✓ 覆盖" : "✗ 未覆盖"}</td></tr>` : "";
    }).join("");
    exactBox.innerHTML = `<strong>精确转移核对</strong>（状态数 ${R.exact.state_count}，
      截断行和最大误差 ${R.exact.row_sum_max_error.toExponential(1)}）
      <table class="data-table"><thead><tr>
      <th>方法</th><th class="num">精确 p</th><th class="num">估计 p̂</th>
      <th class="num">绝对误差</th><th class="num">z 偏差</th><th>CI 覆盖</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  } else {
    exactBox.innerHTML = "精确核对不可用：" + escapeHtml(R.exact?.reason || "状态空间超限");
  }

  // 零命中 3/n 上界
  $("rule-of-three").innerHTML =
    `普通模拟零命中时不能把“风险为零”写进结论；经验上界（3/n 规则）=
     <code>${fmt(R.rule_of_three_bound)}</code>。${escapeHtml(R.disclaimer || "")}`;

  // 路径与权重
  if (ist.traces && ist.traces.length) renderTrace(ist, doc.scenario);
  else $("trace-card").classList.add("hidden");
  drawWeightHist(ist.hit_logw_sample || []);
}

function drawCompareTable(R, crude, ist) {
  const ce = crude.estimates, ie = ist.estimates;
  const rows = [
    ["偏置倍率 r", "1.00（无偏置）", ist.tilt.toFixed(2)],
    ["完成路径数", String(ce.n), String(ie.n)],
    ["命中数", String(ce.hits), String(ie.hits)],
    ["估计 p̂", fmt(ce.estimate), fmt(ie.estimate)],
    ["标准误 SE", fmt(ce.se), fmt(ie.se)],
    ["相对半宽", ce.rel_halfwidth === null ? "—" : (ce.rel_halfwidth * 100).toFixed(1) + "%",
     ie.rel_halfwidth === null ? "—" : (ie.rel_halfwidth * 100).toFixed(1) + "%"],
    ["有效样本 ESS", ce.ess_all.toFixed(1), ie.ess_all.toFixed(1)],
    ["ESS 占比", (ce.ess_all_ratio * 100).toFixed(2) + "%",
     (ie.ess_all_ratio * 100).toFixed(2) + "%"],
    ["模拟时隙数", String(ce.slots), String(ie.slots)],
    ["RNG 调用数", String(crude.cost.rng_calls), String(ist.cost.rng_calls)],
    ["计算耗时 (s)", crude.cost.elapsed_seconds.toFixed(3),
     ist.cost.elapsed_seconds.toFixed(3)],
    ["时间 FOM (1/(var·s))", fmt(crude.cost.fom_time), fmt(ist.cost.fom_time)],
  ];
  let html = tableHtml(["指标", "普通模拟", "重要抽样"], rows, [false, false, false]);
  if (R.comparison) {
    const c = R.comparison;
    html += `<p class="hint">方差缩减 ${c.variance_reduction.toExponential(2)}× ·
      SE 缩减 ${c.se_reduction.toFixed(2)}× ·
      IS/普通 时间 FOM 比 ${c.fom_ratio_is_over_crude_time === null ? "—"
        : c.fom_ratio_is_over_crude_time.toFixed(2)}×
      （>1 表示单位计算时间内重要抽样更精确）</p>`;
  }
  $("compare-table").innerHTML = html;
}

function drawConvergenceTable(methods) {
  const conv = methods.crude.convergence;
  if (!conv.length) { $("convergence-table").innerHTML = '<p class="hint">暂无收敛数据</p>'; return; }
  const idx = (arr, n) => arr.reduce((best, r) => (r.n <= n ? r : best), arr[0]);
  const ns = methods.crude.convergence.map((r) => r.n)
    .filter((_, i) => i % Math.max(1, Math.floor(methods.crude.convergence.length / 8)) === 0);
  const rows = ns.map((n) => {
    const c = idx(methods.crude.convergence, n);
    const i = idx(methods.is.convergence, n);
    return [String(n), fmt(c.estimate), fmt(c.se), String(c.hits),
      fmt(i.estimate), fmt(i.se), String(i.hits)];
  });
  $("convergence-table").innerHTML = tableHtml(
    ["路径数 n", "普通 p̂", "普通 SE", "普通命中", "IS p̂", "IS SE", "IS 命中"],
    rows, [true, true, true, true, true, true, true]);
}

// ------------------------------------------------------------ 路径明细
function renderTrace(ist, scenario) {
  $("trace-card").classList.remove("hidden");
  const trace = ist.traces[0];
  const final = trace[trace.length - 1];
  $("trace-meta").innerHTML =
    `<span>路径长度：<strong>${trace.length}</strong> 时隙（时域上限 ${scenario.horizon}）</span>
     <span>容量 C=${scenario.capacity}</span>
     <span>最终对数权重 log W = <strong>${final.log_weight_cum.toFixed(4)}</strong>
       （W=${Math.exp(final.log_weight_cum).toExponential(3)}）</span>
     <span>偏置 r=${ist.tilt}</span>`;

  const head = ["t", "到达相位", "服务相位", "到达 a", "排队前 q",
    "溢出", "本时隙 log LR", "累计 log W", "服务后 q"];
  const body = trace.map((s) => [
    String(s.t),
    s.arrival_phase === null || s.arrival_phase === undefined ? "—" : String(s.arrival_phase),
    s.service_phase === null || s.service_phase === undefined ? "—" : String(s.service_phase),
    String(s.arrivals), String(s.queue_before), s.overflow ? "是" : "否",
    s.log_lr.toFixed(4), s.log_weight_cum.toFixed(4),
    s.queue_after === undefined ? "—" : String(s.queue_after),
  ]);
  $("trace-table").innerHTML = tableHtml(head, body,
    [true, true, true, true, true, false, true, true, true]);

  drawLrChart(trace);
  drawQueueChart(trace, scenario.capacity);
}

function drawWeightHist(samples) {
  const el = $("chart-hist");
  if (!samples.length) { el.innerHTML = '<p class="hint">无命中权重样本</p>'; return; }
  const bins = 24;
  const lo = Math.min(...samples), hi = Math.max(...samples);
  const width = (hi - lo) / bins || 1;
  const counts = Array(bins).fill(0);
  samples.forEach((v) => {
    const b = Math.min(bins - 1, Math.floor((v - lo) / width));
    counts[b]++;
  });
  const W = 720, H = 240, pad = { l: 52, r: 16, t: 14, b: 38 };
  const maxC = Math.max(...counts);
  const x = (i) => pad.l + i * (W - pad.l - pad.r) / bins;
  const y = (c) => pad.t + (H - pad.t - pad.b) * (1 - c / maxC);
  const bw = (W - pad.l - pad.r) / bins - 2;
  let bars = counts.map((c, i) =>
    `<rect class="is-fill" x="${x(i) + 1}" y="${y(c)}" width="${bw}" height="${H - pad.b - y(c)}"
       rx="4" data-tip="log W ∈ [${(lo + i * width).toFixed(2)}, ${(lo + (i + 1) * width).toFixed(2)})&#10;频数 ${c}"/>`).join("");
  el.innerHTML = svgShell(W, H) +
    gridY(H, pad, maxC, W) + bars +
    axisX(pad, W, H, (i) => (lo + i * width).toFixed(1), bins, 6) +
    `<text class="axis-title" x="${pad.l}" y="${H - 4}">命中路径对数权重 log W</text>
     <text class="axis-title" transform="rotate(-90 14 ${pad.t + (H - pad.t - pad.b) / 2})"
       x="14" y="${pad.t + (H - pad.t - pad.b) / 2}">频数</text></svg>`;
  bindTooltips(el);
}

// ------------------------------------------------------------ SVG 图表
function svgShell(w, h) {
  return `<svg viewBox="0 0 ${w} ${h}" role="img" width="${w}" height="${h}">`;
}

function gridY(H, pad, maxV, width) {
  const ticks = 4;
  let out = "";
  for (let i = 0; i <= ticks; i++) {
    const v = maxV * i / ticks;
    const yy = pad.t + (H - pad.t - pad.b) * (1 - i / ticks);
    out += `<line class="grid-line" x1="${pad.l}" x2="${width - pad.r}" y1="${yy}" y2="${yy}"/>
      <text class="tick-text" x="${pad.l - 6}" y="${yy + 3}" text-anchor="end">${v >= 1 ? v.toFixed(0) : v.toExponential(1)}</text>`;
  }
  return out;
}

function axisX(pad, W, H, labelFn, steps, labels) {
  let out = `<line class="axis-line" x1="${pad.l}" x2="${W - pad.r}" y1="${H - pad.b}" y2="${H - pad.b}"/>`;
  const n = Math.min(labels, steps);
  for (let i = 0; i <= n; i++) {
    const idx = Math.round(i * (steps - 1) / n);
    const xx = pad.l + idx * (W - pad.l - pad.r) / Math.max(1, steps - 1);
    out += `<text class="tick-text" x="${xx}" y="${H - pad.b + 15}" text-anchor="middle">${labelFn(idx)}</text>`;
  }
  return out;
}

function drawConvergence(methods) {
  const el = $("chart-convergence");
  const W = 760, H = 300, pad = { l: 64, r: 18, t: 18, b: 42 };
  const series = [
    { key: "crude", data: methods.crude.convergence, cls: "crude-line", band: "ci-band", name: "普通模拟" },
    { key: "is", data: methods.is.convergence, cls: "is-line", band: "ci-band-is", name: "重要抽样" },
  ];
  const allN = series.flatMap((s) => s.data.map((d) => d.n));
  const allP = series.flatMap((s) => s.data.map((d) => d.estimate).filter((v) => v > 0));
  if (!allN.length || !allP.length) {
    el.innerHTML = '<p class="hint">两种方法均零命中，无法绘制非零收敛曲线；请参考 3/n 经验上界。</p>';
    return;
  }
  const xMin = Math.min(...allN), xMax = Math.max(...allN);
  const yMin = Math.min(...allP) * 0.6, yMax = Math.max(...allP) * 1.6;
  const X = (n) => pad.l + (n - xMin) / (xMax - xMin || 1) * (W - pad.l - pad.r);
  const Y = (p) => pad.t + (H - pad.t - pad.b)
    * (1 - (Math.log(p) - Math.log(yMin)) / (Math.log(yMax) - Math.log(yMin)));

  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const lv = Math.log(yMin) + i * (Math.log(yMax) - Math.log(yMin)) / 4;
    const yy = Y(Math.exp(lv));
    grid += `<line class="grid-line" x1="${pad.l}" x2="${W - pad.r}" y1="${yy}" y2="${yy}"/>
      <text class="tick-text" x="${pad.l - 8}" y="${yy + 3}" text-anchor="end">${Math.exp(lv).toExponential(1)}</text>`;
  }
  let body = grid + `<line class="axis-line" x1="${pad.l}" x2="${W - pad.r}" y1="${H - pad.b}" y2="${H - pad.b}"/>`;
  for (const s of series) {
    const pts = s.data.filter((d) => d.estimate > 0);
    if (!pts.length) continue;
    const line = pts.map((d) => `${X(d.n)},${Y(d.estimate)}`).join(" ");
    const bandPts = pts.map((d) => `${X(d.n)},${Y(Math.max(d.estimate + 1.96 * d.se, yMin * 1.01))}`)
      .concat(pts.slice().reverse().map((d) => `${X(d.n)},${Y(Math.max(d.estimate - 1.96 * d.se, yMin))}`));
    body += `<polygon class="${s.band}" points="${bandPts.join(" ")}"/>`;
    body += `<polyline class="${s.cls}" points="${line}" fill="none" stroke-width="2"/>`;
    const last = pts[pts.length - 1];
    body += `<circle cx="${X(last.n)}" cy="${Y(last.estimate)}" r="4"
      class="${s.key === "is" ? "is-fill" : "crude-fill"}"/>
      <text class="series-label ${s.cls}" x="${X(last.n) - 6}" y="${Y(last.estimate) - 8}"
        text-anchor="end" fill="${cssVar(s.key === "is" ? "--series-is" : "--series-crude")}">${s.name}</text>`;
  }
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const n = Math.round(xMin + i * (xMax - xMin) / ticks);
    body += `<text class="tick-text" x="${X(n)}" y="${H - pad.b + 16}" text-anchor="middle">${n}</text>`;
  }
  body += `<text class="axis-title" x="${(W) / 2}" y="${H - 6}">完成路径数 n</text>
    <text class="axis-title" transform="rotate(-90 16 ${pad.t + (H - pad.t - pad.b) / 2})"
      x="16" y="${pad.t + (H - pad.t - pad.b) / 2}">p̂（对数轴），色带为 95% CI</text>`;
  el.innerHTML = svgShell(W, H) + body + "</svg>"
    + `<div class="legend"><span><span class="dot crude"></span>普通模拟</span>
       <span><span class="dot is"></span>重要抽样</span></div>`;
}

function drawLrChart(trace) {
  const el = $("chart-lr");
  const W = 760, H = 220, pad = { l: 56, r: 16, t: 16, b: 38 };
  const maxV = Math.max(0.1, ...trace.map((s) => Math.abs(s.log_lr)));
  const X = (t) => pad.l + t / Math.max(1, trace.length - 1) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (H - pad.t - pad.b) * (1 - (v + maxV) / (2 * maxV));
  const bw = Math.max(2, (W - pad.l - pad.r) / trace.length - 2);
  let bars = trace.map((s) => {
    const zero = Y(0);
    const yy = s.log_lr >= 0 ? Y(s.log_lr) : zero;
    const h = Math.abs(Y(s.log_lr) - zero);
    const cls = s.overflow ? "bar-overflow" : "is-fill";
    return `<rect class="${cls}" x="${X(s.t) + 1}" y="${yy}" width="${bw}" height="${Math.max(1, h)}"
      rx="2" data-tip="t=${s.t}&#10;到达=${s.arrivals}&#10;log LR=${s.log_lr.toFixed(4)}&#10;${s.overflow ? "触发溢出，路径终止" : ""}"/>`;
  }).join("");
  const zero = Y(0);
  el.innerHTML = svgShell(W, H)
    + `<line class="axis-line" x1="${pad.l}" x2="${W - pad.r}" y1="${zero}" y2="${zero}"/>`
    + `<text class="tick-text" x="${pad.l - 8}" y="${zero + 3}" text-anchor="end">0</text>
       <text class="tick-text" x="${pad.l - 8}" y="${pad.t + 4}" text-anchor="end">${maxV.toFixed(1)}</text>
       <text class="tick-text" x="${pad.l - 8}" y="${H - pad.b}" text-anchor="end">${(-maxV).toFixed(1)}</text>`
    + bars + axisX(pad, W, H, (i) => String(trace[Math.min(i, trace.length - 1)].t),
      trace.length, Math.min(8, trace.length))
    + `<text class="axis-title" x="${W / 2}" y="${H - 4}">时隙 t</text></svg>`
    + `<div class="legend"><span><span class="dot is"></span>常规到达抽样</span>
       <span><span class="dot" style="background:${cssVar("--critical")}"></span>溢出抽样</span></div>`;
  bindTooltips(el);
}

function drawQueueChart(trace, capacity) {
  const el = $("chart-queue");
  const W = 760, H = 220, pad = { l: 56, r: 16, t: 16, b: 38 };
  const maxV = capacity + 2;
  const X = (t) => pad.l + t / Math.max(1, trace.length - 1) * (W - pad.l - pad.r);
  const Y = (q) => pad.t + (H - pad.t - pad.b) * (1 - q / maxV);
  const pts = trace.map((s) => `${X(s.t)},${Y(s.queue_before)}`).join(" ");
  const capY = Y(capacity);
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const q = maxV * i / 4;
    grid += `<line class="grid-line" x1="${pad.l}" x2="${W - pad.r}" y1="${Y(q)}" y2="${Y(q)}"/>
      <text class="tick-text" x="${pad.l - 8}" y="${Y(q) + 3}" text-anchor="end">${q.toFixed(0)}</text>`;
  }
  const dots = trace.map((s) =>
    `<circle cx="${X(s.t)}" cy="${Y(s.queue_before)}" r="${s.overflow ? 6 : 3.5}"
      class="${s.overflow ? "bar-overflow" : "crude-fill"}"
      data-tip="t=${s.t}&#10;排队前=${s.queue_before}&#10;到达=${s.arrivals}&#10;${s.overflow ? "q+a&gt;C 溢出" : ""}"/>`).join("");
  el.innerHTML = svgShell(W, H) + grid
    + `<line x1="${pad.l}" x2="${W - pad.r}" y1="${capY}" y2="${capY}"
        stroke="${cssVar("--critical")}" stroke-width="1.5" stroke-dasharray="6 4"/>
       <text class="tick-text" x="${W - pad.r}" y="${capY - 5}" text-anchor="end"
        fill="${cssVar("--critical")}">容量 C=${capacity}</text>`
    + `<polyline class="crude-line" points="${pts}" fill="none" stroke-width="2"/>`
    + dots + axisX(pad, W, H, (i) => String(trace[Math.min(i, trace.length - 1)].t),
      trace.length, Math.min(8, trace.length))
    + `<text class="axis-title" x="${W / 2}" y="${H - 4}">时隙 t</text>
       <text class="axis-title" transform="rotate(-90 16 ${pad.t + (H - pad.t - pad.b) / 2})"
         x="16" y="${pad.t + (H - pad.t - pad.b) / 2}">队列长度</text></svg>`;
  bindTooltips(el);
}

// ------------------------------------------------------------ 表格与提示
function tableHtml(head, rows, rightFlags) {
  const th = head.map((h, i) => `<th class="${rightFlags && rightFlags[i] ? "num" : ""}">${h}</th>`).join("");
  const tr = rows.map((r) => "<tr>" + r.map((c, i) =>
    `<td class="${rightFlags && rightFlags[i] ? "num" : ""}">${c}</td>`).join("") + "</tr>").join("");
  return `<table class="data-table"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
}

function bindTooltips(container) {
  container.querySelectorAll("[data-tip]").forEach((node) => {
    node.addEventListener("mousemove", (e) => {
      tooltip.style.display = "block";
      tooltip.innerHTML = node.getAttribute("data-tip").replaceAll("&#10;", "<br>");
      tooltip.style.left = Math.min(e.clientX + 14, window.innerWidth - 340) + "px";
      tooltip.style.top = (e.clientY + 14) + "px";
    });
    node.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ------------------------------------------------------------ 精确计算（独立按钮）
async function runExact() {
  const cfg = readConfig();
  // 深拷贝避免污染内存中的预设对象
  const scenario = JSON.parse(JSON.stringify(presets[cfg.scenario.preset_id].config));
  scenario.capacity = cfg.scenario.overrides.capacity;
  scenario.horizon = cfg.scenario.overrides.horizon;
  $("result-card").classList.remove("hidden");
  try {
    const res = await api("/api/exact", { method: "POST", body: { scenario } });
    $("exact-box").innerHTML = `<strong>精确首次溢出概率</strong>
      p = ${res.probability.toExponential(4)}（状态数 ${res.state_count}，
      截断行和最大误差 ${res.row_sum_max_error.toExponential(1)}）`;
  } catch (e) {
    $("exact-box").innerHTML = `<span style="color:${cssVar("--critical")}">${escapeHtml(e.message)}</span>`;
  }
}

// ------------------------------------------------------------ 历史任务表
async function refreshTaskTable() {
  const { tasks } = await api("/api/tasks");
  const rows = tasks.map((t) => [
    `<button class="ghost" data-load="${t.id}" style="padding:2px 8px">载入</button>`,
    t.scenario.name,
    `<span class="pill-inline">${STATUS_LABEL[t.status] || t.status}</span>`,
    String(t.n_total),
    new Date(t.created_at * 1000).toLocaleString(),
  ]);
  $("task-table").innerHTML = tableHtml(
    ["", "场景", "状态", "路径数/方法", "创建时间"], rows,
    [false, false, false, true, false]);
  $("task-table").querySelectorAll("[data-load]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      currentTaskId = btn.getAttribute("data-load");
      const doc = await api(`/api/tasks/${currentTaskId}`);
      showStatusCard(doc);
      if (doc.status === "done") renderResults(doc);
      if (["queued", "interrupted", "cancelled"].includes(doc.status)) {
        await api(`/api/tasks/${currentTaskId}/start`, { method: "POST" });
        $("btn-cancel").disabled = false;
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(pollTask, 1200);
        pollTask();
      }
    });
  });
}

init();
