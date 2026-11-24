/* Metric maths and small accessible SVG charts. No external scripts, analytics, or charting runtime. */
(function (root) {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const valid = p => (p.label === 0 || p.label === 1) && Number.isFinite(p.score) && p.score >= 0 && p.score <= 1;

  // ---- Metric maths -------------------------------------------------------------------------
  function curves(predictions) {
    const items = predictions.filter(valid);
    const positive = items.filter(p => p.label === 1).length, negative = items.length - positive;
    if (!positive || !negative) return null;
    const sorted = [...items].sort((a, b) => b.score - a.score);
    const roc = [[0, 0]], pr = [[0, 1]];
    let tp = 0, fp = 0, ap = 0, lastRecall = 0;
    for (let i = 0; i < sorted.length;) {
      const score = sorted[i].score;
      do { sorted[i].label === 1 ? tp++ : fp++; i++; } while (i < sorted.length && sorted[i].score === score);
      const recall = tp / positive, precision = tp / (tp + fp);
      roc.push([fp / negative, recall]); pr.push([recall, precision]);
      ap += (recall - lastRecall) * precision; lastRecall = recall;
    }
    return {roc, pr, ap, prevalence: positive / items.length};
  }

  function histogram(predictions, bins = 10) {
    const result = Array.from({length: bins}, (_, i) => ({x: (i + .5) / bins, x0: i / bins, x1: (i + 1) / bins, real: 0, fake: 0}));
    for (const p of predictions) if (valid(p)) result[Math.min(bins - 1, Math.floor(p.score * bins))][p.label === 1 ? 'fake' : 'real']++;
    return result;
  }

  /** Mann-Whitney ROC-AUC with tied scores counted as half. */
  function auc(predictions) {
    const items = predictions.filter(valid).sort((a, b) => a.score - b.score);
    const positive = items.filter(p => p.label === 1).length, negative = items.length - positive;
    if (!positive || !negative) return null;
    let rankSum = 0;
    for (let i = 0; i < items.length;) {
      let j = i;
      while (j < items.length && items[j].score === items[i].score) j++;
      const rank = (i + j + 1) / 2;
      for (let k = i; k < j; k++) if (items[k].label === 1) rankSum += rank;
      i = j;
    }
    return (rankSum - positive * (positive + 1) / 2) / (positive * negative);
  }

  function random(seed) { // mulberry32: seeded so the interval does not change on every refresh
    return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }

  /** 95% percentile bootstrap interval for ROC-AUC, resampling videos with replacement. */
  function bootstrapAuc(predictions, resamples = 300, seed = 7) {
    const items = predictions.filter(valid);
    if (auc(items) === null) return null;
    const next = random(seed), values = [];
    for (let r = 0; r < resamples; r++) {
      const sample = Array.from({length: items.length}, () => items[Math.floor(next() * items.length)]);
      const value = auc(sample);
      if (value !== null) values.push(value);
    }
    values.sort((a, b) => a - b);
    return values.length ? [values[Math.floor(values.length * .025)], values[Math.min(values.length - 1, Math.ceil(values.length * .975) - 1)]] : null;
  }

  function confusion(predictions, threshold) {
    const c = {tp: 0, fp: 0, tn: 0, fn: 0};
    for (const p of predictions) if (valid(p)) c[p.score >= threshold ? (p.label ? 'tp' : 'fp') : (p.label ? 'fn' : 'tn')]++;
    return c;
  }

  function rates({tp, fp, tn, fn}) {
    const div = (a, b) => b ? a / b : null;
    const precision = div(tp, tp + fp), recall = div(tp, tp + fn), specificity = div(tn, tn + fp);
    const mccDen = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    return {count: tp + fp + tn + fn, accuracy: div(tp + tn, tp + fp + tn + fn), precision, recall, specificity,
      fpr: div(fp, fp + tn), f1: precision !== null && recall !== null && precision + recall ? 2 * precision * recall / (precision + recall) : null,
      balanced: recall !== null && specificity !== null ? (recall + specificity) / 2 : null, mcc: mccDen ? (tp * tn - fp * fn) / mccDen : null};
  }

  /** Reliability bins, expected calibration error, Brier score and log loss of the raw scores. */
  function calibration(predictions, bins = 10) {
    const items = predictions.filter(valid);
    if (!items.length) return null;
    const groups = Array.from({length: bins}, (_, i) => ({lo: i / bins, hi: (i + 1) / bins, n: 0, scoreSum: 0, positives: 0}));
    let brier = 0, logloss = 0;
    for (const p of items) {
      const g = groups[Math.min(bins - 1, Math.floor(p.score * bins))];
      g.n++; g.scoreSum += p.score; g.positives += p.label;
      brier += (p.score - p.label) ** 2;
      const s = Math.min(1 - 1e-7, Math.max(1e-7, p.score));
      logloss -= p.label ? Math.log(s) : Math.log(1 - s);
    }
    const used = groups.filter(g => g.n).map(g => ({...g, mean: g.scoreSum / g.n, frac: g.positives / g.n}));
    const ece = used.reduce((sum, g) => sum + g.n / items.length * Math.abs(g.frac - g.mean), 0);
    return {bins: used, ece, brier: brier / items.length, logloss: logloss / items.length};
  }

  function sweep(predictions, steps = 100) {
    return Array.from({length: steps + 1}, (_, i) => ({t: i / steps, ...rates(confusion(predictions, i / steps))}));
  }

  /** Exact two-sided McNemar test on the discordant pairs b and c. */
  function mcnemar(b, c) {
    const n = b + c;
    if (!n) return 1;
    let logFact = [0];
    for (let i = 1; i <= n; i++) logFact[i] = logFact[i - 1] + Math.log(i);
    let tail = 0;
    for (let k = 0; k <= Math.min(b, c); k++) tail += Math.exp(logFact[n] - logFact[k] - logFact[n - k] - n * Math.LN2);
    return Math.min(1, 2 * tail);
  }

  // ---- Rendering helpers ----------------------------------------------------------------------
  function node(name, attrs = {}, text) {
    const e = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function html(tag, text, className) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (className) e.className = className; return e; }
  function ticks(min, max, count = 5) {
    const span = max - min || 1, raw = span / (count - 1), mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const fmt = (v, digits = 2) => Number.isFinite(v) ? (Math.abs(v) >= 100 ? Math.round(v).toLocaleString('en-IN') : +v.toFixed(digits) + '') : '–';

  function legend(container, items) {
    if (items.length < 2) return;
    const key = html('div', undefined, 'key');
    for (const item of items) { const span = html('span'); const dot = html('i', undefined, item.key); span.append(dot, item.name); key.append(span); }
    container.append(key);
  }

  function tableView(container, headers, rows) {
    if (!rows.length) return;
    const details = html('details', undefined, 'table-view');
    details.append(html('summary', 'Show as table'));
    const wrap = html('div', undefined, 'table-wrap'), table = html('table'), head = html('tr');
    headers.forEach(h => { const th = html('th', h); th.scope = 'col'; head.append(th); });
    const thead = html('thead'); thead.append(head); table.append(thead);
    const body = html('tbody');
    rows.forEach(row => { const tr = html('tr'); row.forEach(v => tr.append(html('td', typeof v === 'number' ? fmt(v, 3) : String(v)))); body.append(tr); });
    table.append(body); wrap.append(table); details.append(wrap); container.append(details);
  }

  function frame(container, {width = 560, height = 300, title}) {
    width = width || 560; height = height || 300;
    container.replaceChildren();
    const m = {l: 50, r: 18, t: 18, b: 46};
    const svg = node('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': title, tabindex: 0});
    svg.append(node('title', {}, title));
    container.append(svg);
    return {svg, m, w: width, h: height, pw: width - m.l - m.r, ph: height - m.t - m.b};
  }

  function axes(f, x, y) {
    const sx = v => f.m.l + (v - x.min) / (x.max - x.min || 1) * f.pw;
    const sy = v => f.m.t + f.ph - (v - y.min) / (y.max - y.min || 1) * f.ph;
    for (const v of ticks(y.min, y.max)) {
      f.svg.append(node('line', {x1: f.m.l, x2: f.m.l + f.pw, y1: sy(v), y2: sy(v), class: v === y.min ? 'base' : 'grid'}));
      f.svg.append(node('text', {x: f.m.l - 8, y: sy(v) + 4, 'text-anchor': 'end', class: 'tick'}, (y.format || fmt)(v)));
    }
    for (const v of x.ticks || ticks(x.min, x.max)) f.svg.append(node('text', {x: sx(v), y: f.m.t + f.ph + 18, 'text-anchor': 'middle', class: 'tick'}, (x.format || fmt)(v)));
    f.svg.append(node('text', {x: f.m.l + f.pw / 2, y: f.h - 6, 'text-anchor': 'middle', class: 'axis-label'}, x.label || ''));
    f.svg.append(node('text', {transform: `translate(13 ${f.m.t + f.ph / 2}) rotate(-90)`, 'text-anchor': 'middle', class: 'axis-label'}, y.label || ''));
    return {sx, sy};
  }

  function decorations(f, {sx, sy}, x, y, {bands = [], rules = [], diagonal = false}) {
    for (const b of bands) {
      const x0 = sx(Math.max(x.min, b.x0)), x1 = sx(Math.min(x.max, b.x1));
      f.svg.append(node('rect', {x: x0, y: f.m.t, width: Math.max(0, x1 - x0), height: f.ph, class: 'band'}));
    }
    if (diagonal) f.svg.append(node('line', {x1: sx(x.min), y1: sy(y.min), x2: sx(x.max), y2: sy(y.max), class: 'chance'}));
    for (const r of rules) {
      if (!(r.x >= x.min && r.x <= x.max)) continue;
      f.svg.append(node('line', {x1: sx(r.x), x2: sx(r.x), y1: f.m.t, y2: f.m.t + f.ph, class: 'rule-line'}));
      const right = sx(r.x) > f.m.l + f.pw * .72;  // label sits at the foot of the line, clear of curves near the top
      f.svg.append(node('text', {x: sx(r.x) + (right ? -5 : 5), y: f.m.t + f.ph - 6, 'text-anchor': right ? 'end' : 'start', class: 'rule-text'}, r.label));
    }
  }

  function tooltip(container) {
    const tip = html('div', undefined, 'tooltip'); tip.hidden = true; container.append(tip); return tip;
  }
  function placeTip(container, tip, svg, vx, vy, lines) {
    tip.replaceChildren();
    lines.forEach((line, i) => {
      const row = html('div');
      if (line.key) { const dot = html('i', undefined, line.key); row.append(dot); }
      if (i === 0) row.append(html('b', line.text)); else row.append(line.text);
      tip.append(row);
    });
    const box = svg.getBoundingClientRect(), outer = container.getBoundingClientRect();
    const scale = box.width / svg.viewBox.baseVal.width;
    tip.hidden = false;
    let left = box.left - outer.left + vx * scale;
    const half = tip.offsetWidth / 2;
    left = Math.max(half, Math.min(outer.width - half, left));
    tip.style.left = `${left}px`; tip.style.top = `${box.top - outer.top + vy * scale}px`;
  }

  /** Line chart: series [{name, key, points: [[x, y]], markers}], shared x and y scales, hover crosshair. */
  function lineChart(container, opts) {
    const {series = [], x = {min: 0, max: 1}, y = {min: 0, max: 1}, title = 'Chart', empty = 'No data yet.'} = opts;
    const f = frame(container, {title, width: opts.width, height: opts.height});
    const s = axes(f, x, y);
    decorations(f, s, x, y, opts);
    const drawn = series.map(se => ({...se, points: se.points.filter(p => p.every(Number.isFinite))})).filter(se => se.points.length);
    if (!drawn.length) { f.svg.append(node('text', {x: f.m.l + f.pw / 2, y: f.m.t + f.ph / 2, 'text-anchor': 'middle', class: 'empty-text'}, empty)); return f.svg; }
    for (const se of drawn) {
      const g = node('g', {class: se.key});
      if (se.points.length > 1) g.append(node('polyline', {points: se.points.map(p => `${s.sx(p[0]).toFixed(1)},${s.sy(p[1]).toFixed(1)}`).join(' '), class: 'series'}));
      if (se.markers || se.points.length === 1) se.points.forEach(p => g.append(node('circle', {cx: s.sx(p[0]), cy: s.sy(p[1]), r: 4, class: 'marker'})));
      f.svg.append(g);
    }
    // Hover and keyboard layer: crosshair at the nearest x, one value per series.
    const cross = node('line', {y1: f.m.t, y2: f.m.t + f.ph, class: 'cross', visibility: 'hidden'});
    const dots = drawn.map(se => { const d = node('circle', {r: 5, class: `hover-dot ${se.key}`, visibility: 'hidden'}); return d; });
    f.svg.append(cross, ...dots);
    const hit = node('rect', {x: f.m.l, y: f.m.t, width: f.pw, height: f.ph, fill: 'transparent'});
    f.svg.append(hit);
    const tip = tooltip(container), anchor = drawn[0].points;
    const show = xValue => {
      const lines = [{text: `${x.label || 'x'} ${(x.format || fmt)(xValue)}`}];
      let top = f.m.t + f.ph;
      drawn.forEach((se, i) => {
        const p = se.points.reduce((best, q) => Math.abs(q[0] - xValue) < Math.abs(best[0] - xValue) ? q : best);
        dots[i].setAttribute('cx', s.sx(p[0])); dots[i].setAttribute('cy', s.sy(p[1])); dots[i].setAttribute('visibility', 'visible');
        top = Math.min(top, s.sy(p[1]));
        lines.push({key: se.key, text: `${se.name}: ${(y.format || fmt)(p[1])}`});
      });
      cross.setAttribute('x1', s.sx(xValue)); cross.setAttribute('x2', s.sx(xValue)); cross.setAttribute('visibility', 'visible');
      placeTip(container, tip, f.svg, s.sx(xValue), top, lines);
    };
    const hide = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); };
    hit.addEventListener('pointermove', event => {
      const box = f.svg.getBoundingClientRect(), vx = (event.clientX - box.left) / box.width * f.w;
      show(x.min + (Math.max(f.m.l, Math.min(f.m.l + f.pw, vx)) - f.m.l) / f.pw * (x.max - x.min));
    });
    hit.addEventListener('pointerleave', hide);
    let index = anchor.length - 1;
    f.svg.addEventListener('keydown', event => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      index = Math.max(0, Math.min(anchor.length - 1, index + (event.key === 'ArrowRight' ? 1 : -1)));
      show(anchor[index][0]);
    });
    f.svg.addEventListener('focus', () => show(anchor[index][0]));
    f.svg.addEventListener('blur', hide);
    legend(container, [...drawn.map(se => ({name: se.name, key: se.key})), ...(opts.legendExtra || [])]);
    const step = Math.max(1, Math.ceil(anchor.length / 60));
    tableView(container, [x.label || 'x', ...drawn.map(se => se.name)], anchor.filter((_, i) => i % step === 0 || i === anchor.length - 1).map(p =>
      [p[0], ...drawn.map(se => { const q = se.points.reduce((best, r) => Math.abs(r[0] - p[0]) < Math.abs(best[0] - p[0]) ? r : best); return q[1]; })]));
    return f.svg;
  }

  /** Grouped bars over continuous bins: bins [{x0, x1}], series [{name, key, values}]. */
  function binChart(container, opts) {
    const {bins = [], series = [], x = {min: 0, max: 1}, title = 'Chart', empty = 'No data yet.'} = opts;
    const total = series.reduce((n, se) => n + se.values.reduce((a, b) => a + b, 0), 0);
    const yMax = Math.max(1, ...series.flatMap(se => se.values));
    const y = {min: 0, max: ticks(0, yMax).at(-1) >= yMax ? ticks(0, yMax).at(-1) : yMax, label: opts.yLabel || 'Count'};
    const f = frame(container, {title, height: opts.height});
    const s = axes(f, x, y);
    decorations(f, s, x, y, opts);
    if (!total) { f.svg.append(node('text', {x: f.m.l + f.pw / 2, y: f.m.t + f.ph / 2, 'text-anchor': 'middle', class: 'empty-text'}, empty)); return f.svg; }
    const tip = tooltip(container);
    bins.forEach((bin, i) => {
      const left = s.sx(bin.x0) + 2, width = (s.sx(bin.x1) - s.sx(bin.x0) - 4 - 2 * (series.length - 1)) / series.length;
      series.forEach((se, j) => {
        const value = se.values[i], top = s.sy(value), bottom = s.sy(0), bx = left + j * (width + 2), r = Math.min(4, width / 2, bottom - top);
        if (!value) return;
        const path = `M${bx},${bottom}V${top + r}Q${bx},${top} ${bx + r},${top}H${bx + width - r}Q${bx + width},${top} ${bx + width},${top + r}V${bottom}Z`;
        const bar = node('path', {d: path, class: `bar ${se.key}`, tabindex: -1});
        const label = `${se.name}: ${value} video${value === 1 ? '' : 's'} with scores ${fmt(bin.x0)}–${fmt(bin.x1)}`;
        bar.append(node('title', {}, label));
        bar.addEventListener('pointerenter', () => placeTip(container, tip, f.svg, bx + width / 2, top, [{text: `Score ${fmt(bin.x0)}–${fmt(bin.x1)}`}, {key: se.key, text: `${se.name}: ${value}`}]));
        bar.addEventListener('pointerleave', () => { tip.hidden = true; });
        f.svg.append(bar);
      });
    });
    legend(container, [...series.map(se => ({name: se.name, key: se.key})), ...(opts.legendExtra || [])]);
    tableView(container, ['Score range', ...series.map(se => se.name)], bins.map((bin, i) => [`${fmt(bin.x0)}–${fmt(bin.x1)}`, ...series.map(se => se.values[i])]));
    return f.svg;
  }

  /** SVG with computed colours written in, so a downloaded chart looks the same outside the page. */
  function snapshot(svg) {
    const clone = svg.cloneNode(true), source = svg.querySelectorAll('*'), target = clone.querySelectorAll('*');
    source.forEach((el, i) => {
      const style = getComputedStyle(el);
      for (const prop of ['fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'opacity', 'font-family', 'font-size']) {
        const value = style.getPropertyValue(prop);
        if (value) target[i].setAttribute(prop, value);
      }
    });
    clone.setAttribute('xmlns', NS);
    const background = getComputedStyle(svg.closest('.sheet') || document.body).backgroundColor;
    clone.insertBefore(node('rect', {width: '100%', height: '100%', fill: background}), clone.firstChild);
    return new XMLSerializer().serializeToString(clone);
  }

  const api = {curves, histogram, auc, bootstrapAuc, confusion, rates, calibration, sweep, mcnemar, lineChart, binChart, snapshot, fmt};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FGCharts = api;
})(typeof window !== 'undefined' ? window : globalThis);
