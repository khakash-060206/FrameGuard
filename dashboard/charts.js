/* Small SVG plots; no external scripts, analytics, or charting runtime. */
(function (root) {
  'use strict';
  const colors = ['#087f78', '#5778c4', '#cb963b'];
  function curves(predictions) {
    const valid = predictions.filter(p => (p.label === 0 || p.label === 1) && Number.isFinite(p.score) && p.score >= 0 && p.score <= 1);
    const positive = valid.filter(p => p.label === 1).length, negative = valid.length - positive;
    if (!positive || !negative) return null;
    const sorted = [...valid].sort((a, b) => b.score - a.score);
    const roc = [[0, 0]], pr = [[0, 1]];
    let tp = 0, fp = 0, ap = 0, lastRecall = 0;
    for (let i = 0; i < sorted.length;) {
      const score = sorted[i].score;
      do { sorted[i].label === 1 ? tp++ : fp++; i++; } while (i < sorted.length && sorted[i].score === score);
      const recall = tp / positive, precision = tp / (tp + fp);
      roc.push([fp / negative, recall]); pr.push([recall, precision]);
      ap += (recall - lastRecall) * precision; lastRecall = recall;
    }
    return {roc, pr, ap, prevalence: positive / valid.length};
  }
  function histogram(predictions) {
    const bins = Array.from({length: 10}, (_, i) => ({x: (i + .5) / 10, real: 0, fake: 0}));
    for (const p of predictions) {
      if ((p.label === 0 || p.label === 1) && Number.isFinite(p.score) && p.score >= 0 && p.score <= 1)
        bins[Math.min(9, Math.floor(p.score * 10))][p.label === 1 ? 'fake' : 'real']++;
    }
    return bins;
  }
  function svgNode(name, attrs = {}, text = '') {
    const element = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (text) element.textContent = text;
    return element;
  }
  function plot(container, series, {title = 'Chart', xLabel = '', yLabel = '', xMax = 1, yMax = 1, diagonal = false, empty = 'Available after a completed run', bars = false, categories = null} = {}) {
    container.replaceChildren();
    xMax = Math.max(Number.isFinite(xMax) ? xMax : 1, 1);
    yMax = Math.max(Number.isFinite(yMax) ? yMax : 1, .01);
    const svg = svgNode('svg', {viewBox: '0 0 500 260', role: 'img', 'aria-label': title});
    svg.append(svgNode('title', {}, title));
    const left = 52, right = 480, top = 20, bottom = 216;
    const x = value => left + value / xMax * (right - left), y = value => bottom - value / yMax * (bottom - top);
    const label = (value, max) => max <= 1 ? value.toFixed(1) : Number(value.toFixed(2)).toString();
    for (let i = 0; i <= 4; i++) {
      svg.append(svgNode('line', {x1: left, x2: right, y1: y(yMax * i / 4), y2: y(yMax * i / 4), stroke: '#e8edf0'}));
      svg.append(svgNode('text', {x: left - 10, y: y(yMax * i / 4) + 3, 'text-anchor': 'end', fill: '#84929b', 'font-size': 10}, label(yMax * i / 4, yMax)));
      if (!categories) svg.append(svgNode('text', {x: x(xMax * i / 4), y: bottom + 18, 'text-anchor': 'middle', fill: '#84929b', 'font-size': 10}, label(xMax * i / 4, xMax)));
    }
    if (categories) categories.forEach((name, i) => svg.append(svgNode('text', {x: x(i + .5), y: bottom + 18, 'text-anchor': 'middle', fill: '#84929b', 'font-size': 10}, name)));
    svg.append(svgNode('text', {x: (left + right) / 2, y: 255, 'text-anchor': 'middle', fill: '#6b7b86', 'font-size': 10}, xLabel));
    svg.append(svgNode('text', {transform: 'translate(12 120) rotate(-90)', 'text-anchor': 'middle', fill: '#6b7b86', 'font-size': 10}, yLabel));
    if (diagonal) svg.append(svgNode('line', {x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: '#b6c1c8', 'stroke-dasharray': '5 5'}));
    let count = 0;
    const legend = document.createElement('div'); legend.className = 'chart-legend';
    series.forEach((s, index) => {
      const points = s.points.filter(p => p.length === 2 && p.every(Number.isFinite)); count += points.length;
      const color = colors[index % colors.length];
      if (bars) {
        const width = (right - left) / Math.max(points.length, 1) / (series.length + 1);
        points.forEach(p => {
          const rect = svgNode('rect', {x: x(p[0]) + (index - series.length / 2) * width, y: y(p[1]), width: Math.max(1, width - 2), height: Math.max(0, bottom - y(p[1])), fill: color, rx: 2});
          rect.append(svgNode('title', {}, `${s.name}: ${p[1]} at ${p[0]}`)); svg.append(rect);
        });
      } else if (points.length) {
        svg.append(svgNode('polyline', {points: points.map(p => `${x(p[0])},${y(p[1])}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2.3, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}));
        if (points.length <= 40) points.forEach(p => {
          const dot = svgNode('circle', {cx: x(p[0]), cy: y(p[1]), r: 3, fill: color});
          dot.append(svgNode('title', {}, `${s.name}: ${p[1].toFixed(4)} at ${p[0]}`)); svg.append(dot);
        });
      }
      const item = document.createElement('span'), dot = document.createElement('i');
      dot.className = `legend-dot legend-${['teal', 'blue', 'gold'][index % 3]}`; item.append(dot, s.name); legend.append(item);
    });
    container.append(svg, legend);
    if (!count) { const message = document.createElement('div'); message.className = 'chart-empty'; const text = document.createElement('span'); text.textContent = empty; message.append(text); container.append(message); }
    return svg;
  }
  const api = {curves, histogram, plot};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FGCharts = api;
})(typeof window !== 'undefined' ? window : globalThis);
