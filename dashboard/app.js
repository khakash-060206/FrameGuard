'use strict';
/* FrameGuard dashboard. Reads saved results (data.js snapshot or /api/state) and never invents a number. */
const $ = id => document.getElementById(id);
const C = window.FGCharts;
let state = window.FRAMEGUARD_DATA || {histories: [], reports: [], dependencies: {}, checkpoints: []};
let stateText = '', shown = 50, mediaURL = null, busy = false;
const cache = new Map();

// ---- Formatting and small helpers ---------------------------------------------------------------
const finite = Number.isFinite;
const pct = (v, d = 1) => finite(v) ? `${(v * 100).toFixed(d)}%` : '–';
const dec = (v, d = 3) => finite(v) ? v.toFixed(d) : '–';
const num = v => finite(v) ? v.toLocaleString('en-IN') : '–';
const signed = (v, d = 3) => finite(v) ? `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}` : '–';
function el(tag, text, className) { const e = document.createElement(tag); if (text !== undefined && text !== null) e.textContent = text; if (className) e.className = className; return e; }
function memo(key, compute) { if (!cache.has(key)) cache.set(key, compute()); return cache.get(key); }
const logit = p => Math.log(p / (1 - p));
function band(threshold) {
  const margin = state.uncertain_margin ?? 1.5;
  if (!finite(threshold) || threshold <= 0 || threshold >= 1) return null;
  return [-1, 1].map(s => 1 / (1 + Math.exp(-(logit(threshold) + s * margin))));
}
const inBand = (score, b) => b && score > b[0] && score < b[1];
const modelName = m => m === 'temporal' ? 'CNN + LSTM' : 'CNN only';

function table(host, headers, rows, {emph = () => false, rowClass = () => '', numeric = []} = {}) {
  host = typeof host === 'string' ? $(host) : host;
  host.replaceChildren();
  if (!rows.length) { host.append(el('p', 'Nothing saved for this selection.', 'empty')); return; }
  const t = el('table'), head = el('tr');
  headers.forEach((h, i) => { const th = el('th', h, numeric.includes(i) ? 'num' : ''); th.scope = 'col'; head.append(th); });
  const thead = el('thead'); thead.append(head); t.append(thead);
  const body = el('tbody');
  rows.forEach((row, r) => {
    const tr = el('tr'); const cls = [emph(r) ? 'emph' : '', rowClass(r)].filter(Boolean).join(' '); if (cls) tr.className = cls;
    row.forEach((value, i) => { const td = el('td', undefined, numeric.includes(i) ? 'num' : ''); if (value instanceof Node) td.append(value); else td.textContent = value; tr.append(td); });
    body.append(tr);
  });
  t.append(body); host.append(t);
}
function tag(text, tone) { return el('span', text, `tag${tone ? ' ' + tone : ''}`); }
function barCell(value, text, key = 'k-lstm', min = 0) {
  const wrap = el('div', undefined, 'cell-bar'), track = el('div', undefined, 'track'), fill = el('div', undefined, `fill ${key}`);
  fill.style.width = `${Math.max(0, Math.min(1, (value - min) / (1 - min))) * 100}%`;
  track.append(fill); wrap.append(track, el('span', text)); return wrap;
}

// ---- Report access --------------------------------------------------------------------------------
function sortedReports() { return [...(state.reports || [])].sort((a, b) => (a.report.split === 'external') - (b.report.split === 'external')); }
function currentReport() { return state.reports.find(r => r.name === $('report-select').value)?.report; }
function domainOrder(names) {
  const rank = n => n === 'All datasets' ? 0 : n.includes('unseen') ? 3 : n.startsWith('FF++ ') ? 2 : 1;
  return [...names].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}
function inDomain(p, domain) { return p.domains ? p.domains.includes(domain) : p.dataset === domain; }
function predictionsFor(report, model, domain) { return (report?.[`${model}_predictions`] || []).filter(p => inDomain(p, domain)); }
function isUnseen(report, domain) { return domain.includes('unseen') || report?.split === 'external'; }

/** Everything derived from one (report, model, group) selection, computed once. */
function stats(reportName, report, model, domain) {
  return memo(`${reportName}|${model}|${domain}`, () => {
    const m = report?.metrics?.[model]?.[domain];
    const preds = predictionsFor(report, model, domain);
    if (!m || !preds.length) return null;
    const b = band(m.threshold), curve = C.curves(preds), calib = C.calibration(preds);
    const decided = preds.filter(p => !inBand(p.score, b)), unsure = preds.filter(p => inBand(p.score, b));
    const right = ps => ps.filter(p => (p.score >= m.threshold) === (p.label === 1)).length;
    return {m, preds, band: b, curve, calib, ci: C.bootstrapAuc(preds), mcc: C.rates(C.confusion(preds, m.threshold)).mcc,
      unsureShare: unsure.length / preds.length, decidedAccuracy: decided.length ? right(decided) / decided.length : null,
      unsureAccuracy: unsure.length ? right(unsure) / unsure.length : null};
  });
}
function paired(reportName, report, domain) {
  return memo(`pair|${reportName}|${domain}`, () => {
    const ms = report?.metrics?.spatial?.[domain], mt = report?.metrics?.temporal?.[domain];
    if (!ms || !mt) return null;
    const spatial = new Map(predictionsFor(report, 'spatial', domain).map(p => [`${p.dataset}|${p.video_id}`, p]));
    const cells = {both: 0, lstm: 0, cnn: 0, neither: 0};
    for (const t of predictionsFor(report, 'temporal', domain)) {
      const s = spatial.get(`${t.dataset}|${t.video_id}`); if (!s) continue;
      const lr = (t.score >= mt.threshold) === (t.label === 1), cr = (s.score >= ms.threshold) === (s.label === 1);
      cells[lr && cr ? 'both' : lr ? 'lstm' : cr ? 'cnn' : 'neither']++;
    }
    return {...cells, p: C.mcnemar(cells.lstm, cells.cnn), n: cells.both + cells.lstm + cells.cnn + cells.neither};
  });
}

// ---- Summary ----------------------------------------------------------------------------------------
function summary() {
  const reports = sortedReports(), main = reports[0];
  $('ruler').replaceChildren(); $('ruler-key').replaceChildren();
  ['fact-uncertain', 'fact-lstm', 'fact-run'].forEach(id => $(id).replaceChildren());
  runFact();
  if (!main) {
    $('headline').textContent = 'No evaluation has been saved yet.';
    $('lede').textContent = 'Train and evaluate a model, and this page will summarise how well it separates real from manipulated faces.';
    $('ruler').append(el('p', 'The ruler fills in after the first evaluation report is saved.', 'empty'));
    return;
  }
  const rows = reports.flatMap(r => domainOrder(Object.keys(r.report.metrics?.temporal || {})).map(domain => ({r, domain, unseen: isUnseen(r.report, domain)})));
  const known = rows.find(x => !x.unseen && x.domain === 'All datasets') || rows.find(x => !x.unseen);
  const unseen = rows.filter(x => x.unseen);
  const km = known?.r.report.metrics.temporal[known.domain];
  const um = unseen[0] ? unseen[0].r.report.metrics.temporal[unseen[0].domain] : null;
  $('headline').textContent = km && um
    ? `Gets ${pct(km.accuracy, 0)} of test videos right on deepfake methods it learned, but catches only ${pct(um.recall, 0)} of fakes from a method it never saw.`
    : km ? `Gets ${pct(km.accuracy, 0)} of held-out test videos right.` : 'Results are saved but incomplete.';
  const datasets = [...new Set((main.report.temporal_predictions || []).map(p => p.dataset))];
  $('lede').textContent = `CNN + LSTM, scored per video on ${num(km?.count)} held-out test videos from ${datasets.join(', ').replace(/, ([^,]*)$/, ' and $1')}. Thresholds were chosen on validation data only; no test video, source video or person was seen in training.`;

  // The ruler: every group's ROC-AUC on a 0.5 (chance) to 1.0 scale, both models, bootstrap interval for the LSTM.
  const scale = el('div', undefined, 'ruler-scale'), ticks = el('div', undefined, 'ruler-ticks');
  [.5, .6, .7, .8, .9, 1].forEach(v => { const t = el('span', v === .5 ? '0.5 chance' : v.toFixed(1)); t.style.left = `${(v - .5) / .5 * 100}%`; ticks.append(t); });
  scale.append(el('span', 'Test group'), ticks, el('span', 'ROC-AUC'));
  $('ruler').append(scale);
  const pos = v => `${Math.max(0, Math.min(1, (v - .5) / .5)) * 100}%`;
  for (const {r, domain, unseen: u} of rows) {
    const mt = r.report.metrics.temporal[domain], ms = r.report.metrics.spatial?.[domain];
    const s = stats(r.name, r.report, 'temporal', domain);
    const row = el('div', undefined, `ruler-row${u ? ' unseen' : ''}`);
    const name = el('div', undefined, 'ruler-name');
    name.append(document.createTextNode(domain.replace(' (unseen method)', '')), el('small', `${num(mt.count)} videos${u ? (domain.includes('unseen') ? ', method never trained on' : ', dataset never trained on') : ''}`));
    const track = el('div', undefined, 'ruler-track'); track.append(el('span', undefined, 'ruler-chance'));
    if (s?.ci) { const ci = el('span', undefined, 'ruler-ci'); ci.style.left = pos(s.ci[0]); ci.style.width = `calc(${pos(s.ci[1])} - ${pos(s.ci[0])})`; track.append(ci); }
    if (ms) { const d = el('span', undefined, 'ruler-dot k-cnn'); d.style.left = pos(ms.roc_auc); d.title = `CNN only: ${dec(ms.roc_auc)}`; track.append(d); }
    const d = el('span', undefined, 'ruler-dot k-lstm'); d.style.left = pos(mt.roc_auc); d.title = `CNN + LSTM: ${dec(mt.roc_auc)}${s?.ci ? ` (95% CI ${dec(s.ci[0])}–${dec(s.ci[1])})` : ''}`; track.append(d);
    const value = el('div', dec(mt.roc_auc), 'ruler-value'); value.append(el('small', `${pct(mt.recall, 0)} of fakes caught`));
    row.append(name, track, value); $('ruler').append(row);
  }
  for (const [key, text] of [['k-lstm', 'CNN + LSTM (value shown)'], ['k-cnn', 'CNN only']]) { const s = el('span'); s.append(el('i', undefined, key), text); $('ruler-key').append(s); }

  // Fact strip: the uncertain band, the LSTM's contribution, and the run.
  const s = known && stats(known.r.name, known.r.report, 'temporal', known.domain);
  const fu = $('fact-uncertain'); fu.append(el('h3', 'When it answers'));
  if (s?.band) {
    fu.append(el('div', pct(s.decidedAccuracy), 'big'),
      el('p', `correct on the ${pct(1 - s.unsureShare, 0)} of ${known.domain === 'All datasets' ? 'test' : known.domain} videos it gives a verdict for. The other ${pct(s.unsureShare, 0)} score too close to the threshold, where it was right only ${pct(s.unsureAccuracy, 0)} of the time, so the site answers "uncertain".`));
  } else fu.append(el('p', 'Appears once a threshold is saved.'));
  const pr = known && paired(known.r.name, known.r.report, known.domain);
  const fl = $('fact-lstm'); fl.append(el('h3', 'Does the LSTM help?'));
  const ss = known?.r.report.metrics.spatial?.[known.domain];
  if (pr && ss && km) {
    fl.append(el('div', `${signed(km.f1 - ss.f1)} F1`, 'big'),
      el('p', `F1 ${dec(ss.f1)} with the CNN alone, ${dec(km.f1)} with the LSTM. The LSTM fixes ${num(pr.lstm)} videos the CNN gets wrong and breaks ${num(pr.cnn)}; McNemar ${pr.p < .001 ? 'p < 0.001' : `p = ${dec(pr.p)}`}, ${pr.p < .05 ? 'a real difference at the 5% level' : 'not significant at the 5% level'}.`));
  } else fl.append(el('p', 'Appears once both models are evaluated.'));
}

function runFact() {
  const fr = $('fact-run'), e = state.experiment, inv = state.inventory;
  fr.append(el('h3', 'This run'));
  const dl = el('dl');
  const add = (k, v) => { if (v === undefined || v === null || v === '') return; dl.append(el('dt', k), el('dd', v)); };
  add('Data', inv ? `${num(inv.selected_videos)} videos, ${num(state.prepared_videos)} usable` : undefined);
  add('Hardware', e?.device ? `${e.device === 'cuda' ? 'GPU' : 'CPU'}${e.amp === 'on' ? ', mixed precision' : ''}` : undefined);
  add('Training', e?.epochs_per_stage ? `${e.epochs_per_stage} epochs per stage, batch ${e.batch_size}` : undefined);
  add('Frames', e?.max_samples ? `${e.max_samples} per video (~${Math.round(e.max_samples / 8)} s)` : undefined);
  add('Augmentation', e?.augment === undefined ? undefined : e.augment ? 'flip, rescale, blur, colour, JPEG' : 'none');
  add('Finished', e?.updated_at && e.phase === 'completed' ? new Date(e.updated_at).toLocaleString() : e?.phase);
  if (dl.children.length) fr.append(dl); else fr.append(el('p', 'Run details appear after the first training run.'));
}

// ---- Test results -----------------------------------------------------------------------------------
function results() {
  const name = $('report-select').value, report = currentReport(), domain = $('domain-select').value, model = $('model-select').value;
  const s = stats(name, report, model, domain), other = stats(name, report, model === 'temporal' ? 'spatial' : 'temporal', domain);
  $('eval-context').textContent = report ? `${report.split === 'external' ? 'Unseen dataset' : 'Held-out test split'}, scored per ${report.unit || 'video'}` : '';
  // Tiles
  $('eval-tiles').replaceChildren();
  const tiles = [['Accuracy', pct(s?.m.accuracy), s ? `${num(s.m.count)} videos` : 'No results yet'],
    ['ROC-AUC', dec(s?.m.roc_auc), s?.ci ? `95% CI ${dec(s.ci[0])} to ${dec(s.ci[1])}` : ''],
    ['F1 score', dec(s?.m.f1), s ? `precision ${pct(s.m.precision, 0)}, recall ${pct(s.m.recall, 0)}` : ''],
    ['False-positive rate', pct(s?.m.false_positive_rate), 'real videos flagged as fake']];
  for (const [label, value, note] of tiles) { const t = el('div', undefined, 'tile'); t.append(el('span', label), el('strong', value), el('small', note)); $('eval-tiles').append(t); }
  // Metric table (both models)
  const lstm = model === 'temporal' ? s : other, cnn = model === 'temporal' ? other : s;
  const row = (label, f) => [label, f(lstm), f(cnn)];
  table('metric-table', ['Metric', 'CNN + LSTM', 'CNN only'], !s ? [] : [
    row('Videos', x => num(x?.m.count)), row('Accuracy', x => pct(x?.m.accuracy)), row('Balanced accuracy', x => pct(x?.m.balanced_accuracy)),
    row('Precision', x => pct(x?.m.precision)), row('Recall (sensitivity)', x => pct(x?.m.recall)), row('Specificity', x => pct(x?.m.specificity)),
    row('F1 score', x => dec(x?.m.f1)), row('Matthews correlation', x => dec(x?.mcc)),
    row('ROC-AUC', x => x ? `${dec(x.m.roc_auc)}${x.ci ? ` (${dec(x.ci[0])}–${dec(x.ci[1])})` : ''}` : '–'),
    row('Average precision', x => dec(x?.curve?.ap)), row('False-positive rate', x => pct(x?.m.false_positive_rate)),
    row('Brier score (lower is better)', x => dec(x?.calib?.brier)), row('Log loss (lower is better)', x => dec(x?.calib?.logloss)),
    row('Calibration error, ECE', x => dec(x?.calib?.ece)), row('Decision threshold', x => dec(x?.m.threshold)),
    row('Answered "uncertain"', x => pct(x?.unsureShare)), row('Accuracy when it answers', x => pct(x?.decidedAccuracy))],
    {numeric: [1, 2]});
  // Confusion matrix
  $('threshold').textContent = s ? `Threshold ${dec(s.m.threshold)}` : '';
  $('confusion').replaceChildren();
  if (s?.m.confusion_matrix) {
    const cm = el('div', undefined, 'cm'), names = ['Real', 'Manipulated'], kinds = [['True negative', 'False positive'], ['False negative', 'True positive']];
    cm.append(el('span', '', 'cm-label'), el('span', 'Predicted real', 'cm-label'), el('span', 'Predicted manipulated', 'cm-label'));
    s.m.confusion_matrix.forEach((r, i) => { cm.append(el('span', `Actually ${names[i].toLowerCase()}`, 'cm-label')); r.forEach((count, j) => { const c = el('div', undefined, `cm-cell${i === j ? '' : ' error'}`); c.append(el('strong', num(count)), el('small', kinds[i][j])); cm.append(c); }); });
    $('confusion').append(cm);
  } else $('confusion').append(el('p', 'Appears after evaluation.', 'empty'));
  // Curves for both models
  const both = ['temporal', 'spatial'].map(k => ({k, s: stats(name, report, k, domain)}));
  C.lineChart($('roc-chart'), {title: 'ROC curve', diagonal: true, x: {min: 0, max: 1, label: 'False-positive rate'}, y: {min: 0, max: 1, label: 'True-positive rate'},
    series: both.map(({k, s: x}) => ({name: modelName(k), key: k === 'temporal' ? 'k-lstm' : 'k-cnn', points: x?.curve?.roc || []})), empty: 'Needs real and manipulated videos.'});
  C.lineChart($('pr-chart'), {title: 'Precision-recall curve', x: {min: 0, max: 1, label: 'Recall'}, y: {min: 0, max: 1, label: 'Precision'},
    series: both.map(({k, s: x}) => ({name: modelName(k), key: k === 'temporal' ? 'k-lstm' : 'k-cnn', points: x?.curve?.pr || []})), empty: 'Needs real and manipulated videos.'});
  $('pr-note').textContent = s?.curve ? `Average precision ${dec(s.curve.ap)} for ${modelName(model)}. ${pct(s.curve.prevalence, 0)} of videos in this group are manipulated, which is what a random guess would score.` : 'Manipulated videos are the positive class.';
  const rules = s ? [{x: s.m.threshold, label: `Threshold ${dec(s.m.threshold, 2)}`}] : [];
  const bands = s?.band ? [{x0: s.band[0], x1: s.band[1]}] : [];
  const extra = s?.band ? [{name: 'Uncertain band', key: 'k-band'}, {name: 'Threshold', key: 'k-thr'}] : [];
  const sweep = s ? C.sweep(s.preds) : [];
  C.lineChart($('sweep-chart'), {title: 'Threshold sweep', x: {min: 0, max: 1, label: 'Threshold'}, y: {min: 0, max: 1, label: 'Score'}, rules, bands, legendExtra: extra,
    series: [['Precision', 'precision', 'k-lstm'], ['Recall', 'recall', 'k-cnn'], ['F1', 'f1', 'k-p3']].map(([n, f, key]) => ({name: n, key, points: sweep.map(r => [r.t, r[f]])}))});
  const bins = C.histogram(s?.preds || [], 20);
  C.binChart($('dist-chart'), {title: 'Score distribution', x: {min: 0, max: 1, label: 'Manipulation score'}, yLabel: 'Videos', bins, rules, bands, legendExtra: extra,
    series: [{name: 'Real', key: 'k-real', values: bins.map(b => b.real)}, {name: 'Manipulated', key: 'k-fake', values: bins.map(b => b.fake)}]});
  C.lineChart($('calib-chart'), {title: 'Reliability diagram', diagonal: true, x: {min: 0, max: 1, label: 'Mean score in bin'}, y: {min: 0, max: 1, label: 'Share manipulated'},
    series: [{name: modelName(model), key: model === 'temporal' ? 'k-lstm' : 'k-cnn', markers: true, points: (s?.calib?.bins || []).map(b => [b.mean, b.frac])}]});
  $('calib-note').textContent = s?.calib ? `ECE ${dec(s.calib.ece)}, Brier ${dec(s.calib.brier)}. Scores are not calibrated probabilities: points off the diagonal show how far a score is from the real share of fakes.` : 'Appears after evaluation.';
  // Paired comparison
  const pr = paired(name, report, domain);
  $('pair').replaceChildren(); $('pair-verdict').textContent = '';
  if (pr) {
    const g = el('div', undefined, 'pair');
    const cell = (n, text, bad) => { const c = el('div', undefined, `cm-cell${bad ? ' error' : ''}`); c.append(el('strong', num(n)), el('small', text)); return c; };
    g.append(el('span', '', 'cm-label'), el('span', 'LSTM right', 'cm-label'), el('span', 'LSTM wrong', 'cm-label'),
      el('span', 'CNN right', 'cm-label'), cell(pr.both, 'both right'), cell(pr.cnn, 'only the CNN right', true),
      el('span', 'CNN wrong', 'cm-label'), cell(pr.lstm, 'only the LSTM right'), cell(pr.neither, 'both wrong', true));
    $('pair').append(g);
    $('pair-verdict').textContent = `On ${num(pr.n)} paired videos, McNemar's exact test gives ${pr.p < .001 ? 'p < 0.001' : `p = ${dec(pr.p)}`}: ${pr.p < .05 ? (pr.lstm > pr.cnn ? 'the LSTM is significantly better' : 'the CNN is significantly better') : 'the difference is not statistically significant'} at the 5% level.`;
  } else $('pair').append(el('p', 'Needs both models evaluated on this group.', 'empty'));
  // All groups for the selected model
  const groups = domainOrder(Object.keys(report?.metrics?.[model] || {}));
  table('group-table', ['Test group', 'Videos', 'ROC-AUC', 'Accuracy', 'Recall', 'False-positive rate', 'F1'], groups.map(d => {
    const m = report.metrics[model][d];
    return [d, num(m.count), barCell(m.roc_auc, dec(m.roc_auc), model === 'temporal' ? 'k-lstm' : 'k-cnn', .5), pct(m.accuracy), pct(m.recall), pct(m.false_positive_rate), dec(m.f1)];
  }), {numeric: [1, 3, 4, 5, 6], emph: i => groups[i] === domain, rowClass: i => isUnseen(report, groups[i]) ? 'unseen' : ''});
  // Most confident mistakes
  const confident = label => (s?.preds || []).filter(p => p.label === label && (p.score >= s.m.threshold) !== (p.label === 1)).sort((a, b) => Math.abs(b.score - s.m.threshold) - Math.abs(a.score - s.m.threshold)).slice(0, 6);
  const wrong = [...confident(1), ...confident(0)];  // six missed fakes and six false alarms
  table('errors-table', ['Video', 'Dataset', 'Actually', 'Score', 'Predicted'], wrong.map(p => [p.video_id, p.dataset, p.label ? 'Manipulated' : 'Real', dec(p.score), p.label ? 'Real (missed fake)' : 'Manipulated (false alarm)']), {numeric: [3]});
  predictionTable();
  const excluded = (report?.unassessed || []).filter(p => p.dataset === domain || domain === 'All datasets' || (domain.startsWith('FF++') && p.dataset === 'FaceForensics++'));
  const reasons = {};
  for (const p of excluded) { const r = (p.reason || 'Not scored').replace(/:.*/, ''); reasons[r] = (reasons[r] || 0) + 1; }
  $('exclusions').textContent = report ? `${num(s?.m.count)} videos scored. ${excluded.length ? `${num(excluded.length)} in these datasets could not be scored: ${Object.entries(reasons).sort((a, b) => b[1] - a[1]).map(([r, n]) => `${r.toLowerCase()} (${num(n)})`).join(', ')}.` : 'None were excluded.'} ${report.note || ''}` : '';
}

function predictionTable() {
  const name = $('report-select').value, report = currentReport(), model = $('model-select').value, domain = $('domain-select').value;
  const s = stats(name, report, model, domain);
  const query = $('prediction-search').value.toLowerCase(), filter = $('prediction-filter').value;
  const rows = (s?.preds || []).map(p => {
    const unsure = inBand(p.score, s.band), correct = (p.score >= s.m.threshold) === (p.label === 1);
    return {p, unsure, correct};
  }).filter(r => r.p.video_id.toLowerCase().includes(query) && (filter === 'all' || (filter === 'wrong' && !r.correct) || (filter === 'right' && r.correct) || (filter === 'uncertain' && r.unsure)));
  table('prediction-table', ['Video', 'Dataset', 'Actually', 'Score', 'Site would say', 'Result'], rows.slice(0, shown).map(({p, unsure, correct}) => [
    p.video_id, p.dataset, p.label ? 'Manipulated' : 'Real', dec(p.score),
    unsure ? 'Uncertain' : p.score >= s.m.threshold ? 'Manipulated' : 'Real',
    unsure ? tag('Uncertain', 'warn') : correct ? tag('Correct', 'good') : tag('Wrong', 'bad')]), {numeric: [3]});
  $('more-predictions').hidden = rows.length <= shown;
}

// ---- Training ---------------------------------------------------------------------------------------
function training() {
  const history = state.histories.find(h => h.name === $('history-select').value), epochs = history?.epochs || [];
  const stage = epochs[0]?.stage, best = epochs.reduce((b, e) => (e.validation?.roc_auc ?? -1) > (b?.validation?.roc_auc ?? -1) ? e : b, null);
  const size = {width: 400, height: 300};  // three charts share a row; a narrower canvas keeps the text legible
  const n = Math.max(1, epochs.length), x = {min: 1, max: Math.max(2, n), label: 'Epoch', ticks: epochs.map(e => e.epoch), format: v => String(v)};
  const pts = f => epochs.map(e => [e.epoch, f(e)]);
  const rules = best ? [{x: best.epoch, label: 'Selected'}] : [];
  const lossMax = Math.max(.1, ...epochs.flatMap(e => [e.training_loss, e.validation_loss]).filter(finite));
  C.lineChart($('loss-chart'), {...size, title: 'Loss', x, y: {min: 0, max: +(lossMax * 1.1).toFixed(2), label: 'Loss'}, rules, empty: 'Fills in as epochs finish.',
    series: [{name: 'Training', key: 'k-lstm', markers: true, points: pts(e => e.training_loss)}, {name: 'Validation', key: 'k-cnn', markers: true, points: pts(e => e.validation_loss)}]});
  const aucMin = Math.min(.9, ...epochs.map(e => e.validation?.roc_auc).filter(finite));
  C.lineChart($('auc-chart'), {...size, title: 'Validation ROC-AUC', x, y: {min: Math.max(0, Math.floor(aucMin * 10) / 10), max: 1, label: 'ROC-AUC'}, rules, empty: 'Fills in as epochs finish.',
    series: [{name: 'Validation ROC-AUC', key: 'k-lstm', markers: true, points: pts(e => e.validation?.roc_auc)}]});
  C.lineChart($('prf-chart'), {...size, title: 'Validation precision, recall and F1', x, y: {min: 0, max: 1, label: 'Score'}, rules, empty: 'Fills in as epochs finish.',
    series: [['Precision', 'precision', 'k-lstm'], ['Recall', 'recall', 'k-cnn'], ['F1', 'f1', 'k-p3']].map(([nm, f, key]) => ({name: nm, key, markers: true, points: pts(e => e.validation?.[f])}))});
  $('history-summary').textContent = !epochs.length ? 'No training history saved.' : stage === 'spatial'
    ? `${epochs.length} epochs. Epoch 1 trains only the new classifier head on a frozen ImageNet backbone; from epoch 2 the whole network is fine-tuned. Epoch ${best?.epoch} had the best validation ROC-AUC and was kept.`
    : `${epochs.length} epochs on cached CNN features; only the LSTM and its head train. Epoch ${best?.epoch} had the best validation ROC-AUC and was kept.`;
  table('epoch-table', ['Epoch', 'Samples', 'Train loss', 'Val. loss', 'Train acc.', 'Val. acc.', 'Val. AUC', 'Val. F1'], epochs.map(e => [
    e.epoch, num(e.training_samples), dec(e.training_loss), dec(e.validation_loss), pct(e.training_accuracy), pct(e.validation?.accuracy), dec(e.validation?.roc_auc), dec(e.validation?.f1)]),
    {numeric: [1, 2, 3, 4, 5, 6, 7], emph: i => epochs[i] === best});
  const e = state.experiment || {}, card = $('model-card'); card.replaceChildren();
  const dl = el('dl'); dl.className = 'fact'; const add = (k, v) => { if (v !== undefined && v !== null && v !== '') dl.append(el('dt', k), el('dd', String(v))); };
  const factHolder = el('div', undefined, 'fact'); factHolder.style.padding = '0';
  add('Backbone', 'EfficientNet-B0, ImageNet weights, 224 px face crops');
  add('Temporal model', 'LSTM, 128 hidden units, 16-frame windows at 8 fps, stride 4');
  add('Protocol', e.protocol === 'mixed' ? 'Pooled FF++, Celeb-DF v2 and DFDC' : e.protocol);
  add('Hardware', e.device ? `${e.device === 'cuda' ? 'GPU' : 'CPU'}${e.amp === 'on' ? ', float16 mixed precision' : ''}` : undefined);
  add('Epochs per stage', e.epochs_per_stage); add('Batch size', e.batch_size);
  add('Frames per video', e.max_samples); add('Training crops per video per epoch', e.frames_per_video ?? (e.epochs_per_stage ? 'all' : undefined));
  add('Augmentation', e.augment === undefined ? undefined : e.augment ? 'flip, rescale, blur, colour, JPEG' : 'none');
  add('Sampler', 'balanced across datasets and classes, then videos');
  add('Model selection', 'best validation ROC-AUC');
  factHolder.append(dl); card.append(factHolder);
}

// ---- Data -------------------------------------------------------------------------------------------
function data() {
  const inv = state.inventory;
  $('data-total').textContent = inv ? `${num(inv.selected_videos)} videos from ${Object.keys(inv.datasets || {}).length} datasets` : 'No dataset inventory yet';
  $('prep-summary').textContent = inv ? `${num(state.prepared_videos)} passed face preparation${finite(inv.rejected_videos) ? `; ${num(inv.rejected_videos)} were left out (no usable face, more than one face, or too few frames)` : ''}.` : 'Import datasets to see how they were split.';
  const host = $('split-groups'); host.replaceChildren();
  const entries = inv?.by_dataset ? Object.entries(inv.by_dataset).map(([k, v]) => [...k.split('/'), v]) : Object.entries(inv?.splits || {}).map(([split, v]) => ['All data', split, v]);
  const max = Math.max(1, ...entries.map(([, , v]) => v.real + v.manipulated));
  const datasets = [...new Set(entries.map(([d]) => d))];
  const splitName = {train: 'Training', val: 'Validation', test: 'Test', external: 'External test'};
  for (const dataset of datasets) {
    const group = el('div', undefined, 'split-group'); group.append(el('h3', dataset));
    const rows = el('div', undefined, 'split-rows');
    for (const [, split, v] of entries.filter(([d]) => d === dataset)) {
      const row = el('div', undefined, 'split-row'), stack = el('div', undefined, 'stack');
      const real = el('span', undefined, 'k-real'), fake = el('span', undefined, 'k-fake');
      real.style.width = `${v.real / max * 100}%`; fake.style.width = `${v.manipulated / max * 100}%`;
      real.title = `${num(v.real)} real`; fake.title = `${num(v.manipulated)} manipulated`;
      stack.append(real, fake);
      row.append(el('span', `${splitName[split] || split}: ${num(v.real)} real, ${num(v.manipulated)} fake`), stack, el('span', num(v.real + v.manipulated), 'total'));
      rows.append(row);
    }
    group.append(rows); host.append(group);
  }
  if (!entries.length) host.append(el('p', 'No splits recorded yet.', 'empty'));
  $('protocol').textContent = inv?.protocol || 'Import the datasets to record how they were split.';
  $('protocol-note').textContent = inv?.unseen_method ? `Thresholds are chosen on validation data only. Unseen-method test: ${inv.unseen_method}.` : inv ? 'Related original and manipulated videos stay in the same split. The external set is never used for training or thresholds.' : '';
  $('identity-note').textContent = inv?.identity_limit || '';
  $('data-verification').textContent = inv?.verification || '';
  $('archive-list').replaceChildren();
  for (const a of inv?.archives || []) { const row = el('div'); row.append(el('span', a.name), el('span', finite(a.bytes) ? `${(a.bytes / 1e9).toFixed(2)} GB` : a.type || '', 'muted')); $('archive-list').append(row); }
}

// ---- Try it -----------------------------------------------------------------------------------------
function uploadHint() {
  const frames = state.experiment?.max_samples;
  $('upload-hint').textContent = `JPG, PNG or MP4, up to 200 MB. ${frames ? `The first ~${Math.round(frames / 8)} s of a video is analysed.` : 'The opening seconds of a video are analysed.'}`;
}
function inferenceReady() {
  const online = location.protocol !== 'file:';
  $('analyse').disabled = busy || !online || !state.can_predict || !$('media-file').files.length;
  if (!busy && !$('inference-results').dataset.complete) $('inference-status').textContent = !online ? 'Start the dashboard server (python -m frameguard dashboard) to analyse files.' : !state.can_predict ? 'Analysis needs the ML dependencies, the face detector and a trained model.' : 'Your file stays on this computer and is deleted after analysis.';
}
function showVerdict(v, maxFrames) {
  const box = $('verdict'); box.replaceChildren(); box.hidden = !v; if (!v) return;
  box.className = `verdict ${v.label}`;
  box.append(el('strong', {manipulated: 'Likely manipulated', real: 'No manipulation detected', uncertain: 'Uncertain: too close to call', unable_to_assess: 'Unable to assess'}[v.label] || v.label));
  if (finite(v.score)) {
    box.append(el('p', `Score ${dec(v.score)}, threshold ${dec(v.threshold)}. ${v.mode === 'temporal' ? `Average of ${v.units} LSTM windows, scored the same way as the test set.` : `Average of ${v.units} frame score${v.units === 1 ? '' : 's'}; no full 16-frame window.`}`));
    const scale = el('div', undefined, 'score-scale'), track = el('span', undefined, 'track'), cut = el('span', undefined, 'cut'), pin = el('span', undefined, 'pin'), ends = el('div', undefined, 'ends');
    cut.style.left = `${v.threshold * 100}%`; pin.style.left = `${v.score * 100}%`;
    scale.append(track);
    if (v.uncertain_range) { const zone = el('span', undefined, 'zone'); zone.style.left = `${v.uncertain_range[0] * 100}%`; zone.style.width = `${(v.uncertain_range[1] - v.uncertain_range[0]) * 100}%`; scale.append(zone); }
    ends.append(el('span', '0 real'), el('span', '1 manipulated'));
    scale.append(cut, pin, ends); box.append(scale);
  }
  const notes = [];
  if (v.label === 'uncertain') notes.push('On held-out test videos, scores in the shaded range were right only about 60% of the time, so no call is made.');
  if (v.max_faces > 1) notes.push('More than one face appeared; the model was tested on single-face videos only.');
  if (v.mode === 'temporal' && maxFrames) notes.push(`Only the first ${maxFrames} sampled frames (~${Math.round(maxFrames / 8)} s) are scored, as in testing.`);
  if (v.label !== 'unable_to_assess') notes.push('Reliable on deepfakes like those in the training datasets; fakes from other tools are often missed, so "no manipulation detected" is not proof a file is real.');
  if (notes.length) box.append(el('small', notes.join(' ')));
}
$('media-file').addEventListener('change', () => {
  if (mediaURL) URL.revokeObjectURL(mediaURL);
  $('media-preview').replaceChildren();
  const file = $('media-file').files[0]; $('file-name').textContent = file?.name || 'No file chosen';
  if (file) {
    mediaURL = URL.createObjectURL(file);
    const preview = document.createElement(file.name.toLowerCase().endsWith('.mp4') ? 'video' : 'img');
    preview.src = mediaURL; if (preview.tagName === 'VIDEO') preview.controls = true; else preview.alt = 'Chosen image';
    $('media-preview').append(preview);
  } else $('media-preview').append(el('span', 'Your file appears here.'));
  delete $('inference-results').dataset.complete; $('verdict').hidden = true; inferenceReady();
});
$('analyse').addEventListener('click', async () => {
  const file = $('media-file').files[0]; if (!file || !state.can_predict || busy) return;
  if (file.size > 200 * 1024 * 1024) { $('inference-status').textContent = 'This file is over 200 MB. Use the command-line predictor for larger videos.'; return; }
  busy = true; inferenceReady(); $('analyse').textContent = 'Analysing'; $('inference-status').textContent = 'Finding and scoring faces. Videos can take a minute.';
  try {
    const query = new URLSearchParams({name: file.name, checkpoint: $('checkpoint-select').value});
    const response = await fetch(`/api/predict?${query}`, {method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: file});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Analysis failed');
    showVerdict(result.verdict, result.max_frames);
    const v = result.verdict, margin = v?.uncertain_range ? logit(v.uncertain_range[1]) - logit(v.threshold) : 0;
    const assess = (score, threshold) => Math.abs(logit(Math.min(1 - 1e-7, Math.max(1e-7, score))) - logit(threshold)) < margin ? 'Uncertain' : score >= threshold ? 'Manipulated' : 'Real';
    const tracks = new Map();
    for (const record of result.records) for (const face of record.faces) {
      const t = tracks.get(face.track_id) || {frames: 0, windows: [], spatial: [], threshold: face.threshold};
      t.frames++;
      if (face.mode === 'temporal') { t.windows.push([record.timestamp, face.score]); t.threshold = face.threshold; } else t.spatial.push(face.score);
      tracks.set(face.track_id, t);
    }
    const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
    table('inference-results', ['Face', 'Frames', 'LSTM windows', 'Mean score', 'Threshold', 'Assessment'], [...tracks].map(([id, t]) => {
      const score = t.windows.length ? mean(t.windows.map(p => p[1])) : mean(t.spatial);
      return [`Face ${id + 1}`, num(t.frames), t.windows.length ? num(t.windows.length) : '–', dec(score), dec(t.threshold), assess(score, t.threshold)];
    }), {numeric: [1, 2, 3, 4]});
    $('inference-results').dataset.complete = 'true';
    $('inference-status').textContent = tracks.size ? `Done. ${tracks.size} face${tracks.size > 1 ? 's' : ''} scored.` : 'No usable face was found, so the file could not be assessed.';
    const timelines = [...tracks].filter(([, t]) => t.windows.length).slice(0, 2);
    $('inference-chart').hidden = !timelines.length;
    if (timelines.length) {
      const xMax = Math.max(1, ...timelines.flatMap(([, t]) => t.windows.map(p => p[0]))), threshold = timelines[0][1].threshold;
      C.lineChart($('inference-chart'), {title: 'Score per LSTM window', x: {min: 0, max: xMax, label: 'Seconds'}, y: {min: 0, max: 1, label: 'Manipulation score'},
        series: [...timelines.map(([id, t], i) => ({name: `Face ${id + 1}`, key: i ? 'k-cnn' : 'k-lstm', markers: t.windows.length < 30, points: t.windows})),
          {name: 'Decision threshold', key: 'k-thr', points: [[0, threshold], [xMax, threshold]]}]});
    }
  } catch (error) { $('inference-status').textContent = error.message; }
  finally { busy = false; $('analyse').textContent = 'Analyse'; inferenceReady(); }
});

// ---- Shell: selects, notice, routing, refresh, theme, export ---------------------------------------
function selectOptions(select, options, empty) {
  const previous = select.value; select.replaceChildren();
  if (!options.length) { select.append(new Option(empty, '')); select.disabled = true; return; }
  select.disabled = false; options.forEach(([value, label]) => select.append(new Option(label, value)));
  if (options.some(([value]) => value === previous)) select.value = previous;
}
function domainOptions() { selectOptions($('domain-select'), domainOrder(Object.keys(currentReport()?.metrics?.temporal || {})).map(d => [d, d]), 'No evaluated group'); }
function notice() {
  const n = $('notice'), e = state.experiment, missing = Object.entries(state.dependencies || {}).filter(([, ok]) => !ok).map(([k]) => k);
  n.removeAttribute('data-tone');
  if (state.warnings?.length) { n.dataset.tone = 'bad'; n.textContent = `Some result files could not be read: ${state.warnings.join('; ')}`; }
  else if (e && e.phase !== 'completed') {
    const phases = {preparing_faces: 'Preparing face crops', training_spatial: 'Training the CNN', caching_features: 'Caching CNN features', training_temporal: 'Training the LSTM', evaluating_internal: 'Evaluating the test set', evaluating_external: 'Evaluating the external set', failed: 'The run stopped with an error'};
    if (e.phase === 'failed') n.dataset.tone = 'bad';
    n.textContent = `${phases[e.phase] || e.phase}. Last update ${new Date(e.updated_at).toLocaleString()}. ${e.error || ''}`;
  } else if (missing.length && state.reports.length) n.textContent = `Saved results are shown. Analysing new files needs: ${missing.join(', ')}.`;
  else if (!state.reports.length) n.textContent = 'No results saved yet. Charts fill in from completed training and evaluation; nothing here is example data.';
  else n.textContent = 'Every number on this page comes from the saved evaluation report. Thresholds were chosen on validation data, never on the test set.';
}
const pages = {summary: 'Summary', results: 'Test results', training: 'Training', data: 'Data', try: 'Try it'};
function route() {
  const page = Object.hasOwn(pages, location.hash.slice(1)) ? location.hash.slice(1) : 'summary';
  document.querySelectorAll('.page').forEach(p => { p.hidden = p.id !== page; });
  document.querySelectorAll('.tabs a').forEach(a => { if (a.dataset.page === page) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  document.title = `${pages[page]}: FrameGuard results`;
}
function render() {
  selectOptions($('report-select'), sortedReports().map(r => [r.name, r.report.split === 'external' ? `${r.name} (unseen dataset)` : r.name]), 'No evaluations yet');
  domainOptions();
  selectOptions($('history-select'), state.histories.map(h => [h.name, h.epochs[0]?.stage === 'temporal' ? 'LSTM (stage B)' : h.epochs[0]?.stage === 'spatial' ? 'CNN (stage A)' : h.name]), 'No training history yet');
  selectOptions($('checkpoint-select'), [...(state.checkpoints || [])].sort((a, b) => (b === 'temporal.pt') - (a === 'temporal.pt')).map(c => [c, c === 'temporal.pt' ? 'CNN + LSTM (temporal.pt, recommended)' : c === 'spatial.pt' ? 'CNN only (spatial.pt)' : c]), 'No trained model');
  notice(); summary(); results(); training(); data(); uploadHint(); inferenceReady();
}
async function refresh() {
  if (location.protocol === 'file:') { $('connection').textContent = 'Offline snapshot'; $('updated').textContent = 'To update this snapshot, run python -m frameguard dashboard --export'; return; }
  try {
    const response = await fetch('/api/state', {cache: 'no-store'}); if (!response.ok) throw new Error('unavailable');
    const text = await response.text();
    $('connection').textContent = 'Live'; $('connection').dataset.state = 'live';
    $('updated').textContent = `Checked ${new Date().toLocaleTimeString()}, refreshes every 15 s`;
    if (text !== stateText) { stateText = text; state = JSON.parse(text); cache.clear(); render(); }
  } catch { $('connection').textContent = 'Server stopped'; $('connection').dataset.state = 'down'; $('updated').textContent = 'Showing the last loaded results.'; }
}
function download(name, content, type) { const url = URL.createObjectURL(new Blob([content], {type})); const a = el('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $('theme').textContent = theme === 'dark' ? 'Light' : 'Dark';
  try { localStorage.setItem('frameguard-theme', theme); } catch { /* storage unavailable: theme lasts for this visit */ }
}
let storedTheme = null;
try { storedTheme = localStorage.getItem('frameguard-theme'); } catch { storedTheme = null; }
setTheme(storedTheme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
$('theme').addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
$('report-select').addEventListener('change', () => { domainOptions(); shown = 50; results(); });
['domain-select', 'model-select'].forEach(id => $(id).addEventListener('change', () => { shown = 50; results(); }));
$('history-select').addEventListener('change', training);
$('prediction-search').addEventListener('input', () => { shown = 50; predictionTable(); });
$('prediction-filter').addEventListener('change', () => { shown = 50; predictionTable(); });
$('more-predictions').addEventListener('click', () => { shown += 50; predictionTable(); });
$('refresh').addEventListener('click', refresh);
$('export').addEventListener('click', () => download('frameguard-results.json', JSON.stringify(state, null, 2), 'application/json'));
document.querySelectorAll('[data-save-chart]').forEach(button => button.addEventListener('click', () => {
  const svg = document.getElementById(button.dataset.saveChart)?.querySelector('svg');
  if (svg) download(`frameguard-${button.dataset.saveChart}.svg`, C.snapshot(svg), 'image/svg+xml');
}));
window.addEventListener('hashchange', route);
stateText = JSON.stringify(state);
route(); render(); refresh();
if (location.protocol !== 'file:') setInterval(refresh, 15000);
