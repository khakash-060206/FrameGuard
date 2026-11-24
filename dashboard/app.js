'use strict';
const $ = id => document.getElementById(id);
let state = window.FRAMEGUARD_DATA || {histories: [], reports: [], dependencies: {}, checkpoints: []};
let shown = 50, mediaURL = null, busy = false;
const titles = {
  overview: ['The experiment, at a glance.', 'From training signals to held-out performance. Every number backed by a saved result.'],
  training: ['Follow the learning process.', 'Epoch-by-epoch signals from your spatial and temporal training runs.'],
  evaluation: ['Measure what generalises.', 'Compare spatial and temporal models on the same held-out videos.'],
  datasets: ['Know the data behind the model.', 'Traceable inputs, explicit partitions, and held-out test data.'],
  inspect: ['Take a closer look.', 'Local facial-manipulation assessment using your own trained checkpoint.']
};
const number = value => Number.isFinite(value) ? value.toLocaleString('en-IN') : '—';
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const decimal = value => Number.isFinite(value) ? value.toFixed(3) : '—';
function element(tag, text, className) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (className) e.className = className; return e; }
function selectOptions(select, options, empty) {
  const previous = select.value; select.replaceChildren();
  if (!options.length) { select.append(new Option(empty, '')); select.disabled = true; return; }
  select.disabled = false; options.forEach(([value, label]) => select.append(new Option(label, value)));
  if (options.some(([value]) => value === previous)) select.value = previous;
}
// Internal test first, so the overview opens on the in-domain result rather than alphabetical order.
function sortedReports() { return [...state.reports].sort((a, b) => (a.report.split === 'external') - (b.report.split === 'external')); }
function currentReport() { return state.reports.find(r => r.name === $('report-select').value)?.report; }
function currentMetrics() { return currentReport()?.metrics?.[$('model-select').value]?.[$('domain-select').value]; }
function inDomain(p, domain) { return p.domains ? p.domains.includes(domain) : p.dataset === domain; }
function predictions(model = $('model-select').value) { return (currentReport()?.[`${model}_predictions`] || []).filter(p => inDomain(p, $('domain-select').value)); }
function accuracy(m) { if (Number.isFinite(m?.accuracy)) return m.accuracy; const c = m?.confusion_matrix; return c && m.count ? (c[0][0] + c[1][1]) / m.count : null; }
function kpis(id, m) {
  $(id).replaceChildren();
  [['Accuracy', percent(accuracy(m)), 'Correctly classified videos', '✓'], ['ROC-AUC', decimal(m?.roc_auc), 'Ranking across all thresholds', '↗'], ['F1 score', decimal(m?.f1), 'Precision–recall balance', '◎'], ['False-positive rate', percent(m?.false_positive_rate), 'Real videos flagged as manipulated', '↘']].forEach(([label, value, note, icon]) => {
    const card = element('article', undefined, 'kpi'), top = element('div', undefined, 'kpi-top');
    top.append(element('span', label), element('span', icon, 'kpi-symbol'));
    card.append(top, element('div', value, 'kpi-value'), element('p', m ? note : 'Awaiting evaluation', 'kpi-note')); $(id).append(card);
  });
}
function table(id, headers, rows, highlight = -1) {
  const host = $(id); host.replaceChildren();
  if (!rows.length) { host.append(element('p', 'No saved results for this selection.', 'empty-copy')); return; }
  const t = document.createElement('table'), head = document.createElement('thead'), tr = document.createElement('tr');
  headers.forEach(h => {const th = element('th', h); th.scope = 'col'; tr.append(th);}); head.append(tr); t.append(head);
  const body = document.createElement('tbody'); rows.forEach((row, i) => {const tr = document.createElement('tr'); if (i === highlight) tr.className = 'selected'; row.forEach(value => tr.append(element('td', String(value)))); body.append(tr);});
  t.append(body); host.append(t);
}
function comparison(id) {
  const report = currentReport(), domain = $('domain-select').value;
  if (!report) { $(id).replaceChildren(element('div', 'The first completed evaluation will show the CNN baseline alongside CNN + LSTM.', 'empty-comparison')); return; }
  const rows = ['spatial', 'temporal'].map(model => {const m = report.metrics?.[model]?.[domain]; return [model === 'spatial' ? 'CNN baseline' : 'CNN + LSTM', percent(accuracy(m)), decimal(m?.roc_auc), percent(m?.precision), percent(m?.recall), decimal(m?.f1), number(m?.count)];});
  table(id, ['Model', 'Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1', 'Videos'], rows, $('model-select').value === 'temporal' ? 1 : 0);
}
function overview() {
  const m = currentMetrics(), inventory = state.inventory;
  kpis('kpis', m); comparison('overview-comparison');
  $('overview-context').textContent = m ? `${$('domain-select').value} · ${currentReport().split} · ${$('model-select').selectedOptions[0].text}` : 'No evaluation saved yet';
  const stages = new Set(state.histories.flatMap(h => h.epochs.map(e => e.stage).filter(Boolean)));
  const trained = stages.has('spatial') && stages.has('temporal');
  const steps = [
    ['Dataset import', inventory ? `${number(inventory.selected_videos)} selected videos · source groups checked` : 'Import and validate your dataset archives', !!inventory],
    ['Runtime & face preparation', state.prepared_videos ? `${state.prepared_videos} videos with prepared face crops` : 'Dependencies, face detection and crop validation', !!state.prepared_videos && Object.values(state.dependencies).every(Boolean)],
    ['Train spatial + temporal models', trained ? `${state.histories.length} saved training histories` : 'ImageNet CNN → face features → LSTM', trained],
    ['Held-out evaluation', state.reports.length ? `${state.reports.length} saved evaluation reports` : 'Accuracy, ROC and error analysis', !!state.reports.length]
  ];
  $('progress-count').textContent = `${steps.filter(s => s[2]).length} / 4 recorded`;
  $('pipeline').replaceChildren(); steps.forEach(([title, description, done], i) => {
    const row = element('div', undefined, 'pipeline-step'), text = element('div', undefined, 'step-text'); text.append(element('strong', title), element('p', description));
    row.append(element('span', done ? '✓' : String(i + 1), `step-number${done ? ' done' : ''}`), text, element('span', done ? 'Recorded' : 'Pending', `badge ${done ? 'teal' : 'neutral'}`)); $('pipeline').append(row);
  });
  generalisation();
  $('data-total').textContent = number(inventory?.selected_videos); $('overview-data').replaceChildren();
  for (const [name, count] of Object.entries(inventory?.datasets || {})) {const line = element('div', undefined, 'dataset-line'); line.append(element('span', name), element('b', number(count))); $('overview-data').append(line);}
}
function generalisation() {
  const host = $('generalisation'); host.replaceChildren();
  const cards = sortedReports().flatMap(r => Object.entries(r.report.metrics?.temporal || {}).map(([domain, m]) => ({domain, external: r.report.split === 'external' || domain.includes('unseen'), unseenMethod: domain.includes('unseen'), m})));
  if (!cards.length) { host.append(element('p', 'Appears after the internal and external evaluations.', 'empty-copy')); return; }
  for (const {domain, external, unseenMethod, m} of cards) {
    const card = element('div', undefined, `gap-card${external ? ' external' : ''}`);
    card.append(element('span', unseenMethod ? 'UNSEEN METHOD' : external ? 'UNSEEN DATASET' : 'IN-DOMAIN TEST', 'eyebrow'), element('h3', domain));
    [['ROC-AUC', m.roc_auc, decimal], ['Recall (fakes caught)', m.recall, percent], ['False-positive rate', m.false_positive_rate, percent]].forEach(([label, value, format]) => {
      const row = element('div', undefined, 'gap-row'), bar = element('div', undefined, 'gap-bar'), fill = element('i');
      fill.style.width = `${Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0)) * 100}%`; bar.append(fill);
      row.append(element('span', label), bar, element('b', format(value))); card.append(row);
    });
    card.append(element('small', `${number(m.count)} assessed videos`, 'muted')); host.append(card);
  }
}
function training() {
  const epochs = state.histories.find(h => h.name === $('history-select').value)?.epochs || [];
  const points = getter => epochs.map(e => [e.epoch, getter(e)]).filter(p => p.every(Number.isFinite));
  const xmax = Math.max(1, ...epochs.map(e => e.epoch));
  const plot = (id, series, title, ymax = 1) => FGCharts.plot($(id), series, {title, xLabel: 'Epoch', yLabel: title, xMax: xmax, yMax: ymax, empty: 'This chart fills in as training epochs finish.'});
  const loss = [{name: 'Training', points: points(e => e.training_loss)}, {name: 'Validation', points: points(e => e.validation_loss)}];
  plot('loss-chart', loss, 'Loss', Math.max(.1, ...loss.flatMap(s => s.points.map(p => p[1]))) * 1.1);
  plot('accuracy-chart', [{name: 'Train · sampled units', points: points(e => e.training_accuracy)}, {name: 'Validation · videos', points: points(e => accuracy(e.validation))}], 'Accuracy');
  plot('auc-chart', [{name: 'Validation ROC-AUC', points: points(e => e.validation?.roc_auc)}], 'ROC-AUC');
  plot('validation-chart', ['precision', 'recall', 'f1'].map(key => ({name: key === 'f1' ? 'F1' : key[0].toUpperCase() + key.slice(1), points: points(e => e.validation?.[key])})), 'Score');
  $('history-summary').textContent = epochs.length ? `${epochs.length} recorded epochs · model selection uses validation ROC-AUC` : 'No training history available';
  table('epoch-table', ['Epoch', 'Train loss', 'Val. loss', 'Train acc.', 'Val. acc.', 'Val. AUC', 'Val. F1', 'Threshold'], epochs.map(e => [e.epoch, decimal(e.training_loss), decimal(e.validation_loss), percent(e.training_accuracy), percent(accuracy(e.validation)), decimal(e.validation?.roc_auc), decimal(e.validation?.f1), decimal(e.validation?.threshold)]));
}
function evaluation() {
  const report = currentReport(), domain = $('domain-select').value, model = $('model-select').value, m = currentMetrics();
  kpis('eval-kpis', m); comparison('comparison');
  const series = ['temporal', 'spatial'].map(key => ({name: key === 'temporal' ? 'CNN + LSTM' : 'CNN baseline', curve: FGCharts.curves(predictions(key))}));
  FGCharts.plot($('roc-chart'), series.map(s => ({name: s.name, points: s.curve?.roc || []})), {title: 'ROC curve', xLabel: 'False-positive rate', yLabel: 'True-positive rate', diagonal: true, empty: 'ROC requires evaluated real and manipulated videos.'});
  FGCharts.plot($('pr-chart'), series.map(s => ({name: s.name, points: s.curve?.pr || []})), {title: 'Precision–recall curve', xLabel: 'Recall', yLabel: 'Precision', empty: 'Precision–recall requires evaluated videos from both classes.'});
  const curve = FGCharts.curves(predictions());
  $('pr-note').textContent = curve ? `Selected model: average precision ${decimal(curve.ap)} · manipulated prevalence ${percent(curve.prevalence)}.` : 'Threshold-independent view of detection trade-offs.';
  $('eval-context').textContent = report ? `${report.split} · ${report.unit || 'video'}-level` : 'Video-level';
  $('threshold').textContent = `Threshold ${decimal(m?.threshold)}`;
  $('confusion').replaceChildren();
  if (m?.confusion_matrix) {
    const matrix = element('div', undefined, 'cm'); matrix.append(element('span', 'Actual ↓ / Pred. →', 'cm-label'), element('span', 'Real', 'cm-label'), element('span', 'Manipulated', 'cm-label'));
    m.confusion_matrix.forEach((row, i) => {matrix.append(element('span', i ? 'Manipulated' : 'Real', 'cm-label')); row.forEach((count, j) => {const cell = element('div', undefined, `cm-cell${i === j ? '' : ' error'}`); cell.append(element('strong', number(count)), element('small', [['True negative', 'False positive'], ['False negative', 'True positive']][i][j])); matrix.append(cell);});}); $('confusion').append(matrix);
  } else $('confusion').append(element('p', 'The confusion matrix appears after evaluation.', 'empty-copy'));
  const ps = predictions(), bins = FGCharts.histogram(ps);
  FGCharts.plot($('distribution'), [{name: 'Real', points: ps.length ? bins.map(b => [b.x, b.real]) : []}, {name: 'Manipulated', points: ps.length ? bins.map(b => [b.x, b.fake]) : []}], {title: 'Score distribution', bars: true, xLabel: 'Manipulation score', yLabel: 'Video count', yMax: Math.max(1, ...bins.flatMap(b => [b.real, b.fake])), empty: 'Video score distributions appear after evaluation.'});
  const excluded = (report?.unassessed || []).filter(p => p.dataset === domain || domain === 'All datasets');
  $('exclusions').textContent = report ? `${number(m?.count)} assessed videos · ${excluded.length} excluded or unassessable. ${report.note || ''}` : 'Excluded and unassessable videos will be counted in each evaluation.';
  predictionTable();
}
function predictionTable() {
  const filter = $('prediction-search').value.toLowerCase(), m = currentMetrics();
  const rows = predictions().filter(p => p.video_id.toLowerCase().includes(filter));
  table('prediction-table', ['Video ID', 'Actual', 'Score', 'Predicted', 'Result'], rows.slice(0, shown).map(p => {const predicted = Number.isFinite(m?.threshold) ? Number(p.score >= m.threshold) : null; return [p.video_id, p.label ? 'Manipulated' : 'Real', decimal(p.score), predicted === null ? '—' : predicted ? 'Manipulated' : 'Real', predicted === null ? '—' : predicted === p.label ? 'Correct' : 'Incorrect'];}));
  $('more-predictions').hidden = rows.length <= shown;
}
function datasets() {
  const data = state.inventory, entries = Object.entries(data?.splits || {});
  FGCharts.plot($('split-chart'), [{name: 'Real', points: entries.map(([, v], i) => [i + .5, v.real])}, {name: 'Manipulated', points: entries.map(([, v], i) => [i + .5, v.manipulated])}], {title: 'Dataset split sizes', bars: true, xMax: Math.max(1, entries.length), yMax: Math.max(1, ...entries.flatMap(([, v]) => [v.real, v.manipulated])), categories: entries.map(([k]) => k), yLabel: 'Videos', empty: 'No imported dataset inventory yet.'});
  const purpose = {train: 'Fit model weights', val: 'Select checkpoint & thresholds', test: 'Held-out evaluation', external: 'Cross-dataset evaluation'};
  const rows = data?.by_dataset ? Object.entries(data.by_dataset).map(([key, v]) => { const [dataset, split] = key.split('/'); return [`${dataset} · ${split}`, v, split]; }) : entries.map(([split, v]) => [split, v, split]);
  table('split-table', ['Partition', 'Real', 'Manipulated', 'Total', 'Purpose'], rows.map(([name, v, split]) => [name, number(v.real), number(v.manipulated), number(v.real + v.manipulated), purpose[split]]));
  $('protocol').textContent = data?.protocol || 'Import archives to register a reproducible split protocol.';
  $('protocol-note').textContent = data?.unseen_method ? `Thresholds are selected on validation data only. Unseen-method test: ${data.unseen_method}.` : 'Related original and manipulated source videos stay in the same split. Celeb-DF is held out for external evaluation; it never selects training thresholds.';
  $('identity-note').textContent = data?.identity_limit || 'Source separation does not establish identity separation when identity metadata is absent.';
  $('data-verification').textContent = data?.verification || 'No extraction verification recorded.';
  $('archive-list').replaceChildren(); for (const archive of data?.archives || []) {const row = element('div', undefined, 'archive-item'); row.append(element('span', archive.name), element('span', Number.isFinite(archive.bytes) ? `${(archive.bytes / 1e9).toFixed(2)} GB` : archive.type || '')); $('archive-list').append(row);}
}
function render() {
  selectOptions($('report-select'), sortedReports().map(r => [r.name, r.name]), 'No evaluations yet');
  selectOptions($('history-select'), state.histories.map(h => [h.name, h.name]), 'No training runs yet');
  selectOptions($('domain-select'), Object.keys(currentReport()?.metrics?.temporal || {}).map(d => [d, d]), 'No evaluated dataset');
  // Full CNN + LSTM model first; the spatial-only checkpoint stays available as the baseline.
  selectOptions($('checkpoint-select'), [...(state.checkpoints || [])].sort((a, b) => (b === 'temporal.pt') - (a === 'temporal.pt')).map(c => [c, c === 'temporal.pt' ? 'temporal.pt · CNN + LSTM (recommended)' : c === 'spatial.pt' ? 'spatial.pt · CNN baseline' : c]), 'No trained checkpoint');
  const missing = Object.entries(state.dependencies || {}).filter(([, ok]) => !ok).map(([name]) => name);
  $('notice').classList.toggle('ready', !!state.reports.length);
  $('notice').textContent = state.warnings?.length ? `Some result files could not be read: ${state.warnings.join(' · ')}` : missing.length ? `Local training needs the ML runtime. Missing: ${missing.join(', ')}. ${state.reports.length ? 'Saved reports are still available below.' : 'No evaluation metrics are available yet.'}` : !state.reports.length ? 'Ready to record experiments. Graphs populate from completed training and evaluation; no example scores are shown.' : 'Showing saved evaluation results. Thresholds are selected on validation data; external data is held out.';
  if (state.experiment) {
    const phases = {preparing_faces: 'Preparing face crops; training is queued next', training_spatial: 'Training the spatial CNN', caching_features: 'Caching CNN features for the LSTM', training_temporal: 'Training the temporal LSTM', evaluating_internal: 'Evaluating the internal test set', evaluating_external: 'Evaluating Celeb-DF', completed: 'Experiment completed', failed: 'Experiment stopped with an error'};
    const e = state.experiment, when = new Date(e.updated_at);
    const hardware = e.device ? ` on ${e.device === 'cuda' ? 'GPU' : 'CPU'}${e.amp === 'on' ? ' with mixed precision' : ''}` : '';
    $('notice').textContent = `${phases[e.phase] || e.phase}${hardware}. Last recorded ${isNaN(when) ? e.updated_at : when.toLocaleString()}. ${e.error || (e.phase === 'completed' ? 'Thresholds were selected on validation data; Celeb-DF was held out.' : '')}`;
  }
  overview(); training(); evaluation(); datasets(); uploadHint(); inferenceReady();
}
function route() {
  const page = Object.hasOwn(titles, location.hash.slice(1)) ? location.hash.slice(1) : 'overview';
  document.querySelectorAll('.page').forEach(p => p.hidden = p.id !== page);
  document.querySelectorAll('nav a').forEach(a => {const active = a.dataset.page === page; a.classList.toggle('active', active); if (active) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');});
  $('page-title').textContent = titles[page][0]; $('page-subtitle').textContent = titles[page][1];
  $('page-crumb').textContent = document.querySelector(`nav a[data-page="${page}"]`).textContent.trim().slice(1).trim();
}
async function refresh() {
  if (location.protocol === 'file:') { $('connection').textContent = 'Offline snapshot'; $('updated').textContent = 'To update this snapshot, run: python -m frameguard dashboard --export'; return; }
  try {
    const response = await fetch('/api/state', {cache: 'no-store'}); if (!response.ok) throw new Error('Server unavailable');
    state = await response.json(); render(); $('connection').textContent = 'Connected locally'; $('updated').textContent = `Refreshed ${new Date().toLocaleTimeString()} · auto-refresh 15s`;
  } catch { $('connection').textContent = 'Disconnected'; $('updated').textContent = 'Server unavailable. Showing the last loaded snapshot.'; }
}
function download(name, content, type) { const url = URL.createObjectURL(new Blob([content], {type})); const a = element('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
function uploadHint() {
  const frames = state.experiment?.max_samples;
  $('upload-hint').textContent = `JPG, PNG or MP4 · up to 200 MB · ${frames ? `first ~${Math.round(frames / 8)} s of video analysed` : 'opening seconds of video analysed'}`;
}
function inferenceReady() {
  const online = location.protocol !== 'file:';
  $('analyse').disabled = busy || !online || !state.can_predict || !$('media-file').files.length;
  if (!busy && !$('inference-results').dataset.complete) $('inference-status').textContent = !online ? 'Start the local dashboard server to analyse media.' : !state.can_predict ? 'Inference unlocks when dependencies, the face detector, and a trained checkpoint are available.' : 'Analysis stays local. Uploaded media is removed from the temporary folder when processing finishes.';
}
function showVerdict(v, maxFrames) {
  const box = $('verdict'); box.replaceChildren(); box.hidden = !v; if (!v) return;
  box.className = `verdict ${v.label}`;
  box.append(element('strong', {manipulated: 'Likely manipulated', real: 'No manipulation detected', uncertain: 'Uncertain: too close to call', unable_to_assess: 'Unable to assess'}[v.label] || v.label));
  const band = v.uncertain_range ? ` · uncertain between ${decimal(v.uncertain_range[0])} and ${decimal(v.uncertain_range[1])}` : '';
  if (Number.isFinite(v.score)) box.append(element('p', `Score ${decimal(v.score)} against threshold ${decimal(v.threshold)}${band} · ${v.mode === 'temporal' ? `mean of ${v.units} LSTM windows, as in evaluation` : `mean of ${v.units} frame scores (no full 16-frame window)`}`));
  const notes = [];
  if (v.label === 'uncertain') notes.push('On held-out test videos, scores in this range were right only about 60% of the time, so no call is made.');
  if (v.max_faces > 1) notes.push('More than one face appeared. The model was evaluated on single-face videos only.');
  if (v.mode === 'temporal' && maxFrames) notes.push(`Only the first ${maxFrames} sampled frames (~${maxFrames / 8} s) are scored, matching the evaluated protocol.`);
  notes.push('Accuracy is measured on the datasets shown in Overview; fakes from generators unlike those can be missed, so "no manipulation detected" is not proof of authenticity.');
  box.append(element('small', notes.join(' ')));
}
$('media-file').addEventListener('change', () => {
  if (mediaURL) URL.revokeObjectURL(mediaURL); $('media-preview').replaceChildren();
  const file = $('media-file').files[0]; $('file-name').textContent = file?.name || 'No file selected';
  if (file) {mediaURL = URL.createObjectURL(file); const preview = document.createElement(file.name.toLowerCase().endsWith('.mp4') ? 'video' : 'img'); preview.src = mediaURL; if (preview.tagName === 'VIDEO') preview.controls = true; else preview.alt = 'Selected image for analysis'; $('media-preview').append(preview);}
  delete $('inference-results').dataset.complete; $('verdict').hidden = true; inferenceReady();
});
$('analyse').addEventListener('click', async () => {
  const file = $('media-file').files[0]; if (!file || !state.can_predict || busy) return;
  if (file.size > 200 * 1024 * 1024) { $('inference-status').textContent = 'File exceeds 200 MB. Use the command-line predictor for larger videos.'; return; }
  busy = true; inferenceReady(); $('analyse').textContent = 'Analysing…'; $('inference-status').textContent = 'Processing faces locally. Videos may take several minutes.';
  try {
    const query = new URLSearchParams({name: file.name, checkpoint: $('checkpoint-select').value});
    const response = await fetch(`/api/predict?${query}`, {method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: file}); const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Analysis failed');
    showVerdict(data.verdict, data.max_frames);
    // Per-face calls use the same band as the overall verdict (same log-odds margin around each face's threshold).
    const margin = data.verdict?.uncertain_range ? Math.log(data.verdict.uncertain_range[1] / (1 - data.verdict.uncertain_range[1])) - Math.log(data.verdict.threshold / (1 - data.verdict.threshold)) : 0;
    const assess = (score, threshold) => { const centre = Math.log(threshold / (1 - threshold)), logit = Math.log(score / (1 - score)); return Math.abs(logit - centre) < margin ? 'Uncertain' : score >= threshold ? 'Manipulated' : 'Real'; };
    const tracks = new Map();
    for (const record of data.records) for (const face of record.faces) {
      const track = tracks.get(face.track_id) || {frames: 0, windows: [], spatial: [], threshold: face.threshold};
      track.frames++;
      if (face.mode === 'temporal') { track.windows.push([record.timestamp, face.score]); track.threshold = face.threshold; } else track.spatial.push(face.score);
      tracks.set(face.track_id, track);
    }
    const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
    table('inference-results', ['Face track', 'Frames', 'LSTM windows', 'Mean score', 'Threshold', 'Assessment'], [...tracks].map(([id, t]) => {
      const score = t.windows.length ? mean(t.windows.map(p => p[1])) : mean(t.spatial);
      return [`Face ${id}`, number(t.frames), t.windows.length ? number(t.windows.length) : '—', decimal(score), decimal(t.threshold), assess(score, t.threshold)];
    }));
    $('inference-results').dataset.complete = 'true'; $('inference-status').textContent = tracks.size ? `Analysis complete. ${tracks.size} face track${tracks.size > 1 ? 's' : ''} assessed. A score is not a guarantee of authenticity.` : 'Unable to assess: no usable faces were detected.';
    const timelines = [...tracks].filter(([, t]) => t.windows.length);
    $('inference-chart').hidden = !timelines.length;
    if (timelines.length) {
      const xMax = Math.max(1, ...timelines.flatMap(([, t]) => t.windows.map(p => p[0])));
      FGCharts.plot($('inference-chart'), [...timelines.map(([id, t]) => ({name: `Face ${id}`, points: t.windows})), {name: 'Decision threshold', points: [[0, timelines[0][1].threshold], [xMax, timelines[0][1].threshold]]}], {title: 'Manipulation score per LSTM window', xLabel: 'Seconds', yLabel: 'Score', xMax});
    }
  } catch (error) { $('inference-status').textContent = error.message; }
  finally {busy = false; $('analyse').textContent = 'Analyse media'; $('analyse').disabled = !state.can_predict;}
});
$('report-select').addEventListener('change', () => {selectOptions($('domain-select'), Object.keys(currentReport()?.metrics?.temporal || {}).map(d => [d, d]), 'No evaluated dataset'); shown = 50; evaluation(); overview();});
['domain-select', 'model-select'].forEach(id => $(id).addEventListener('change', () => {shown = 50; evaluation(); overview();}));
$('history-select').addEventListener('change', training);
$('prediction-search').addEventListener('input', () => {shown = 50; predictionTable();});
$('more-predictions').addEventListener('click', () => {shown += 50; predictionTable();});
$('refresh').addEventListener('click', refresh);
$('export').addEventListener('click', () => download('frameguard-results.json', JSON.stringify(state, null, 2), 'application/json'));
document.querySelectorAll('[data-save-chart]').forEach(button => button.addEventListener('click', () => {const svg = button.closest('.panel').querySelector('svg'); if (svg) download('frameguard-chart.svg', new XMLSerializer().serializeToString(svg), 'image/svg+xml');}));
window.addEventListener('hashchange', route);
render(); route(); refresh();
if (location.protocol !== 'file:') setInterval(refresh, 15000);
