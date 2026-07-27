(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const ui = {
    run: $('run-select'), batch: $('batch-select'), coordinate: $('coordinate-select'), rate: $('rate-select'), slider: $('frame-slider'), label: $('frame-label'), play: $('play'), prev: $('previous'), next: $('next'), loop: $('loop'), svg: $('network'), status: $('load-status'), file: $('file-input'), frameInfo: $('frame-info'), selection: $('selection-info'), metrics: $('metrics'), events: $('events'), form: $('experiment-form'), runStatus: $('run-status'), overview: $('overview-cards'),
    searchForm: $('evolution-form'), searchSeed: $('search-seed'), searchSamples: $('search-samples'), searchGenerations: $('search-generations'), searchPopulation: $('search-population'), random: $('start-random'), cma: $('start-cma'), searchStatus: $('search-status'), searchProgress: $('search-progress'), searchProgressLabel: $('search-progress-label'), chart: $('fitness-chart'), searchCards: $('search-cards'), process: $('process-table'), result: $('result-summary'), outcomeStatus: $('outcome-status'), liveForm: $('live-form'), liveModel: $('live-model-select'), liveModelScope: $('live-model-scope'), liveModelDetail: $('live-model-detail'), liveRefresh: $('refresh-live-models'), liveStatus: $('live-status'), livePlay: $('live-play'), liveStep: $('live-step'), liveRate: $('live-rate'), liveLabel: $('live-frame-label'), workspace: $('shared-workspace'), replayHost: $('replay-workspace-host'), liveHost: $('live-workspace-host'),
    asyncForm: $('async-form'), asyncSeed: $('async-seed'), asyncRun: $('async-run'), asyncDiagnostic: $('async-diagnostic'), asyncRefresh: $('async-refresh'), asyncStatus: $('async-status'), asyncProgress: $('async-progress'), asyncProgressLabel: $('async-progress-label'), asyncMetrics: $('async-metrics'), asyncSlots: $('async-slots'), asyncCauses: $('async-causes'), asyncCurriculum: $('async-curriculum'), asyncCurriculumCopy: $('async-curriculum-copy'), asyncCandidates: $('async-candidates'), asyncDetail: $('async-detail'), asyncArtifacts: $('async-artifacts'), asyncLearningState: $('async-learning-state'), asyncLearningCopy: $('async-learning-copy'), asyncRunFacts: $('async-run-facts'), asyncEstimate: $('async-work-estimate'), asyncCandidateBudget: $('async-candidates-budget'), asyncSlotsInput: $('async-slots-input'), asyncReplicasInput: $('async-replicas-input'), asyncBatchInput: $('async-batch-input'), asyncStateWidth: $('async-state-width'), asyncTicksInput: $('async-ticks-input')
  };
  const state = { data: null, runName: '', frame: 0, batch: 0, coordinate: 0, selected: null, playing: false, lastTick: 0, layout: [], job: null, jobTimer: null, live: null, liveModels: [] };
  const NS = 'http://www.w3.org/2000/svg';
  const element = (tag, attrs = {}) => { const node = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value)); return node; };
  const number = (value) => Number.isFinite(value) ? Number(value).toFixed(5) : '—';
  const apiError = (data, fallback) => Array.isArray(data?.detail)
    ? data.detail.map((item) => item.msg).join('; ')
    : (data?.detail || data?.error || fallback);

  function activate(view) {
    if (view === 'live') ui.liveHost.append(ui.workspace); else ui.replayHost.append(ui.workspace);
    document.body.dataset.view = view;
    document.querySelectorAll('.view').forEach((item) => item.classList.toggle('active', item.dataset.view === view));
    document.querySelectorAll('.nav').forEach((item) => item.classList.toggle('active', item.dataset.view === view));
  }
  function load(data) {
    if (!data || ![1, 2].includes(data.schema_version) || !data.graph || !data.runs) {
      const keys = data && typeof data === 'object' ? Object.keys(data).join(', ') : typeof data;
      throw new Error(`Expected a replay JSON from an evolution run's replays folder (for example generation-0-demo.json). Found: ${keys || 'no JSON object'}.`);
    }
    state.live = null; state.playing = false; ui.livePlay.textContent = 'Play'; state.data = data; state.runName = Object.keys(data.runs)[0]; state.frame = state.batch = state.coordinate = 0; state.selected = null;
    ui.run.replaceChildren(...Object.keys(data.runs).map((name) => new Option(name, name))); refreshSelectors(); drawGraph(); update(); updateOverview();
    ui.status.textContent = `Loaded ${data.graph.nodes} nodes and ${data.graph.edges.length} directed edges.`;
  }
  function activeRun() { return state.data.runs[state.runName]; }
  function trajectory() { return activeRun().trajectory; }
  function refreshSelectors() { const first = trajectory().node_states[0]; ui.batch.replaceChildren(...first.map((_, index) => new Option(`batch ${index}`, index))); ui.coordinate.replaceChildren(...first[0][0].map((_, index) => new Option(`coordinate ${index}`, index))); ui.slider.max = String(trajectory().steps.length - 1); ui.slider.value = String(state.frame); }
  function updateOverview() {
    if (!state.data) return;
    const metrics = activeRun().metrics, width = trajectory().node_states[0][0][0].length;
    const flags = Object.values(metrics).filter((value) => value && typeof value === 'object').reduce((count, value) => count + Number(value.bounded === false || value.non_silent === false || value.diverse === false || value.responsive === false || value.recovered === false || value.saturated === true), 0);
    const cards = [['Replay data', `${state.data.graph.nodes} nodes / ${state.data.graph.edges.length} edges`], ['State width', String(width)], ['Latest viability', flags ? `${flags} metric flag${flags === 1 ? '' : 's'}` : 'No metric flags']];
    ui.overview.replaceChildren(...cards.map(([label, value]) => statCard(label, value)));
  }
  function statCard(label, value) { const card = document.createElement('article'); card.className = 'stat'; const title = document.createElement('span'); title.textContent = label; const result = document.createElement('strong'); result.textContent = value; card.append(title, result); return card; }
  function nodePositions(count, edges) {
    if (count === 1) return [{x:360, y:280}]; const columns = Math.ceil(Math.sqrt(count * 1.25)), rows = Math.ceil(count / columns), left = 70, right = 650, top = 64, bottom = 500;
    const anchors = Array.from({length: count}, (_, index) => ({x: left + (index % columns) * (right - left) / Math.max(1, columns - 1), y: top + Math.floor(index / columns) * (bottom - top) / Math.max(1, rows - 1)})); const points = anchors.map((point) => ({...point}));
    for (let iteration = 0; iteration < 180; iteration += 1) { const force = Array.from({length: count}, () => ({x:0, y:0})); for (let a = 0; a < count; a += 1) for (let b = a + 1; b < count; b += 1) { const dx = points[a].x - points[b].x, dy = points[a].y - points[b].y, distance = Math.max(1, Math.hypot(dx, dy)), amount = 2100 / (distance * distance), x = dx / distance * amount, y = dy / distance * amount; force[a].x += x; force[a].y += y; force[b].x -= x; force[b].y -= y; } edges.forEach((edge) => { const a = edge.source, b = edge.target, dx = points[a].x - points[b].x, dy = points[a].y - points[b].y, distance = Math.max(1, Math.hypot(dx, dy)), amount = (distance - 118) * .025, x = dx / distance * amount, y = dy / distance * amount; force[a].x -= x; force[a].y -= y; force[b].x += x; force[b].y += y; }); points.forEach((point, index) => { force[index].x += (anchors[index].x - point.x) * .055; force[index].y += (anchors[index].y - point.y) * .055; point.x = Math.max(28, Math.min(692, point.x + Math.max(-6, Math.min(6, force[index].x)))); point.y = Math.max(28, Math.min(532, point.y + Math.max(-6, Math.min(6, force[index].y)))); }); }
    return points;
  }
  function drawGraph() { state.layout = nodePositions(state.data.graph.nodes, state.data.graph.edges); ui.svg.replaceChildren(); const defs = element('defs'), marker = element('marker', {id:'arrow', markerWidth:'7', markerHeight:'7', refX:'6', refY:'3.5', orient:'auto'}); marker.append(element('path', {d:'M0,0 L7,3.5 L0,7 z', fill:'#71809b'})); defs.append(marker); ui.svg.append(defs); const edgeLayer = element('g'), nodeLayer = element('g'); ui.svg.append(edgeLayer, nodeLayer); state.data.graph.edges.forEach((edge, index) => { const a = state.layout[edge.source], b = state.layout[edge.target], line = element('line', {x1:a.x, y1:a.y, x2:b.x, y2:b.y, 'marker-end':'url(#arrow)', class:'edge'}); line.addEventListener('click', () => { state.selected = {kind:'edge', index}; update(); }); edgeLayer.append(line); }); state.layout.forEach((point, index) => { const group = element('g'), circle = element('circle', {cx:point.x, cy:point.y, r:'12', class:'node'}), label = element('text', {x:point.x, y:point.y + 4, class:'node-label'}); label.textContent = index; circle.addEventListener('click', () => { state.selected = {kind:'node', index}; update(); }); group.append(circle, label); nodeLayer.append(group); }); }
  function currentValues() { return trajectory().node_states[state.frame][state.batch].map((vector) => vector[state.coordinate]); }
  function color(value, scale) { const t = Math.min(1, Math.abs(value) / scale), base = value >= 0 ? [250,130,103] : [90,169,255], mix = base.map((item) => Math.round(198 + (item - 198) * t)); return `rgb(${mix.join(',')})`; }
  function strengths(trace) { return trace.effective_edge_strengths?.[state.frame]?.[state.batch] ?? []; }
  function update() { if (!state.data) return; const trace = trajectory(), values = currentValues(), scale = Math.max(.15, ...values.map(Math.abs)), edgeStrengths = strengths(trace); ui.slider.value = String(state.frame); ui.label.value = `step ${trace.steps[state.frame]} | t=${Number(trace.times[state.frame]).toFixed(3)}`; ui.svg.querySelectorAll('.node').forEach((node, index) => { const value = values[index]; node.setAttribute('r', '12'); node.setAttribute('fill', color(value, scale)); node.classList.toggle('selected', state.selected?.kind === 'node' && state.selected.index === index); }); ui.svg.querySelectorAll('.edge').forEach((line, index) => { const baseWeight = state.data.graph.edges[index].weight, strength = edgeStrengths[index] ?? 1, effectiveWeight = baseWeight * strength; line.setAttribute('stroke', effectiveWeight >= 0 ? '#f39b72' : '#6daafa'); line.setAttribute('stroke-width', String(.7 + Math.min(3.6, Math.abs(effectiveWeight) * 2.5))); line.setAttribute('opacity', String(.10 + Math.min(.80, Math.abs(effectiveWeight)))); line.classList.toggle('selected', state.selected?.kind === 'edge' && state.selected.index === index); }); updateDiagnostics(values); updateSelection(values); updateMetrics(); updateEvents(); }
  function updateDiagnostics(values) { const trace = trajectory(), frame = trace.node_states[state.frame][state.batch], effectiveStrengths = strengths(trace), finite = values.every(Number.isFinite), magnitude = values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length, config = state.data.simulation_config || {}; const rows = [['run', state.runName], ['recorded frame', `${state.frame + 1} / ${trace.steps.length}`], ['integration dt', config.dt ?? 'not recorded'], ['state vector width', frame[0].length], ['mean |selected coordinate|', magnitude.toFixed(5)], ['all finite', finite ? 'yes' : 'NO'], ['edge-state width', trace.edge_states[state.frame]?.[state.batch]?.[0]?.length ?? 0], ['mean communication strength', effectiveStrengths.length ? (effectiveStrengths.reduce((sum, value) => sum + value, 0) / effectiveStrengths.length).toFixed(5) : 'fixed']]; ui.frameInfo.replaceChildren(...rows.flatMap(([term, value]) => { const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = term; dd.textContent = value; return [dt, dd]; })); }
  function updateSelection(values) { if (!state.selected) { ui.selection.textContent = 'Select a node or edge.'; return; } const trace = trajectory(), item = state.selected, edgeStrengths = strengths(trace); if (item.kind === 'node') { const edges = state.data.graph.edges.map((edge, index) => ({edge, index})).filter(({edge}) => edge.source === item.index || edge.target === item.index).map(({edge,index}) => { const strength = edgeStrengths[index] ?? 1; return {edge:index, direction:edge.source === item.index ? 'out' : 'in', other:edge.source === item.index ? edge.target : edge.source, base_weight:+edge.weight.toFixed(5), communication_strength:+strength.toFixed(5), effective_weight:+(edge.weight * strength).toFixed(5)}; }); ui.selection.textContent = JSON.stringify({kind:'node', node:item.index, state:trace.node_states[state.frame][state.batch][item.index], external_input:trace.inputs[state.frame][state.batch][item.index], selected_coordinate:values[item.index], incident_edges:edges}, null, 2); } else { const edge = state.data.graph.edges[item.index], vector = trace.edge_states[state.frame]?.[state.batch]?.[item.index] ?? [], strength = edgeStrengths[item.index] ?? 1; ui.selection.textContent = JSON.stringify({kind:'edge', edge:item.index, source:edge.source, target:edge.target, base_weight:edge.weight, edge_state:vector, communication_strength:strength, effective_weight:edge.weight * strength, source_state:trace.node_states[state.frame][state.batch][edge.source], target_state:trace.node_states[state.frame][state.batch][edge.target]}, null, 2); } }
  function updateMetrics() { const metrics = activeRun().metrics; ui.metrics.replaceChildren(...Object.entries(metrics).map(([name, value]) => { const row = document.createElement('div'); row.className = 'metric'; const label = document.createElement('span'); label.textContent = name.replaceAll('_', ' '); const outcome = document.createElement('strong'); const status = value.bounded ?? value.non_silent ?? value.diverse ?? value.responsive ?? value.recovered; const pass = name === 'saturation' ? !value.saturated : status; outcome.textContent = pass === undefined ? JSON.stringify(value) : (pass ? 'pass' : 'flag'); row.append(label, outcome); return row; })); }
  function updateEvents() { const step = trajectory().steps[state.frame]; ui.events.replaceChildren(...trajectory().events.map((event) => { const entry = document.createElement('div'); entry.className = `event ${step >= event.start && step <= event.end + 1 ? 'active' : ''}`; entry.textContent = `${event.kind}: ${event.start}–${event.end}`; return entry; })); }
  function move(delta) { const last = trajectory().steps.length - 1; state.frame += delta; if (state.frame > last) state.frame = ui.loop.checked ? 0 : last; if (state.frame < 0) state.frame = ui.loop.checked ? last : 0; update(); }
  async function advanceLive() {
    if (!state.live || state.live.pending) return;
    state.live.pending = true;
    try {
      const response = await fetch(`/api/live/sessions/${state.live.id}/step`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({steps:1})}), snapshot = await response.json();
      if (!response.ok) throw new Error(apiError(snapshot, 'sandbox step failed'));
      applyLiveSnapshot(snapshot);
    } catch (error) { state.playing = false; ui.livePlay.textContent = 'Play'; ui.liveStatus.textContent = `Sandbox update failed: ${error.message}`; } finally { if (state.live) state.live.pending = false; }
  }
  function animate(now) {
    if (!state.playing) return;
    if (state.live) { if (now - state.lastTick >= 1000 / Number(ui.liveRate.value)) { state.lastTick = now; advanceLive(); } requestAnimationFrame(animate); return; }
    if (now - state.lastTick >= 1000 / Number(ui.rate.value)) { move(1); state.lastTick = now; if (state.frame === trajectory().steps.length - 1 && !ui.loop.checked) { state.playing = false; ui.play.textContent = 'Play'; return; } } requestAnimationFrame(animate);
  }

  async function startJob(endpoint) { const payload = {samples:Number(ui.searchSamples.value), generations:Number(ui.searchGenerations.value), population:Number(ui.searchPopulation.value)}; if (ui.searchSeed.value.trim() !== '') payload.seed = Number(ui.searchSeed.value); ui.random.disabled = ui.cma.disabled = true; ui.searchStatus.textContent = 'Starting local background job…'; try { const response = await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}), data = await response.json(); if (!response.ok) throw new Error(apiError(data, 'could not start evolution job')); state.job = {id:data.job_id, samples:[], generations:[]}; activate('evolution'); pollJob(); } catch (error) { ui.searchStatus.textContent = `Could not start: ${error.message}`; ui.random.disabled = ui.cma.disabled = false; } }
  async function pollJob() { if (!state.job) return; try { const response = await fetch(`/api/jobs/${state.job.id}`), job = await response.json(); if (!response.ok) throw new Error(job.detail || job.error || 'job status unavailable'); state.job = job; renderJob(job); if (job.status === 'running') { state.jobTimer = window.setTimeout(pollJob, 450); } else { ui.random.disabled = ui.cma.disabled = false; } } catch (error) { ui.searchStatus.textContent = `Status unavailable: ${error.message}`; ui.random.disabled = ui.cma.disabled = false; } }
  function renderJob(job) { const samples = job.samples || [], generations = job.generations || [], fitnesses = samples.map((sample) => sample.fitness), best = fitnesses.length ? Math.max(...fitnesses) : null, mean = fitnesses.length ? fitnesses.reduce((sum, value) => sum + value, 0) / fitnesses.length : null, viable = samples.length ? samples.reduce((sum, sample) => sum + sample.viable_fraction, 0) / samples.length : null, initialLabel = job.kind === 'search' ? 'generation 0 population' : 'random diagnostic samples'; const phase = job.status === 'failed' ? `Failed: ${job.error}` : job.status === 'complete' ? `Complete · seed ${job.seed}` : `${String(job.phase).replaceAll('_', ' ')} in progress · seed ${job.seed}`; ui.searchStatus.textContent = phase; ui.searchProgress.max = job.samples_total || 1; ui.searchProgress.value = Math.min(samples.length, job.samples_total || 1); ui.searchProgressLabel.textContent = `${samples.length} / ${job.samples_total || 0} ${initialLabel}${generations.length ? ` · ${generations.length} CMA generation${generations.length === 1 ? '' : 's'}` : ''}`; ui.searchCards.replaceChildren(statCard('Best fitness', best === null ? '—' : number(best)), statCard('Mean fitness', mean === null ? '—' : number(mean)), statCard('Viable fraction', viable === null ? '—' : `${(100 * viable).toFixed(1)}%`)); drawFitnessChart(samples, generations, job.kind); renderProcess(samples, generations); renderResult(job.result, job.status); }
  function drawFitnessChart(samples, generations, kind) {
    const canvas = ui.chart, context = canvas.getContext('2d'), width = canvas.width, height = canvas.height, split = width * .62, top = 48, bottom = height - 42, plotHeight = bottom - top, bounds = (series) => { const minimum = Math.min(0, ...series), maximum = Math.max(0, ...series), padding = Math.max(.02, (maximum - minimum) * .08); return {minimum: minimum - padding, maximum: maximum + padding}; }, drawGrid = (left, right, range, ticks = 4) => { const y = (value) => bottom - (value - range.minimum) / (range.maximum - range.minimum) * plotHeight; context.strokeStyle = '#33425d'; context.lineWidth = 1; for (let tick = 0; tick <= ticks; tick += 1) { const value = range.minimum + (range.maximum - range.minimum) * tick / ticks, py = y(value); context.beginPath(); context.moveTo(left, py); context.lineTo(right, py); context.stroke(); context.fillStyle = '#9caac2'; context.fillText(value.toFixed(2), left === 52 ? 8 : left, py - 3); } if (range.minimum < 0 && range.maximum > 0) { const py = y(0); context.strokeStyle = '#71809e'; context.beginPath(); context.moveTo(left, py); context.lineTo(right, py); context.stroke(); } return y; };
    const initialTitle = kind === 'search' ? 'CMA-ES smoke samples — full scale' : 'Random-search diagnostic — full scale'; context.clearRect(0, 0, width, height); context.fillStyle = '#111724'; context.fillRect(0, 0, width, height); context.font = '12px system-ui'; context.fillStyle = '#9caac2'; context.fillText(initialTitle, 24, 24); context.fillText(generations.length ? 'CMA-ES training fitness by generation' : 'Low-score zoom (outlier excluded)', split + 22, 24);
    if (!samples.length) { context.fillText('Fitness samples will appear here while the local evaluator runs.', 24, 58); return; }
    const values = samples.map((sample) => sample.fitness), left = 52, mainRight = split - 18, mainWidth = mainRight - left, mainRange = bounds(values), y = drawGrid(left, mainRight, mainRange), x = (index) => left + index / Math.max(1, values.length - 1) * mainWidth;
    context.fillStyle = '#9caac2'; context.fillText('sample number', mainRight - 88, height - 15); context.fillText('1', left - 3, height - 15); context.fillText(String(values.length), mainRight - 8, height - 15);
    const bestIndex = values.indexOf(Math.max(...values)); values.forEach((value, index) => { context.fillStyle = samples[index].viable_fraction > 0 ? '#63d5c2' : '#f39b72'; context.beginPath(); context.arc(x(index), y(value), index === bestIndex ? 6 : 4, 0, 2 * Math.PI); context.fill(); }); context.strokeStyle = '#63d5c2'; context.beginPath(); context.moveTo(x(bestIndex), y(values[bestIndex]) + 8); context.lineTo(x(bestIndex), Math.max(top, y(values[bestIndex]) - 18)); context.stroke(); context.fillStyle = '#63d5c2'; context.fillText(`sample ${bestIndex + 1}: ${number(values[bestIndex])}`, Math.max(left, x(bestIndex) - 48), Math.max(38, y(values[bestIndex]) - 22));
    const rightLeft = split + 28, right = width - 28, rightWidth = right - rightLeft;
    if (!generations.length) { const sorted = [...values].sort((a, b) => a - b), zoomCeiling = sorted[Math.max(0, Math.floor((sorted.length - 1) * .85))], zoomValues = values.filter((value) => value <= zoomCeiling), zoomRange = bounds(zoomValues), zoomY = drawGrid(rightLeft, right, zoomRange, 3), zoomX = (index) => rightLeft + index / Math.max(1, values.length - 1) * rightWidth; values.forEach((value, index) => { if (value <= zoomCeiling) { context.fillStyle = samples[index].viable_fraction > 0 ? '#63d5c2' : '#f39b72'; context.beginPath(); context.arc(zoomX(index), zoomY(value), 4, 0, 2 * Math.PI); context.fill(); } }); context.fillStyle = '#9caac2'; context.fillText(`zoom ceiling ${number(zoomCeiling)}`, rightLeft, height - 15); return; }
    const series = generations.flatMap((item) => [item.best_fitness, item.mean_fitness]), cmaRange = bounds(series), cy = drawGrid(rightLeft, right, cmaRange, 3), cx = (index) => rightLeft + index / Math.max(1, generations.length - 1) * rightWidth; [["best_fitness", '#d1a9ff', 'best'], ["mean_fitness", '#63d5c2', 'mean']].forEach(([key, color, label], lineIndex) => { context.strokeStyle = color; context.lineWidth = 2; context.beginPath(); generations.forEach((generation, index) => { const px = cx(index), py = cy(generation[key]); if (index) context.lineTo(px, py); else context.moveTo(px, py); }); context.stroke(); context.fillStyle = color; context.fillText(label, rightLeft + lineIndex * 58, height - 15); }); context.fillStyle = '#9caac2'; context.fillText(`latest ${number(generations[generations.length - 1].best_fitness)}`, right - 100, height - 15);
  }
  function renderProcess(samples, generations) { const fragment = document.createDocumentFragment(); const heading = document.createElement('div'); heading.className = 'process-row heading'; heading.textContent = 'Recent event                                  Fitness / detail'; fragment.append(heading); [...samples.slice(-8).map((sample) => ({kind:`sample ${sample.sample}`, detail:`fitness ${number(sample.fitness)} · viable ${Number(sample.viable_fraction).toFixed(2)} · failures ${sample.failed_scenarios}`})), ...generations.slice(-8).map((generation) => ({kind:`generation ${generation.generation}`, detail:`best ${number(generation.best_fitness)} · mean ${number(generation.mean_fitness)} · sigma ${number(generation.sigma)}`}))].slice(-10).forEach((row) => { const line = document.createElement('div'); line.className = 'process-row'; const name = document.createElement('strong'), detail = document.createElement('span'); name.textContent = row.kind; detail.textContent = row.detail; line.append(name, detail); fragment.append(line); }); if (!samples.length && !generations.length) { const empty = document.createElement('p'); empty.textContent = 'Start a job to see individual samples and generation summaries.'; fragment.append(empty); } ui.process.replaceChildren(fragment); }
  function renderResult(result, status) {
    if (!result) { ui.outcomeStatus.textContent = status === 'failed' ? 'Job failed' : 'Waiting for a completed job'; ui.result.textContent = status === 'failed' ? 'The job failed before it produced an export.' : 'Final validation, test, and artifact links appear here after CMA-ES completes.'; return; }
    if (result.smoke_report) { const report = result.smoke_report; ui.outcomeStatus.textContent = report.meaningful ? 'Random-search suite is non-degenerate: CMA-ES may proceed' : 'Random-search suite is degenerate: CMA-ES remains blocked'; ui.result.replaceChildren(statCard('Suite gate', report.meaningful ? 'PASS' : 'BLOCKED'), statCard('Best sampled fitness', number(report.maximum)), statCard('Mean ± SD', `${number(report.mean)} ± ${number(report.standard_deviation)}`), statCard('Sampled range', `${number(report.minimum)} – ${number(report.maximum)}`)); return; }
    ui.outcomeStatus.textContent = 'CMA-ES complete: this genome was re-evaluated on held-out validation and test suites'; ui.result.replaceChildren(statCard('Best training fitness', number(result.best_fitness)), statCard('Validation fitness', number(result.validation_fitness)), statCard('Test fitness', number(result.test_fitness)), statCard('Train → test gap', number(result.best_fitness - result.test_fitness)));
    const links = document.createElement('p'); const report = document.createElement('a'); report.href = result.output_url; report.textContent = 'Complete experiment report'; report.target = '_blank'; const genome = document.createElement('a'); genome.href = result.best_genome_url; genome.textContent = 'Best genome'; genome.target = '_blank'; const analysis = document.createElement('a'); analysis.href = result.analysis_url; analysis.textContent = 'Analysis data'; analysis.target = '_blank'; links.append(report, document.createTextNode(' · '), genome, document.createTextNode(' · '), analysis); ui.result.append(links);
    const previews = document.createElement('div'); previews.className = 'analysis-previews'; [['Held-out trajectory', result.trajectory_svg_url], ['Perturbation recovery', result.recovery_svg_url]].forEach(([label, url]) => { const figure = document.createElement('figure'); const caption = document.createElement('figcaption'); caption.textContent = label; const image = document.createElement('img'); image.src = url; image.alt = `${label} analysis`; figure.append(caption, image); previews.append(figure); }); ui.result.append(previews);
  }
  function liveModelLabel(model) {
    if (model.source !== 'survival') return `legacy · ${model.run_id} · validation ${number(model.validation_fitness)} · test ${number(model.test_fitness)}`;
    const functional = model.functional ? 'functional' : 'not functional';
    return `#${model.global_rank} · run ${model.run_id.slice(0, 8)} · elite ${model.elite_rank} · stage ${model.stage} · ${model.lifetime} ticks · ${functional} · burden ${Number(model.worst_pathology_burden).toFixed(3)}`;
  }

  function renderLiveModelDetail() {
    const model = state.liveModels.find((item) => item.id === ui.liveModel.value);
    if (!model) { ui.liveModelDetail.innerHTML = '<p>No model matches this filter.</p>'; return; }
    const intro = document.createElement('div'), title = document.createElement('h4'), copy = document.createElement('p');
    title.textContent = model.source === 'survival' ? `Overall survival rank #${model.global_rank}` : 'Legacy fixed-window comparison';
    copy.textContent = model.source === 'survival'
      ? `Candidate ${model.candidate_id} is elite ${model.elite_rank} within run ${model.run_id.slice(0, 8)}. Ranking is lexicographic—not an invented scalar score.`
      : `Run ${model.run_id}. Legacy fitness is not directly comparable with death-driven survival evidence.`;
    intro.append(title, copy);
    if (model.global_rank === 1) { const badge = document.createElement('span'); badge.className = 'recommended'; badge.textContent = 'RECOMMENDED: strongest discovered survival rank'; intro.prepend(badge); }
    const facts = model.source === 'survival' ? [
      ['run type', model.run_kind === 'training' ? 'configured training' : '80-tick smoke test'],
      ['node state coordinates', model.node_state_width ?? 'unknown'],
      ['edge state coordinates', model.edge_state_width ?? 'unknown'],
      ['curriculum evidence', `stage ${model.stage}; survived ${model.lifetime} ticks`],
      ['functional across replicas', model.functional ? 'yes' : 'no'],
      ['worst pathology burden', Number(model.worst_pathology_burden).toFixed(4)],
      ['minimum input response', Number(model.minimum_response).toExponential(3)],
      ['minimum graph propagation', Number(model.minimum_propagation).toExponential(3)],
      ['minimum probe separation', Number(model.minimum_distinguishability).toExponential(3)],
      ['recovered in every replica', model.recovered_across_replicas ? 'yes' : 'no'],
    ] : [
      ['target', model.target], ['node state coordinates', model.node_state_width ?? 'unknown'], ['edge state coordinates', model.edge_state_width ?? 'unknown'], ['validation fitness', number(model.validation_fitness)], ['test fitness', number(model.test_fitness)],
    ];
    const list = document.createElement('dl'); facts.forEach(([key, value]) => { const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = key; dd.textContent = value; list.append(dt, dd); });
    ui.liveModelDetail.replaceChildren(intro, list);
  }

  function renderLiveModelOptions() {
    const prior = ui.liveModel.value, scope = ui.liveModelScope.value;
    let visible = state.liveModels.filter((model) => scope === 'legacy' ? model.source === 'legacy' : model.source === 'survival');
    if (scope === 'best') {
      visible = visible.filter((model) => model.run_kind === 'training' && model.elite_rank === 1);
      if (!visible.length) visible = state.liveModels.filter((model) => model.source === 'survival' && model.elite_rank === 1);
    }
    ui.liveModel.replaceChildren(...visible.map((model) => new Option(liveModelLabel(model), model.id)));
    if (visible.some((model) => model.id === prior)) ui.liveModel.value = prior;
    renderLiveModelDetail();
    return visible.length;
  }

  async function refreshLiveModels() {
    ui.liveStatus.textContent = 'Finding trained survival elites and legacy exports…';
    try {
      const response = await fetch('/api/live/models'), data = await response.json();
      if (!response.ok) throw new Error(apiError(data, 'model list unavailable'));
      state.liveModels = data.models; const visible = renderLiveModelOptions();
      const survival = data.models.filter((model) => model.source === 'survival').length, legacy = data.models.length - survival, latest = data.latest_survival;
      if (latest?.available && Number(latest.report?.graduations || 0) === 0) {
        const causes = (latest.candidates || []).reduce((counts, candidate) => { const cause = causeLabel(candidate.death_cause); counts[cause] = (counts[cause] || 0) + 1; return counts; }, {});
        const leading = Object.entries(causes).sort((left, right) => right[1] - left[1]).slice(0, 2).map(([cause, count]) => `${count} ${cause}`).join(', ');
        ui.liveStatus.textContent = `Latest run ${latest.run_id.slice(0, 8)} produced no Live model: ${latest.report.completed_candidates || 0} candidates died before graduation${leading ? ` (${leading})` : ''}. ${data.models.length ? `Showing ${visible} older eligible model${visible === 1 ? '' : 's'}.` : 'Train again after adjusting the survival guards.'}`;
      } else {
        ui.liveStatus.textContent = data.models.length ? `Showing ${visible} of ${survival} survival elites; ${legacy} legacy export${legacy === 1 ? '' : 's'} also available.` : (latest?.available && Number(latest.report?.graduations || 0) ? `Latest run has ${latest.report.graduations} interim graduations, but no final-stage, functional, pathology-free Live model yet.` : 'No usable model exists yet. Complete final-stage functional survival validation.');
      }
    } catch (error) { ui.liveStatus.textContent = `Could not load exports: ${error.message}`; }
  }

  function liveDocument(snapshot) {
    return {schema_version:2, graph:snapshot.graph, simulation_config:snapshot.simulation_config, runs:{live_sandbox:{trajectory:{times:[snapshot.time], steps:[snapshot.step], node_states:[snapshot.node_state], edge_states:[snapshot.edge_state], effective_edge_strengths:[snapshot.effective_edge_strengths], inputs:[snapshot.inputs], events:[]}, metrics:{}}}};
  }
  function applyLiveSnapshot(snapshot) {
    const trace = trajectory();
    trace.times[0] = snapshot.time; trace.steps[0] = snapshot.step; trace.node_states[0] = snapshot.node_state; trace.edge_states[0] = snapshot.edge_state; trace.effective_edge_strengths[0] = snapshot.effective_edge_strengths; trace.inputs[0] = snapshot.inputs;
    state.frame = 0; update(); ui.liveLabel.textContent = `step ${snapshot.step} · t=${Number(snapshot.time).toFixed(3)} · in-memory only`;
    const safety = snapshot.last_safety_event, warning = snapshot.last_warning;
    if (safety?.details) {
      const detail = safety.details;
      ui.liveStatus.textContent = `Observation at step ${safety.step}: node ${detail.node}, coordinate ${detail.coordinate} proposed ${Number(detail.proposed).toFixed(4)} from ${Number(detail.previous).toFixed(4)}; after the per-step delta limit it was ${Number(detail.after_delta_limit).toFixed(4)}, so the displayed state is clipped to ±${detail.bound}. Sandbox continues for inspection.${warning ? ` Latest health warning: ${causeLabel(warning.cause)}.` : ''}`;
    } else if (warning) {
      ui.liveStatus.textContent = `Observation at step ${warning.step}: ${causeLabel(warning.cause)}. Sandbox continues for inspection.`;
    }
  }
  async function launchLive(event) {
    event.preventDefault();
    const fields = {model_id: ui.liveModel.value};
    [['seed','live-seed'], ['nodes','live-nodes'], ['batch_size','live-batch'], ['input_seed','live-input-seed']].forEach(([key, id]) => { fields[key] = Number($(id).value); });
    [['mean_degree','live-degree'], ['dt','live-dt'], ['input_standard_deviation','live-input-std']].forEach(([key, id]) => { fields[key] = Number($(id).value); });
    fields.topology = $('live-topology').value;
    ui.liveStatus.textContent = 'Opening in-memory sandbox…';
    try {
      const response = await fetch('/api/live/sessions', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(fields)}), snapshot = await response.json();
      if (!response.ok) throw new Error(apiError(snapshot, 'sandbox launch failed'));
      load(liveDocument(snapshot)); state.live = {id:snapshot.session_id, pending:false}; activate('live'); applyLiveSnapshot(snapshot); state.playing = true; state.lastTick = performance.now(); ui.livePlay.textContent = 'Pause'; requestAnimationFrame(animate);
      ui.liveStatus.textContent = `Loaded ${snapshot.model_id} on a new ${fields.topology} graph. State remains in memory only.`;
    } catch (error) { ui.liveStatus.textContent = `Could not launch: ${error.message}`; }
  }

  function asyncMetricValues(report) {
    return [
      `${Math.round(100 * (report.active_slot_utilization || 0))}%`,
      String(report.completed_candidates ?? 0),
      String(report.currently_living_right_censored ?? 0),
      String(report.viable_survivors ?? 0),
      String(report.optimizer_updates ?? 0),
      `L${report.curriculum_level ?? 0}`,
    ];
  }

  const causeLabels = {
    one_direction_degeneration: 'persistent one-way drift',
    trajectory_indistinguishable: 'probe trajectories became indistinguishable',
    state_homogenization: 'node states collapsed to one value',
    input_unresponsive: 'failed to respond to input',
    communication_unresponsive: 'failed to propagate a probe',
    communication_collapse: 'edge communication collapsed',
    edge_dynamics_inactive: 'edge dynamics never became active',
    edge_gate_saturation: 'edge gates stayed saturated',
    edge_runaway_growth: 'edge latent state kept growing',
    disturbance_unrecovered: 'failed to recover after disturbance',
    boundary_saturation: 'too many states saturated at the bound',
    absolute_safety_limit: 'crossed an absolute safety limit',
    nonfinite: 'produced a non-finite number',
    simulator_failure: 'simulation failed',
  };
  const causeLabel = (cause) => cause ? (causeLabels[cause] || cause.replaceAll('_', ' ')) : 'survival milestone reached';
  const sourceLabel = (source) => ({cma:'CMA-ES proposal', elite:'elite mutation', exploration:'random exploration', initial:'reference genome'}[source] || source);

  function renderLearningVerdict(report = {}, settings = {}, runKind = 'training') {
    const updates = Number(report.optimizer_updates || 0), completed = Number(report.completed_candidates || 0);
    let stateLabel = 'No optimizer update yet', copy = `${completed} candidate lives have finished, but CMA-ES has not received a complete comparable result batch.`;
    if (updates > 0 && updates < 5) { stateLabel = 'Warm-up: very early learning'; copy = `CMA-ES updated ${updates} time${updates === 1 ? '' : 's'}. This proves the loop is learning, but it is too early to infer convergence.`; }
    else if (updates >= 5 && updates < 20) { stateLabel = 'Training is underway'; copy = `CMA-ES has made ${updates} updates from completed survival evidence. Compare passage rates and elite changes before treating the result as stable.`; }
    else if (updates >= 20) { stateLabel = 'Substantial optimization history'; copy = `CMA-ES has made ${updates} updates. This is enough history to inspect trends, though held-out survival validation is still required.`; }
    if (runKind === 'diagnostic') copy += ' This run is an 80-tick smoke test, not a training budget.';
    ui.asyncLearningState.textContent = stateLabel;
    ui.asyncLearningCopy.textContent = copy;
    const stop = ({candidate_budget_reached:'candidate-life budget reached', tick_limit_reached:'safety tick limit reached', running:'still running'}[report.stop_reason] || report.stop_reason || 'older run completed');
    const origins = Object.entries(report.proposals_by_source || {}).map(([source, count]) => `${count} ${sourceLabel(source)}`).join(', ') || 'not recorded';
    const facts = [
      ['stop condition', stop],
      ['ticks elapsed', `${report.ticks_elapsed ?? '—'} / ${report.tick_limit ?? settings.max_ticks ?? '—'}`],
      ['candidate evidence', `${completed}${report.candidate_budget ? ` / ${report.candidate_budget} target` : ''}`],
      ['replica trajectories', `${report.completed_replica_lives ?? completed * Number(settings.replicas || 0)} completed`],
      ['outcomes', `${report.graduations ?? report.viable_survivors ?? '—'} graduated / ${report.deaths ?? '—'} died`],
      ['per-coordinate runaway guard', settings.node_growth_alert === undefined ? 'not recorded' : `|coordinate| >= ${settings.node_growth_alert}; ${settings.one_direction_steps} growing ticks`],
      ['proposal origins', origins],
    ];
    ui.asyncRunFacts.replaceChildren(...facts.flatMap(([key, value]) => { const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = key; dd.textContent = value; return [dt, dd]; }));
  }

  function renderAsyncSlots(slots = []) {
    if (!slots.length) {
      ui.asyncSlots.innerHTML = '<p class="empty-state">No active slot snapshot is available.</p>';
      return;
    }
    ui.asyncSlots.replaceChildren(...slots.map((slot) => {
      const lane = document.createElement('div'); lane.className = 'slot-lane';
      const name = document.createElement('span'); name.className = 'slot-name'; name.textContent = `SLOT ${String(slot.slot).padStart(2, '0')}`;
      const candidate = document.createElement('span'); candidate.className = 'candidate-name'; candidate.textContent = `candidate ${slot.candidate_id} · ${slot.source}`;
      const track = document.createElement('span'); track.className = 'lane-track';
      const fill = document.createElement('i'); fill.className = `lane-fill ${Number(slot.worst_burden) >= .5 ? 'burdened' : ''}`; fill.style.width = `${Math.min(100, 100 * Number(slot.age) / Math.max(1, Number(slot.milestone)))}%`; track.append(fill);
      const meta = document.createElement('span'); meta.className = 'lane-meta'; meta.textContent = `${slot.age} / ${slot.milestone} · L${slot.level}`;
      lane.append(name, candidate, track, meta); return lane;
    }));
  }

  function renderAsyncCauses(report) {
    const entries = Object.entries(report.deaths_per_cause || {}).sort((a, b) => b[1] - a[1]);
    if (!entries.length) {
      ui.asyncCauses.innerHTML = '<p class="empty-state">No deaths recorded.</p>';
      return;
    }
    const maximum = Math.max(...entries.map(([, count]) => count));
    ui.asyncCauses.replaceChildren(...entries.map(([cause, count]) => {
      const row = document.createElement('div'); row.className = 'cause-row';
      const label = document.createElement('span'); label.textContent = causeLabel(cause); label.title = cause;
      const bar = document.createElement('span'); bar.className = 'cause-bar';
      const fill = document.createElement('i'); fill.style.width = `${100 * count / maximum}%`; bar.append(fill);
      const value = document.createElement('strong'); value.textContent = count;
      row.append(label, bar, value); return row;
    }));
  }

  function renderAsyncCurriculum(report, settings = {}) {
    const rates = report.milestone_passage_rates || {};
    ui.asyncCurriculumCopy.textContent = `Current stage ${(report.curriculum_level ?? 0) + 1}; advancement requires at least 60% of the recent candidate lives to graduate.`;
    ui.asyncCurriculum.replaceChildren(...Object.entries(rates).map(([level, rate]) => {
      const item = document.createElement('div'); item.className = `curriculum-level ${Number(level) === Number(report.curriculum_level) ? 'active' : ''}`;
      const config = settings.levels?.[Number(level)] || {};
      const label = document.createElement('span'); label.textContent = `stage ${Number(level) + 1}${config.lifetime ? ` · survive ${config.lifetime} ticks` : ''}`;
      const value = document.createElement('strong'); value.textContent = rate == null ? 'not reached' : `${Math.round(100 * rate)}%`;
      const detail = document.createElement('small'); detail.textContent = rate == null ? 'No completed lives at this stage' : `${config.graph_nodes || '?'}-node graphs · graduation rate`;
      item.append(label, value, detail); return item;
    }));
  }

  function showAsyncCandidate(candidate, row) {
    ui.asyncCandidates.querySelectorAll('.candidate-row').forEach((item) => item.classList.remove('selected'));
    if (row) row.classList.add('selected');
    const title = document.createElement('h4'); title.textContent = `Candidate ${candidate.candidate_id}`;
    const list = document.createElement('dl');
    [['result', candidate.status === 'graduation' ? 'healthy graduation' : 'pathology death'], ['ticks lived', candidate.age], ['curriculum stage', candidate.level + 1], ['proposal origin', sourceLabel(candidate.source)], ['optimizer updates when proposed', candidate.optimizer_update], ['why it stopped', causeLabel(candidate.death_cause)]].forEach(([key, value]) => {
      const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = key; dd.textContent = value; list.append(dt, dd);
    });
    const replicas = candidate.replicas.map((replica, index) => {
      const block = document.createElement('div'); block.className = 'replica-line';
      const heading = document.createElement('strong'); heading.textContent = `Replica ${replica.index ?? index}`;
      const open = document.createElement('button'); open.type = 'button'; open.className = 'replay-survival'; open.textContent = 'Replay this exact life';
      open.addEventListener('click', async () => {
        open.disabled = true; open.textContent = 'Reconstructing...';
        try {
          const response = await fetch(replica.replay_url), document = await response.json();
          if (!response.ok) throw new Error(apiError(document, 'survival replay is unavailable'));
          load(document); activate('replay'); window.scrollTo({top: 0, behavior: 'smooth'});
        } catch (error) {
          evidence.textContent = `Could not reconstruct this survival replay: ${error.message}`;
        } finally { open.disabled = false; open.textContent = 'Replay this exact life'; }
      });
      const coordinateEvidence = replica.coordinate_responsiveness?.length ? ` · per-coordinate response [${replica.coordinate_responsiveness.map((value) => Number(value).toExponential(2)).join(', ')}] · propagation [${replica.coordinate_propagation.map((value) => Number(value).toExponential(2)).join(', ')}] · separation [${replica.coordinate_distinguishability.map((value) => Number(value).toExponential(2)).join(', ')}] · recovery [${replica.coordinate_recovered.map((value) => value ? 'ok' : 'failed').join(', ')}]` : '';
      const evidence = document.createElement('span'); evidence.textContent = `lived ${replica.age} ticks · worst pathology burden ${Number(replica.burden).toFixed(3)} · worst-coordinate input response ${Number(replica.responsiveness).toExponential(2)} · worst-coordinate graph propagation ${Number(replica.propagation).toExponential(2)} · worst-coordinate paired-probe separation ${Number(replica.distinguishability).toExponential(2)} · ${replica.recovered ? 'recovered after probe' : 'recovery not demonstrated'}${coordinateEvidence}`;
      block.append(heading, evidence, open); return block;
    });
    ui.asyncDetail.replaceChildren(title, list, ...replicas);
  }

  function renderAsyncCandidates(candidates = []) {
    if (!candidates.length) {
      ui.asyncCandidates.innerHTML = '<p class="empty-state">No completed candidates yet.</p>';
      return;
    }
    const heading = document.createElement('div'); heading.className = 'candidate-row heading';
    ['candidate', 'result', 'stage', 'ticks lived', 'why it stopped', 'origin'].forEach((label) => { const span = document.createElement('span'); span.textContent = label; heading.append(span); });
    const rows = candidates.slice().reverse().map((candidate) => {
      const row = document.createElement('button'); row.type = 'button'; row.className = 'candidate-row';
      const values = [candidate.candidate_id, candidate.status === 'graduation' ? 'graduated' : 'died', candidate.level + 1, candidate.age, causeLabel(candidate.death_cause), sourceLabel(candidate.source)];
      values.forEach((value, index) => { const span = document.createElement('span'); span.textContent = value; if (index === 1) span.className = `status-${candidate.status}`; row.append(span); });
      row.addEventListener('click', () => showAsyncCandidate(candidate, row)); return row;
    });
    ui.asyncCandidates.replaceChildren(heading, ...rows);
    showAsyncCandidate(candidates[candidates.length - 1], rows[0]);
  }

  function renderAsyncArtifacts(artifacts = {}) {
    ui.asyncArtifacts.replaceChildren(...Object.entries(artifacts).map(([label, url]) => {
      const link = document.createElement('a'); link.href = url; link.target = '_blank'; link.textContent = label; return link;
    }));
  }

  function renderAsyncData(data, slotsOverride) {
    const report = data.report || {};
    [...ui.asyncMetrics.querySelectorAll('strong')].forEach((node, index) => { node.textContent = asyncMetricValues(report)[index]; });
    renderAsyncSlots(slotsOverride || data.slots || []);
    renderLearningVerdict(report, data.settings || {}, data.run_kind || 'training');
    renderAsyncCauses(report); renderAsyncCurriculum(report, data.settings || {});
    if (data.candidates) renderAsyncCandidates(data.candidates);
    if (data.artifacts) renderAsyncArtifacts(data.artifacts);
  }

  async function loadLatestAsync() {
    ui.asyncStatus.textContent = 'Loading the latest survival run…';
    try {
      const response = await fetch('/api/async/latest'), data = await response.json();
      if (!response.ok) throw new Error(apiError(data, 'latest survival run is unavailable'));
      if (!data.available) { ui.asyncStatus.textContent = 'No survival run exists yet. Configure training above and start it.'; return; }
      renderAsyncData(data); ui.asyncProgress.max = data.report.candidate_budget || data.report.tick_limit || 1; ui.asyncProgress.value = data.report.candidate_budget ? data.report.completed_candidates : data.report.ticks_elapsed;
      ui.asyncProgressLabel.textContent = 'complete';
      ui.asyncStatus.textContent = `Loaded ${data.run_kind || 'survival'} run ${data.run_id}: ${data.report.completed_candidates} completed candidate lives.`;
    } catch (error) { ui.asyncStatus.textContent = `Could not load survival results: ${error.message}`; }
  }

  function setAsyncBusy(busy) { ui.asyncRun.disabled = busy; ui.asyncDiagnostic.disabled = busy; }

  async function pollAsyncJob(jobId) {
    try {
      const response = await fetch(`/api/jobs/${jobId}`), job = await response.json();
      if (!response.ok) throw new Error(job.detail || job.error || 'job status unavailable');
      if (job.status === 'running') {
        const snapshot = job.latest || {}, tick = Number(snapshot.tick || 0);
        const report = snapshot.report || {}, budget = Number(report.candidate_budget || 0), completed = Number(report.completed_candidates || 0);
        ui.asyncProgress.max = budget || Number(snapshot.max_ticks || 80); ui.asyncProgress.value = budget ? completed : tick;
        ui.asyncProgressLabel.textContent = budget ? `${completed} / ${budget} candidate lives · tick ${tick}` : `${tick} / ${snapshot.max_ticks || 80} ticks`;
        ui.asyncStatus.textContent = `${job.kind === 'async_training' ? 'Survival training' : 'Smoke test'} is running · seed ${job.seed}`;
        if (snapshot.report) renderAsyncData({report:snapshot.report}, snapshot.slots);
        window.setTimeout(() => pollAsyncJob(jobId), 300);
      } else if (job.status === 'complete') {
        setAsyncBusy(false); renderAsyncData(job.result); const report = job.result.report || {};
        ui.asyncProgress.max = report.candidate_budget || report.tick_limit || 1; ui.asyncProgress.value = report.candidate_budget ? report.completed_candidates : report.ticks_elapsed;
        ui.asyncProgressLabel.textContent = 'complete'; ui.asyncStatus.textContent = `${job.kind === 'async_training' ? 'Training' : 'Smoke test'} complete · ${report.completed_candidates} candidate lives · seed ${job.seed}`;
      } else { throw new Error(job.error || 'survival run failed'); }
    } catch (error) { setAsyncBusy(false); ui.asyncStatus.textContent = `Survival run stopped: ${error.message}`; }
  }

  async function startAsync(event) {
    event.preventDefault();
    const fields = {
      candidate_budget:'async-candidates-budget', max_ticks:'async-ticks-input', slots:'async-slots-input', replicas:'async-replicas-input', optimizer_batch:'async-batch-input', state_width:'async-state-width',
      stage_1_lifetime:'async-stage1-life', stage_2_lifetime:'async-stage2-life', stage_1_nodes:'async-stage1-nodes', stage_2_nodes:'async-stage2-nodes', mean_degree:'async-degree', input_scale:'async-input-scale',
      disturbance_interval:'async-disturbance-interval', disturbance_strength:'async-disturbance-strength', fatal_threshold:'async-fatal-threshold', node_growth_alert:'async-node-growth-alert', one_direction_steps:'async-one-direction-steps', probe_interval:'async-probe-interval',
    };
    const payload = Object.fromEntries(Object.entries(fields).map(([key, id]) => [key, Number($(id).value)]));
    if (ui.asyncSeed.value.trim() !== '') payload.seed = Number(ui.asyncSeed.value);
    setAsyncBusy(true); ui.asyncStatus.textContent = 'Starting configured survival training…'; ui.asyncProgress.max = payload.candidate_budget; ui.asyncProgress.value = 0;
    try {
      const response = await fetch('/api/async/train', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}), data = await response.json();
      if (!response.ok) throw new Error(apiError(data, 'could not start survival training'));
      pollAsyncJob(data.job_id);
    } catch (error) { setAsyncBusy(false); ui.asyncStatus.textContent = `Could not start: ${error.message}`; }
  }

  async function startAsyncDiagnostic() {
    const payload = {}; if (ui.asyncSeed.value.trim() !== '') payload.seed = Number(ui.asyncSeed.value);
    setAsyncBusy(true); ui.asyncStatus.textContent = 'Starting the 80-tick smoke test…'; ui.asyncProgress.max = 80; ui.asyncProgress.value = 0;
    try {
      const response = await fetch('/api/async/diagnostic', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}), data = await response.json();
      if (!response.ok) throw new Error(apiError(data, 'could not start smoke test'));
      pollAsyncJob(data.job_id);
    } catch (error) { setAsyncBusy(false); ui.asyncStatus.textContent = `Could not start: ${error.message}`; }
  }

  function updateAsyncEstimate() {
    const lives = Number(ui.asyncCandidateBudget.value || 0), replicas = Number(ui.asyncReplicasInput.value || 0), batch = Number(ui.asyncBatchInput.value || 1);
    ui.asyncEstimate.value = `${lives} candidate lives × ${replicas} replicas = about ${lives * replicas} completed trajectories; at most about ${Math.floor(lives / batch)} CMA result batches.`;
  }

  document.querySelectorAll('.nav').forEach((button) => button.addEventListener('click', () => activate(button.dataset.view)));
  ui.run.addEventListener('change', () => { state.runName = ui.run.value; state.frame = state.batch = state.coordinate = 0; state.selected = null; refreshSelectors(); drawGraph(); update(); updateOverview(); }); ui.batch.addEventListener('change', () => { state.batch = Number(ui.batch.value); update(); }); ui.coordinate.addEventListener('change', () => { state.coordinate = Number(ui.coordinate.value); update(); }); ui.slider.addEventListener('input', () => { state.frame = Number(ui.slider.value); update(); }); ui.prev.addEventListener('click', () => move(-1)); ui.next.addEventListener('click', () => move(1)); ui.play.addEventListener('click', () => { state.playing = !state.playing; ui.play.textContent = state.playing ? 'Pause' : 'Play'; state.lastTick = performance.now(); if (state.playing) requestAnimationFrame(animate); });
  ui.form.addEventListener('submit', async (event) => { event.preventDefault(); const values = Object.fromEntries(new FormData(ui.form)); ['seed','nodes','steps','batch_size'].forEach((key) => { values[key] = Number(values[key]); }); ['mean_degree','dt'].forEach((key) => { values[key] = Number(values[key]); }); ui.runStatus.textContent = 'Running deterministic local experiment…'; try { const response = await fetch('/api/experiment', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(values)}), document = await response.json(); if (!response.ok) throw new Error(apiError(document, 'experiment request failed')); load(document); ui.runStatus.textContent = 'Reference simulation loaded. Open Replay for details.'; } catch (error) { ui.runStatus.textContent = `Could not start: ${error.message}`; } });
  ui.random.addEventListener('click', () => startJob('/api/evolution/random-search')); ui.cma.addEventListener('click', () => startJob('/api/evolution/search')); ui.file.addEventListener('change', async () => { const file = ui.file.files[0]; if (!file) return; try { load(JSON.parse(await file.text())); } catch (error) { ui.status.textContent = error.message; } });
  ui.asyncForm.addEventListener('submit', startAsync); ui.asyncDiagnostic.addEventListener('click', startAsyncDiagnostic); ui.asyncRefresh.addEventListener('click', loadLatestAsync); [ui.asyncCandidateBudget, ui.asyncReplicasInput, ui.asyncBatchInput].forEach((input) => input.addEventListener('input', updateAsyncEstimate)); updateAsyncEstimate(); activate('survival'); loadLatestAsync();
  ui.liveRefresh.addEventListener('click', refreshLiveModels); ui.liveModel.addEventListener('change', renderLiveModelDetail); ui.liveModelScope.addEventListener('change', () => { const visible = renderLiveModelOptions(); ui.liveStatus.textContent = `${visible} model${visible === 1 ? '' : 's'} shown by the current filter.`; }); ui.liveForm.addEventListener('submit', launchLive); ui.livePlay.addEventListener('click', () => { if (!state.live) return; state.playing = !state.playing; ui.livePlay.textContent = state.playing ? 'Pause' : 'Play'; state.lastTick = performance.now(); if (state.playing) requestAnimationFrame(animate); }); ui.liveStep.addEventListener('click', () => advanceLive()); refreshLiveModels();
})();
