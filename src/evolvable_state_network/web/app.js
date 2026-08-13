(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const ui = {
    run: $('run-select'), batch: $('batch-select'), coordinate: $('coordinate-select'), rate: $('rate-select'), slider: $('frame-slider'), label: $('frame-label'), play: $('play'), prev: $('previous'), next: $('next'), loop: $('loop'), svg: $('network'), status: $('load-status'), file: $('file-input'), frameInfo: $('frame-info'), selection: $('selection-info'), metrics: $('metrics'), events: $('events'), form: $('experiment-form'), runStatus: $('run-status'), overview: $('overview-cards'),
    liveForm: $('live-form'), liveModel: $('live-model-select'), liveModelDetail: $('live-model-detail'), liveRefresh: $('refresh-live-models'), liveStatus: $('live-status'), livePlay: $('live-play'), liveStep: $('live-step'), liveRate: $('live-rate'), liveLabel: $('live-frame-label'), workspace: $('shared-workspace'), replayHost: $('replay-workspace-host'), liveHost: $('live-workspace-host'),
    asyncForm: $('async-form'), asyncSeed: $('async-seed'), asyncRun: $('async-run'), asyncDiagnostic: $('async-diagnostic'), asyncRefresh: $('async-refresh'), asyncStatus: $('async-status'), asyncProgress: $('async-progress'), asyncProgressLabel: $('async-progress-label'), asyncMetrics: $('async-metrics'), asyncValidation: $('async-validation'), asyncSlots: $('async-slots'), asyncCauses: $('async-causes'), asyncCurriculum: $('async-curriculum'), asyncCurriculumCopy: $('async-curriculum-copy'), asyncCandidates: $('async-candidates'), asyncDetail: $('async-detail'), asyncArtifacts: $('async-artifacts'), asyncLearningState: $('async-learning-state'), asyncLearningCopy: $('async-learning-copy'), asyncRunFacts: $('async-run-facts'), asyncEstimate: $('async-work-estimate'), asyncCandidateBudget: $('async-candidates-budget'), asyncSlotsInput: $('async-slots-input'), asyncReplicasInput: $('async-replicas-input'), asyncStablePopulation: $('async-stable-population'), asyncBatchInput: $('async-batch-input'), asyncStateWidth: $('async-state-width'), asyncInitialStateScale: $('async-initial-state-scale'), asyncTicksInput: $('async-ticks-input'),
    embodiedForm: $('embodied-form'), embodiedModel: $('embodied-model'), embodiedContinueRun: $('embodied-continue-run'), embodiedRefresh: $('embodied-refresh'), embodiedAlgorithm: $('embodied-algorithm'), embodiedTrainingMode: $('embodied-training-mode'), embodiedExecutionBackend: $('embodied-execution-backend'), embodiedDevice: $('embodied-device'), embodiedWorkers: $('embodied-workers'), embodiedPopulation: $('embodied-population'), embodiedTicks: $('embodied-ticks'), embodiedBatchGenerations: $('embodied-batch-generations'), embodiedBatchSteps: $('embodied-batch-steps'), embodiedHorizonSuggestion: $('embodied-horizon-suggestion'), embodiedBatchTrials: $('embodied-batch-trials'), embodiedBatchValidationTrials: $('embodied-batch-validation-trials'), embodiedBatchTestTrials: $('embodied-batch-test-trials'), embodiedBatchOpponents: $('embodied-batch-opponents'), embodiedPreyCount: $('embodied-prey-count'), embodiedPredatorCount: $('embodied-predator-count'), embodiedMaxFood: $('embodied-max-food'), embodiedFoodGrowthRate: $('embodied-food-growth-rate'), embodiedMaxSpeed: $('embodied-max-speed'), embodiedMaxTurn: $('embodied-max-turn'), embodiedPlantClusters: $('embodied-plant-clusters'), embodiedPlantClusterRadius: $('embodied-plant-cluster-radius'), embodiedNodes: $('embodied-nodes'), embodiedStateWidth: $('embodied-state-width'), embodiedDegree: $('embodied-degree'), embodiedStateScale: $('embodied-state-scale'), embodiedNetworkDt: $('embodied-network-dt'), embodiedMaxDelta: $('embodied-max-delta'), embodiedEdgeStepScale: $('embodied-edge-step-scale'), embodiedEnergyScale: $('embodied-energy-scale'), embodiedSurvivalPressure: $('embodied-survival-pressure'), embodiedSeed: $('embodied-seed'), embodiedRun: $('embodied-run'), embodiedTerminate: $('embodied-terminate'), embodiedStatus: $('embodied-status'), embodiedProgress: $('embodied-progress'), embodiedResult: $('embodied-result'), embodiedDeathsChart: $('embodied-deaths-chart'), embodiedMealsChart: $('embodied-meals-chart'), embodiedChartOneLabel: $('embodied-chart-one-label'), embodiedChartTwoLabel: $('embodied-chart-two-label'),
    demoForm: $('demo-form'), demoRun: $('demo-run'), demoRefresh: $('demo-refresh'), demoSeed: $('demo-seed'), demoPreyCount: $('demo-prey-count'), demoPredatorCount: $('demo-predator-count'), demoInitialFood: $('demo-initial-food'), demoMaxFood: $('demo-max-food'), demoFoodGrowthRate: $('demo-food-growth-rate'), demoRate: $('demo-rate'), demoStart: $('demo-start'), demoStatus: $('demo-status'), demoPlay: $('demo-play'), demoStep: $('demo-step'), demoRecord: $('demo-record'), demoLabel: $('demo-label'), demoCanvas: $('ecology-canvas'), demoEvents: $('demo-events'), demoNetwork: $('demo-network'), demoColorChannel: $('demo-color-channel'), demoChannel: $('demo-channel'), demoNodeChart: $('demo-node-chart'), demoIndividualTitle: $('demo-individual-title'), demoIndividualSummary: $('demo-individual-summary')
  };
  const state = { data: null, runName: '', frame: 0, batch: 0, coordinate: 0, selected: null, playing: false, lastTick: 0, layout: [], job: null, jobTimer: null, live: null, liveModels: [], demo: null, demoLastTick: 0, demoIndividual: null, demoHistory: [], demoRecorder: null, demoRecordingChunks: [], embodiedCheckpointVersion: '', embodiedJobId: '', embodiedModelWidths: {}, embodiedRunWidths: {} };
  const NS = 'http://www.w3.org/2000/svg';
  const element = (tag, attrs = {}) => { const node = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value)); return node; };
  const number = (value) => Number.isFinite(value) ? Number(value).toFixed(5) : '—';
  const apiError = (data, fallback) => Array.isArray(data?.detail)
    ? data.detail.map((item) => item.msg).join('; ')
    : (data?.detail || data?.error || fallback);
  const viewTitles = {
    survival: 'Survival training',
    embodied: 'Embodied learning',
    'ecology-demo': 'Ecology demo',
    overview: 'Reference simulation',
    replay: 'Trajectory replay',
    live: 'Live graph',
    evolution: 'Evolution search',
  };
  const preferenceForms = ['async-form', 'embodied-form', 'demo-form', 'experiment-form', 'live-form'];

  function restorePreferences() {
    preferenceForms.forEach((formId) => {
      const form = $(formId);
      if (!form) return;
      [...form.elements].forEach((control) => {
        if (!control.id && !control.name) return;
        try {
          const saved = localStorage.getItem(`state-network-lab:${formId}:${control.id || control.name}`);
          if (saved === null) return;
          if (control.type === 'checkbox') control.checked = saved === 'true';
          else control.value = saved;
        } catch (_) { /* Storage may be disabled; defaults remain usable. */ }
      });
      form.addEventListener('change', ({target}) => {
        if (!target.matches('input, select') || (!target.id && !target.name)) return;
        try {
          localStorage.setItem(
            `state-network-lab:${formId}:${target.id || target.name}`,
            target.type === 'checkbox' ? String(target.checked) : target.value,
          );
        } catch (_) { /* Preferences are an optional enhancement. */ }
      });
    });
  }

  function activate(view, updateUrl = true) {
    if (!document.querySelector(`.view[data-view="${view}"]`)) view = 'survival';
    if (view === 'live') ui.liveHost.append(ui.workspace); else ui.replayHost.append(ui.workspace);
    document.body.dataset.view = view;
    document.querySelectorAll('.view').forEach((item) => {
      const active = item.dataset.view === view;
      item.classList.toggle('active', active);
      item.setAttribute('aria-hidden', String(!active));
    });
    document.querySelectorAll('.nav').forEach((item) => {
      const active = item.dataset.view === view;
      item.classList.toggle('active', active);
      item.setAttribute('aria-current', active ? 'page' : 'false');
    });
    document.title = `${viewTitles[view] || 'Workspace'} · State Network Lab`;
    if (updateUrl && location.hash !== `#${view}`) history.pushState({view}, '', `#${view}`);
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
    const functional = model.functional ? 'functional' : 'not functional';
    return `#${model.global_rank} · run ${model.run_id.slice(0, 8)} · elite ${model.elite_rank} · stage ${model.stage} · ${model.lifetime} ticks · ${functional} · burden ${Number(model.worst_pathology_burden).toFixed(3)}`;
  }

  function renderLiveModelDetail() {
    const model = state.liveModels.find((item) => item.id === ui.liveModel.value);
    if (!model) { ui.liveModelDetail.innerHTML = '<p>No model matches this filter.</p>'; return; }
    const intro = document.createElement('div'), title = document.createElement('h4'), copy = document.createElement('p');
    title.textContent = `Overall survival rank #${model.global_rank}`;
    copy.textContent = `Candidate ${model.candidate_id} is elite ${model.elite_rank} within run ${model.run_id.slice(0, 8)}. Ranking is lexicographic—not an invented scalar score.`;
    intro.append(title, copy);
    if (model.global_rank === 1) { const badge = document.createElement('span'); badge.className = 'recommended'; badge.textContent = 'RECOMMENDED: strongest discovered survival rank'; intro.prepend(badge); }
    const facts = [
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
    ];
    const list = document.createElement('dl'); facts.forEach(([key, value]) => { const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = key; dd.textContent = value; list.append(dt, dd); });
    ui.liveModelDetail.replaceChildren(intro, list);
  }

  function renderLiveModelOptions() {
    const prior = ui.liveModel.value;
    const visible = state.liveModels;
    ui.liveModel.replaceChildren(...visible.map((model) => new Option(liveModelLabel(model), model.id)));
    if (visible.some((model) => model.id === prior)) ui.liveModel.value = prior;
    renderLiveModelDetail();
    return visible.length;
  }

  async function refreshLiveModels() {
    ui.liveStatus.textContent = 'Finding final-stage survival elites…';
    try {
      const response = await fetch('/api/live/models'), data = await response.json();
      if (!response.ok) throw new Error(apiError(data, 'model list unavailable'));
      state.liveModels = data.models; const visible = renderLiveModelOptions();
      const latest = data.latest_survival;
      if (latest?.available && Number(latest.report?.graduations || 0) === 0) {
        const causes = (latest.candidates || []).reduce((counts, candidate) => { const cause = causeLabel(candidate.death_cause); counts[cause] = (counts[cause] || 0) + 1; return counts; }, {});
        const leading = Object.entries(causes).sort((left, right) => right[1] - left[1]).slice(0, 2).map(([cause, count]) => `${count} ${cause}`).join(', ');
        ui.liveStatus.textContent = `Latest run ${latest.run_id.slice(0, 8)} produced no Live model: ${latest.report.completed_candidates || 0} candidates died before graduation${leading ? ` (${leading})` : ''}. ${data.models.length ? `Showing ${visible} older eligible model${visible === 1 ? '' : 's'}.` : 'Train again after adjusting the survival guards.'}`;
      } else {
        ui.liveStatus.textContent = data.models.length ? `${visible} final-stage survival elite${visible === 1 ? '' : 's'} available.` : (latest?.available && Number(latest.report?.graduations || 0) ? `Latest run has ${latest.report.graduations} interim graduations, but no final-stage, functional, pathology-free Live model yet.` : 'No usable model exists yet. Complete final-stage functional survival validation.');
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
    [['seed','live-seed'], ['nodes','live-nodes'], ['batch_size','live-batch']].forEach(([key, id]) => { fields[key] = Number($(id).value); });
    [['mean_degree','live-degree'], ['initial_state_scale','live-initial-state-scale'], ['dt','live-dt']].forEach(([key, id]) => { fields[key] = Number($(id).value); });
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
  const sourceLabel = (source) => ({cma:'CMA-ES proposal', initial:'reference genome'}[source] || source);

  function renderLearningVerdict(report = {}, settings = {}, runKind = 'training') {
    const updates = Number(report.optimizer_updates || 0), completed = Number(report.completed_candidates || 0), cohort = report.optimizer_batch_progress || {};
    const cohortDetail = cohort.batch_size ? ` Cohort: ${cohort.completed || 0}/${cohort.batch_size} finished, ${cohort.inflight || 0} still evaluating.` : '';
    let stateLabel = 'No optimizer update yet', copy = `${completed} candidate lives have finished, but CMA-ES has not received a complete comparable result batch.${cohortDetail}`;
    if (updates > 0 && updates < 5) { stateLabel = 'Warm-up: very early learning'; copy = `CMA-ES updated ${updates} time${updates === 1 ? '' : 's'}. This proves the loop is learning, but it is too early to infer convergence.`; }
    else if (updates >= 5 && updates < 20) { stateLabel = 'Training is underway'; copy = `CMA-ES has made ${updates} updates from completed survival evidence. Compare passage rates and elite changes before treating the result as stable.`; }
    else if (updates >= 20) { stateLabel = 'Substantial optimization history'; copy = `CMA-ES has made ${updates} updates. This is enough history to inspect trends, though held-out survival validation is still required.`; }
    if (runKind === 'diagnostic') copy += ' This run is an 80-tick smoke test, not a training budget.';
    ui.asyncLearningState.textContent = stateLabel;
    ui.asyncLearningCopy.textContent = copy;
    const stop = ({final_stage_population_established:'final-stage stable population established', stage_not_passed_tick_limit:'safety tick limit reached before this stage became stable', running:'still evolving'}[report.stop_reason] || report.stop_reason || 'older run completed');
    const origins = Object.entries(report.proposals_by_source || {}).map(([source, count]) => `${count} ${sourceLabel(source)}`).join(', ') || 'not recorded';
    const tickLimit = report.tick_limit ?? settings.max_ticks;
    const facts = [
      ['stop condition', stop],
      ['ticks elapsed', tickLimit == null ? `${report.ticks_elapsed ?? '—'} (no cap)` : `${report.ticks_elapsed ?? '—'} / ${tickLimit}`],
      ['candidate evidence', `${completed}${report.candidate_budget ? ` / ${report.candidate_budget} checkpoint` : ''}`],
      ['replica lives', `${report.completed_replica_lives ?? completed * Number(settings.replicas || 0)} completed`],
      ['stable survivors', `${report.stable_survivor_count ?? 0} / ${report.stable_population_size ?? settings.stable_population_size ?? '?'}`],
      ['functional survivor archive', `${report.survivor_archive_size ?? '—'}`],
      ['CMA cohort', cohort.batch_size ? `${cohort.completed || 0}/${cohort.batch_size} complete, ${cohort.inflight || 0} active` : 'not recorded'],
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

  function renderAsyncValidation(validation) {
    if (!validation) {
      ui.asyncValidation.classList.remove('active');
      ui.asyncValidation.replaceChildren(Object.assign(document.createElement('div'), {innerHTML:'<p class="eyebrow">FINAL DEPLOYMENT VALIDATION</p><h3>Waiting for a final-stage graduation</h3>'}), Object.assign(document.createElement('p'), {textContent:'A final-stage candidate is tested first with no injected disturbance, then on fresh perturbed graphs. Progress appears here while those held-out lives run.'}));
      return;
    }
    const phase = validation.phase === 'autonomous' ? 'Autonomous stability' : 'Perturbed recovery';
    const fraction = Math.min(1, Number(validation.step || 0) / Math.max(1, Number(validation.steps || 1)));
    const heading = document.createElement('div');
    heading.innerHTML = `<p class="eyebrow">FINAL DEPLOYMENT VALIDATION</p><h3>${phase}</h3>`;
    const detail = document.createElement('div'); detail.className = 'validation-detail';
    const copy = document.createElement('span'); copy.textContent = `Candidate ${validation.candidate_id} · held-out replica ${validation.replica} / ${validation.replicas} · ${validation.step} / ${validation.steps} ticks`;
    const progress = document.createElement('progress'); progress.value = fraction; progress.max = 1;
    detail.append(copy, progress);
    ui.asyncValidation.classList.add('active');
    ui.asyncValidation.replaceChildren(heading, detail);
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
    ui.asyncCurriculumCopy.textContent = `Current stage ${(report.curriculum_level ?? 0) + 1}; it continues until ${report.stable_population_size ?? settings.stable_population_size ?? '?'} healthy survivors establish a stable population.`;
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
      const open = document.createElement('button'); open.type = 'button'; open.className = 'replay-survival'; open.textContent = 'Debug: reconstruct this life';
      open.addEventListener('click', async () => {
        open.disabled = true; open.textContent = 'Reconstructing...';
        try {
          const response = await fetch(replica.debug_replay_url), document = await response.json();
          if (!response.ok) throw new Error(apiError(document, 'survival replay is unavailable'));
          load(document); activate('replay'); window.scrollTo({top: 0, behavior: 'smooth'});
        } catch (error) {
          evidence.textContent = `Could not reconstruct this survival replay: ${error.message}`;
        } finally { open.disabled = false; open.textContent = 'Debug: reconstruct this life'; }
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
    renderAsyncValidation(data.validation);
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
      renderAsyncData(data); ui.asyncProgress.max = data.report.tick_limit || Math.max(1, data.report.ticks_elapsed || 1); ui.asyncProgress.value = data.report.ticks_elapsed || 0;
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
        const report = snapshot.report || {}, checkpoint = Number(report.candidate_budget || 0), completed = Number(report.completed_candidates || 0);
        const tickLimit = snapshot.max_ticks ?? report.tick_limit;
        ui.asyncProgress.max = Number(tickLimit || 1); ui.asyncProgress.value = tickLimit ? tick : 0;
        ui.asyncProgressLabel.textContent = tickLimit == null
          ? `${tick} ticks · no cap · ${completed}${checkpoint ? ` / ${checkpoint} evidence checkpoint` : ''}`
          : `${tick} / ${tickLimit} ticks · ${completed}${checkpoint ? ` / ${checkpoint} evidence checkpoint` : ''}`;
        const validation = snapshot.validation;
        if (validation) {
          const phase = validation.phase === 'autonomous' ? 'autonomous stability' : 'perturbed recovery';
          ui.asyncProgress.max = Math.max(1, Number(validation.steps || 1)); ui.asyncProgress.value = Number(validation.step || 0);
          ui.asyncProgressLabel.textContent = `final validation · ${phase} · replica ${validation.replica}/${validation.replicas} · ${validation.step}/${validation.steps}`;
          ui.asyncStatus.textContent = `Final held-out validation is running · candidate ${validation.candidate_id}`;
        } else {
          ui.asyncStatus.textContent = `${job.kind === 'async_training' ? 'Survival training' : 'Smoke test'} is running · seed ${job.seed}`;
        }
        if (snapshot.report) renderAsyncData({report:snapshot.report, validation}, snapshot.slots);
        window.setTimeout(() => pollAsyncJob(jobId), 300);
      } else if (job.status === 'complete') {
        setAsyncBusy(false); renderAsyncData(job.result); const report = job.result.report || {};
        ui.asyncProgress.max = report.tick_limit || Math.max(1, report.ticks_elapsed || 1); ui.asyncProgress.value = report.ticks_elapsed || 0;
        ui.asyncProgressLabel.textContent = report.stop_reason === 'final_stage_population_established' ? 'stable population established' : 'explicit tick cap reached before stage completion'; ui.asyncStatus.textContent = `${job.kind === 'async_training' ? 'Training stopped' : 'Smoke test complete'} · ${report.completed_candidates} candidate lives · seed ${job.seed}`;
      } else { throw new Error(job.error || 'survival run failed'); }
    } catch (error) { setAsyncBusy(false); ui.asyncStatus.textContent = `Survival run stopped: ${error.message}`; }
  }

  async function startAsync(event) {
    event.preventDefault();
    const fields = {
      candidate_budget:'async-candidates-budget', slots:'async-slots-input', replicas:'async-replicas-input', stable_population_size:'async-stable-population', optimizer_batch:'async-batch-input', state_width:'async-state-width', initial_state_scale:'async-initial-state-scale',
      stage_1_lifetime:'async-stage1-life', stage_2_lifetime:'async-stage2-life', stage_1_nodes:'async-stage1-nodes', stage_2_nodes:'async-stage2-nodes', mean_degree:'async-degree',
      disturbance_interval:'async-disturbance-interval', disturbance_strength:'async-disturbance-strength', fatal_threshold:'async-fatal-threshold', node_growth_alert:'async-node-growth-alert', one_direction_steps:'async-one-direction-steps', probe_interval:'async-probe-interval',
    };
    const payload = Object.fromEntries(Object.entries(fields).map(([key, id]) => [key, Number($(id).value)]));
    if (ui.asyncTicksInput.value.trim() !== '') payload.max_ticks = Number(ui.asyncTicksInput.value);
    if (ui.asyncSeed.value.trim() !== '') payload.seed = Number(ui.asyncSeed.value);
    setAsyncBusy(true); ui.asyncStatus.textContent = 'Starting configured survival training…'; ui.asyncProgress.max = payload.max_ticks || 1; ui.asyncProgress.value = 0;
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
    const checkpoint = Number(ui.asyncCandidateBudget.value || 0), replicas = Number(ui.asyncReplicasInput.value || 0), survivors = Number(ui.asyncStablePopulation.value || 0);
    ui.asyncEstimate.value = `Evidence checkpoint: ${checkpoint} completed lives across ${replicas} replicas. The stage advances only after ${survivors} healthy survivors; no trajectory is stored by default.`;
  }

  async function refreshEmbodiedModels() {
    const prior = ui.embodiedModel.value, priorRun = ui.embodiedContinueRun.value; ui.embodiedModel.replaceChildren(new Option('Fresh joint rules', '')); ui.embodiedContinueRun.replaceChildren(new Option('Do not continue a prior run', ''));
    state.embodiedModelWidths = {}; state.embodiedRunWidths = {};
    try {
      const [response, runsResponse] = await Promise.all([fetch('/api/live/models'), fetch('/api/embodied/runs')]), [data, runs] = await Promise.all([response.json(), runsResponse.json()]);
      if (!response.ok) throw new Error(apiError(data, 'basic-model list unavailable')); if (!runsResponse.ok) throw new Error(apiError(runs, 'embodied-run list unavailable'));
      data.models.filter((model) => model.target === 'joint').forEach((model) => { const width = Number(model.node_state_width); if (Number.isInteger(width)) state.embodiedModelWidths[model.id] = width; ui.embodiedModel.append(new Option(`#${model.global_rank} · stage ${model.stage} · ${model.lifetime} ticks${Number.isInteger(width) ? ` · ${width} ch` : ''}`, model.id)); });
      ui.embodiedModel.value = [...ui.embodiedModel.options].some((option) => option.value === prior) ? prior : '';
      runs.runs.forEach((run) => { const unit = run.training_mode === 'batch' ? 'gen' : 'tick', status = run.complete ? 'complete' : `checkpoint ${run.checkpoint_tick}/${run.ticks} ${unit}`, algorithm = run.algorithm === 'genetic' ? 'GA' : 'CMA-ES', lifetime = String(run.objective || '').includes('lifetime'), score = lifetime ? `prey ${number(run.prey_best_lifetime)} ticks · predator ${number(run.predator_best_lifetime)} ticks` : `legacy shaped score ${number(run.prey_best_lifetime)}`, width = Number(run.state_width); if (Number.isInteger(width)) state.embodiedRunWidths[run.id] = width; ui.embodiedContinueRun.append(new Option(`${run.id.slice(0, 8)} · ${run.training_mode || 'continuous'} · ${algorithm} · ${status} · ${score}${Number.isInteger(width) ? ` · ${width} ch` : ''}`, run.id)); });
      ui.embodiedContinueRun.value = [...ui.embodiedContinueRun.options].some((option) => option.value === priorRun) ? priorRun : '';
      const selectedWidth = state.embodiedModelWidths[ui.embodiedModel.value] || state.embodiedRunWidths[ui.embodiedContinueRun.value];
      if (selectedWidth) ui.embodiedStateWidth.value = String(selectedWidth);
    } catch (error) { ui.embodiedStatus.textContent = `Could not load basic rules: ${error.message}`; }
  }
  function drawEmbodiedTelemetry(canvas, telemetry, series) {
    const context = canvas.getContext('2d'), width = canvas.width, height = canvas.height, left = 48, right = width - 18, top = 28, bottom = height - 32;
    context.clearRect(0, 0, width, height); context.fillStyle = '#0d131d'; context.fillRect(0, 0, width, height);
    if (!telemetry.length) { context.fillStyle = '#9caac2'; context.font = '13px system-ui'; context.fillText('Waiting for continuous-world telemetry…', 22, 42); return; }
    const values = telemetry.flatMap((point) => series.map((item) => Number(point[item.key])).filter(Number.isFinite));
    if (!values.length) { context.fillStyle = '#9caac2'; context.font = '13px system-ui'; context.fillText('No meal events yet.', 22, 42); return; }
    const maximum = Math.max(1, ...values), minimum = Math.min(0, ...values), range = Math.max(.001, maximum - minimum);
    const x = (index) => left + index / Math.max(1, telemetry.length - 1) * (right - left), y = (value) => bottom - (value - minimum) / range * (bottom - top);
    context.font = '11px system-ui'; context.strokeStyle = '#29394f'; context.fillStyle = '#9caac2'; context.lineWidth = 1;
    for (let tick = 0; tick <= 4; tick += 1) { const value = minimum + range * tick / 4, py = y(value); context.beginPath(); context.moveTo(left, py); context.lineTo(right, py); context.stroke(); context.fillText(value.toFixed(2), 3, py + 3); }
    context.fillText(String(telemetry[0].tick ?? telemetry[0].generation), left, height - 12); context.fillText(String(telemetry[telemetry.length - 1].tick ?? telemetry[telemetry.length - 1].generation), right - 22, height - 12);
    series.forEach((item, index) => { context.strokeStyle = item.color; context.lineWidth = 2; context.beginPath(); let drawing = false; telemetry.forEach((point, pointIndex) => { const value = Number(point[item.key]); if (!Number.isFinite(value)) { drawing = false; return; } const px = x(pointIndex), py = y(value); if (drawing) context.lineTo(px, py); else { context.moveTo(px, py); drawing = true; } }); context.stroke(); context.fillStyle = item.color; context.fillText(item.label, left + index * 132, 15); });
  }
  function renderEmbodiedJob(job) {
    const latest = job.latest || {}, population = latest.population || {}, mode = latest.training_mode || job.result?.training_mode || 'continuous', progress = mode === 'batch' ? (latest.generation || 0) : (latest.tick || 0), total = mode === 'batch' ? (latest.generations || job.samples_total || 0) : (latest.ticks || job.samples_total || 0), checkpoint = latest.checkpoint_url ? ` · checkpoint saved at ${progress}` : '', algorithm = (latest.algorithm || latest.prey?.algorithm || job.result?.algorithm) === 'genetic' ? 'GA' : 'CMA-ES', running = mode === 'batch' ? `Generation ${progress} / ${total} · ${algorithm} · episode evaluations ${latest.prey?.evaluations ?? 0}/${latest.predator?.evaluations ?? 0}` : `World tick ${progress} / ${total} · ${algorithm} · live prey ${population.prey ?? '—'} · predators ${population.predator ?? '—'} · optimizer updates ${latest.prey?.updates ?? 0}/${latest.predator?.updates ?? 0}`, phase = job.status === 'failed' ? `Failed: ${job.error}` : job.status === 'complete' ? `Complete · ${mode} · ${algorithm}` : `${running}${checkpoint}`;
    ui.embodiedProgress.textContent = phase;
    const execution = job.result?.execution || latest;
    if (job.status !== 'failed') ui.embodiedProgress.textContent += ` · ${execution.execution_backend || execution.backend || 'python'}/${execution.device || 'cpu'}${mode === 'batch' ? ` · ${execution.workers || 1} worker${Number(execution.workers || 1) === 1 ? '' : 's'}` : ''}`;
    if (job.status === 'terminated') ui.embodiedProgress.textContent = `Terminated at ${mode === 'batch' ? 'generation' : 'tick'} ${progress} / ${total}${checkpoint}`;
    if (mode === 'batch') {
      const history = latest.history || job.result?.history || [];
      ui.embodiedChartOneLabel.textContent = 'Restricted mean lifetime on training and selection-validation lives'; ui.embodiedChartTwoLabel.textContent = 'Fraction of first lives surviving the evaluation horizon';
      drawEmbodiedTelemetry(ui.embodiedDeathsChart, history, [{key:'prey_best_lifetime', label:'prey train', color:'#63d5c2'}, {key:'prey_validation_lifetime', label:'prey held-out', color:'#d1a9ff'}, {key:'predator_best_lifetime', label:'predator train', color:'#f39b72'}, {key:'predator_validation_lifetime', label:'predator held-out', color:'#ffd166'}]);
      drawEmbodiedTelemetry(ui.embodiedMealsChart, history, [{key:'prey_horizon_survival_rate', label:'prey survival', color:'#63d5c2'}, {key:'predator_horizon_survival_rate', label:'predator survival', color:'#f39b72'}]);
    } else {
      const telemetry = latest.telemetry || job.result?.telemetry || [];
      ui.embodiedChartOneLabel.textContent = 'Completed lifetime by world tick'; ui.embodiedChartTwoLabel.textContent = 'Cumulative deaths by world tick';
      drawEmbodiedTelemetry(ui.embodiedDeathsChart, telemetry, [{key:'prey_mean_lifetime', label:'prey lifetime', color:'#63d5c2'}, {key:'predator_mean_lifetime', label:'predator lifetime', color:'#f39b72'}]);
      drawEmbodiedTelemetry(ui.embodiedMealsChart, telemetry, [{key:'prey_deaths', label:'prey deaths', color:'#63d5c2'}, {key:'predator_deaths', label:'predator deaths', color:'#f39b72'}]);
    }
    if (job.result) {
      const result = job.result, source = result.initialization?.kind === 'embodied_run' ? `embodied run ${result.initialization.run_id.slice(0, 8)}` : result.initialization?.kind === 'basic_model' ? 'one basic Survival rule for both species' : 'fresh joint rules';
      const batch = result.training_mode === 'batch';
      const behavior = result.prey.behavior || {}, testBehavior = result.prey.test_behavior || behavior, baselines = result.prey.baselines || {}, ecology = result.ecology || {};
      if (batch) ui.embodiedResult.replaceChildren(statCard('Prey untouched test lifetime', `${number(result.prey.test_lifetime ?? result.prey.best_lifetime)} ticks`), statCard('Predator untouched test lifetime', `${number(result.predator.test_lifetime ?? result.predator.best_lifetime)} ticks`), statCard('Selection-validation lifetime', `${number(result.prey.selection_validation_lifetime ?? result.prey.best_lifetime)} ticks`), statCard('Neutral-rule test lifetime', `${number(baselines.zero_rule_lifetime)} ticks`), statCard('Lifetime gain over neutral rule', `${number(baselines.lifetime_gain_over_zero_rule)} ticks`), statCard('First lives surviving horizon', `${(100 * Number(testBehavior.horizon_survival_rate || 0)).toFixed(1)}%`), statCard('Mean completed lifetime', `${number(testBehavior.mean_completed_lifetime)} ticks`), statCard('Test deaths / 1000 steps', number(testBehavior.deaths_per_1000_steps)), statCard('Mean hunger (diagnostic only)', number(testBehavior.mean_hunger)), statCard(`Optimizer updates · prey/predator`, `${result.prey.updates} / ${result.predator.updates}`));
      else ui.embodiedResult.replaceChildren(statCard('Best completed prey lifetime', `${number(result.prey.best_lifetime)} ticks`), statCard('Observed mean prey lifetime', behavior.mean_lifetime == null ? '—' : `${number(behavior.mean_lifetime)} ticks`), statCard('Observed mean hunger (diagnostic only)', number(behavior.mean_hunger)), statCard('Regrowth supply / prey demand', `${number(ecology.prey_energy_supply_ratio)}${ecology.population_sustainable_from_regrowth === false ? ' · insufficient' : ''}`), statCard(`Optimizer updates · ${result.prey.evaluation_replicates || 1} lives/candidate`, `${result.prey.updates} / ${result.predator.updates}`));
      const budgetWarning = !batch && result.ecology?.population_sustainable_from_regrowth === false ? ' Warning: plant regrowth supplies less energy than the prey population consumes even under perfect collection.' : '';
      const visionWarning = batch && Number(baselines.vision_lifetime_delta) <= 0 ? ' Warning: masking ray vision did not reduce final test lifetime.' : '';
      ui.embodiedStatus.textContent = `${batch ? 'Batch episodic' : 'Continuous'} ${algorithm} coevolution complete from ${source}: prey ${result.prey.updates} and predator ${result.predator.updates} optimizer updates.${budgetWarning}${visionWarning}`;
    }
  }
  async function pollEmbodiedJob(jobId) {
    try {
      const response = await fetch(`/api/jobs/${jobId}`), job = await response.json();
      if (!response.ok) throw new Error(apiError(job, 'embodied job unavailable'));
      renderEmbodiedJob(job);
      const checkpointUrl = job.latest?.checkpoint_url || '', checkpointProgress = job.latest?.generation ?? job.latest?.tick ?? '';
      const checkpointVersion = `${checkpointUrl}:${checkpointProgress}`;
      if (checkpointUrl && checkpointVersion !== state.embodiedCheckpointVersion) {
        state.embodiedCheckpointVersion = checkpointVersion;
        refreshDemoRuns();
        refreshEmbodiedModels();
      }
      if (job.status === 'running') window.setTimeout(() => pollEmbodiedJob(jobId), 500); else { state.embodiedJobId = ''; ui.embodiedRun.disabled = false; ui.embodiedTerminate.disabled = true; if (job.status === 'terminated') ui.embodiedStatus.textContent = 'Embodied evolution terminated. Its latest checkpoint remains available for a demo or a new continued run.'; refreshDemoRuns(); refreshEmbodiedModels(); }
    } catch (error) { ui.embodiedStatus.textContent = `Embodied run status unavailable: ${error.message}`; ui.embodiedRun.disabled = false; ui.embodiedTerminate.disabled = true; }
  }
  function updateEmbodiedHorizonSuggestion() {
    const scale = Math.max(0, Number(ui.embodiedEnergyScale.value) || 0), minimum = Math.ceil(60 * scale), enforced = ui.embodiedSurvivalPressure.checked;
    ui.embodiedBatchSteps.min = String(enforced ? Math.max(8, minimum) : 8);
    ui.embodiedHorizonSuggestion.textContent = enforced
      ? `Suggested minimum: ${minimum} steps for energy scale ${scale || '—'}.`
      : `Survival-horizon enforcement is disabled; ${minimum} steps would provide the standard three-lifetime window.`;
  }
  async function startEmbodied(event) {
    event.preventDefault();
    ui.embodiedTerminate.disabled = true;
    const payload = {training_mode:ui.embodiedTrainingMode.value, algorithm:ui.embodiedAlgorithm.value, execution_backend:ui.embodiedExecutionBackend.value, device:ui.embodiedDevice.value, workers:Number(ui.embodiedWorkers.value), population_size:Number(ui.embodiedPopulation.value), ticks:Number(ui.embodiedTicks.value), batch_generations:Number(ui.embodiedBatchGenerations.value), batch_episode_steps:Number(ui.embodiedBatchSteps.value), batch_trials:Number(ui.embodiedBatchTrials.value), batch_validation_trials:Number(ui.embodiedBatchValidationTrials.value), batch_test_trials:Number(ui.embodiedBatchTestTrials.value), batch_opponents:Number(ui.embodiedBatchOpponents.value), prey_count:Number(ui.embodiedPreyCount.value), predator_count:Number(ui.embodiedPredatorCount.value), max_food:Number(ui.embodiedMaxFood.value), food_growth_rate:Number(ui.embodiedFoodGrowthRate.value), max_speed:Number(ui.embodiedMaxSpeed.value), max_turn:Number(ui.embodiedMaxTurn.value), plant_cluster_count:Number(ui.embodiedPlantClusters.value), plant_cluster_radius:Number(ui.embodiedPlantClusterRadius.value), hidden_nodes:Number(ui.embodiedNodes.value), state_width:Number(ui.embodiedStateWidth.value), mean_degree:Number(ui.embodiedDegree.value), initial_state_scale:Number(ui.embodiedStateScale.value), network_dt:Number(ui.embodiedNetworkDt.value), max_delta:Number(ui.embodiedMaxDelta.value), edge_step_scale:Number(ui.embodiedEdgeStepScale.value), initial_energy_scale:Number(ui.embodiedEnergyScale.value), enforce_survival_pressure:ui.embodiedSurvivalPressure.checked};
    if (ui.embodiedModel.value) payload.model_id = ui.embodiedModel.value;
    if (ui.embodiedContinueRun.value) payload.continue_run_id = ui.embodiedContinueRun.value;
    if (ui.embodiedSeed.value.trim()) payload.seed = Number(ui.embodiedSeed.value);
    ui.embodiedRun.disabled = true; ui.embodiedStatus.textContent = payload.training_mode === 'batch' ? 'Preparing common episode seeds and frozen opponent pools…' : 'Starting the persistent world and assigning its first random networks…';
    try {
      const response = await fetch('/api/embodied/food-web/train', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}), data = await response.json();
      if (!response.ok) throw new Error(apiError(data, 'could not start embodied evolution'));
      state.embodiedJobId = data.job_id; ui.embodiedTerminate.disabled = false; pollEmbodiedJob(data.job_id);
    } catch (error) { ui.embodiedStatus.textContent = `Could not start embodied evolution: ${error.message}`; ui.embodiedRun.disabled = false; ui.embodiedTerminate.disabled = true; }
  }
  async function terminateEmbodied() {
    if (!state.embodiedJobId) return;
    ui.embodiedTerminate.disabled = true; ui.embodiedStatus.textContent = 'Termination requested. The current safe evolution boundary will be completed before stopping.';
    try { const response = await fetch(`/api/embodied/jobs/${state.embodiedJobId}/terminate`, {method:'POST'}), data = await response.json(); if (!response.ok) throw new Error(apiError(data, 'could not terminate embodied evolution')); } catch (error) { ui.embodiedStatus.textContent = `Could not terminate embodied evolution: ${error.message}`; ui.embodiedTerminate.disabled = false; }
  }

  async function refreshDemoRuns() {
    const prior = ui.demoRun.value; ui.demoRun.replaceChildren();
    try {
      const response = await fetch('/api/embodied/runs'), data = await response.json();
      if (!response.ok) throw new Error(apiError(data, 'embodied-run list unavailable'));
      if (!data.runs.length) { ui.demoRun.append(new Option('No usable embodied model yet', '')); return; }
      data.runs.forEach((run) => { const progress = run.training_mode === 'batch' ? `${run.checkpoint_tick}/${run.ticks} generations` : `${run.checkpoint_tick}/${run.ticks} ticks`, source = run.complete ? 'complete' : 'current checkpoint', lifetime = String(run.objective || '').includes('lifetime'), score = lifetime ? `prey ${number(run.prey_best_lifetime)} ticks · predator ${number(run.predator_best_lifetime)} ticks` : `legacy shaped score ${number(run.prey_best_lifetime)}`; ui.demoRun.append(new Option(`${run.id.slice(0, 8)} · ${source} · ${progress} · ${score}`, run.id)); });
      ui.demoRun.value = [...ui.demoRun.options].some((option) => option.value === prior) ? prior : ui.demoRun.options[0].value;
    } catch (error) { ui.demoStatus.textContent = `Could not load embodied runs: ${error.message}`; }
  }
  function drawDemoNetwork(network) {
    // Preserve Live graph visual encoding, but arrange boundary nodes as a
    // readable body interface: sensations at left, actions at right, and
    // anonymous recurrent tissue in the centre.
    const svg = ui.demoNetwork, width = 920, height = 560, channel = Number(ui.demoColorChannel.value) || 0, edges = network.edges.map((edge) => ({...edge, weight:1})), values = network.node_state.map((vector) => Number(vector[channel] || 0)), scale = Math.max(.15, ...values.map(Math.abs)), inputs = network.input_nodes, actions = network.action_nodes, hidden = Array.from({length:Number(network.nodes)}, (_, index) => index).filter((index) => !inputs.includes(index) && !actions.includes(index)), positions = Array(Number(network.nodes));
    const spread = (items, x, top, bottom) => items.forEach((node, index) => { positions[node] = {x, y:items.length === 1 ? (top + bottom) / 2 : top + index * (bottom - top) / (items.length - 1)}; });
    spread(inputs, 130, 62, height - 44); spread(actions, width - 130, 185, height - 185);
    const columns = Math.max(1, Math.ceil(Math.sqrt(hidden.length * 1.25))), rows = Math.max(1, Math.ceil(hidden.length / columns)); hidden.forEach((node, index) => { positions[node] = {x:300 + (index % columns) * 320 / Math.max(1, columns - 1), y:74 + Math.floor(index / columns) * (height - 148) / Math.max(1, rows - 1)}; });
    svg.replaceChildren();
    const defs = element('defs'), marker = element('marker', {id:'demo-arrow', markerWidth:'7', markerHeight:'7', refX:'6', refY:'3.5', orient:'auto'}); marker.append(element('path', {d:'M0,0 L7,3.5 L0,7 z', fill:'#71809b'})); defs.append(marker); svg.append(defs);
    const edgeLayer = element('g'), nodeLayer = element('g'); svg.append(edgeLayer, nodeLayer);
    edges.forEach((edge, index) => { const from = positions[edge.source], to = positions[edge.target], strength = Number(network.edges[index].communication_strength ?? 1), line = element('line', {x1:from.x, y1:from.y, x2:to.x, y2:to.y, 'marker-end':'url(#demo-arrow)', class:'edge'}); line.setAttribute('stroke', strength >= 0 ? '#f39b72' : '#6daafa'); line.setAttribute('stroke-width', String(.7 + Math.min(3.6, Math.abs(strength) * 2.5))); line.setAttribute('opacity', String(.10 + Math.min(.80, Math.abs(strength)))); edgeLayer.append(line); });
    positions.forEach((point, index) => { const group = element('g'), circle = element('circle', {cx:point.x, cy:point.y, r:'12', class:'node', fill:color(values[index], scale)}), label = element('text', {x:point.x, y:point.y + 4, class:'node-label'}); label.textContent = index; group.append(circle, label); nodeLayer.append(group); });
    const inputLabels = demoInputLabels(network), actionLabels = ['turn (−1 left · +1 right)', 'throttle (0 stopped · 1 full)'];
    const heading = (text, x, anchor) => { const item = element('text', {x, y:25, fill:'#63d5c2', 'font-size':'12', 'font-weight':'700', 'letter-spacing':'1.2', 'text-anchor':anchor}); item.textContent = text; svg.append(item); };
    heading('SENSORY INPUTS', 18, 'start'); heading('ACTION OUTPUTS', width - 18, 'end');
    inputs.forEach((node, index) => { const point = positions[node], text = element('text', {x:18, y:point.y + 4, fill:'#cbd6e8', 'font-size':'10', 'text-anchor':'start'}); const channel = network.input_signal_channels?.[index] ?? 0; text.textContent = `${inputLabels[index] || `input ${index}`} · ch ${channel}`; svg.append(text); });
    actions.forEach((node, index) => { const point = positions[node], text = element('text', {x:width - 18, y:point.y + 4, fill:'#cbd6e8', 'font-size':'10', 'text-anchor':'end'}); text.textContent = actionLabels[index] || `action ${index}`; svg.append(text); });
  }
  function demoInputLabels(network) {
    const body = ['hunger', 'energy change', 'ate last tick', 'time since meal'];
    return [...body, ...Array.from({length:network.vision_pixels || Math.max(0, Math.floor((network.input_nodes.length - 4) / 3))}, (_, pixel) => [`ray ${pixel + 1}: plant`, `ray ${pixel + 1}: prey`, `ray ${pixel + 1}: predator`]).flat()];
  }
  function drawDemoNodeChart() {
    const canvas = ui.demoNodeChart, context = canvas.getContext('2d'), width = canvas.width, height = canvas.height, channel = Number(ui.demoChannel.value) || 0, history = state.demoHistory;
    context.clearRect(0, 0, width, height); context.fillStyle = '#101621'; context.fillRect(0, 0, width, height);
    if (!history.length) { context.fillStyle = '#9caac2'; context.font = '14px system-ui'; context.fillText('Select an organism to collect node-state history.', 22, 32); return; }
    const values = history.flatMap((entry) => entry.node_state.map((vector) => Number(vector[channel] || 0))), scale = Math.max(.1, ...values.map((value) => Math.abs(value))) * 1.12, left = 45, right = width - 12, top = 16, bottom = height - 30, x = (index) => left + index / Math.max(1, history.length - 1) * (right - left), y = (value) => (top + bottom) / 2 - value / scale * (bottom - top) / 2;
    context.strokeStyle = '#46566f'; context.lineWidth = 1; context.beginPath(); context.moveTo(left, y(0)); context.lineTo(right, y(0)); context.stroke(); context.fillStyle = '#9caac2'; context.font = '11px ui-monospace'; context.fillText(`+${scale.toFixed(2)}`, 4, top + 4); context.fillText('0', 26, y(0) + 4); context.fillText(`−${scale.toFixed(2)}`, 4, bottom);
    const palette = ['#63d5c2','#f39b72','#6daafa','#d1a9ff','#e6cc72','#f47f96','#84c87b','#b1bfd8'];
    history[history.length - 1].node_state.forEach((_, node) => { context.strokeStyle = palette[node % palette.length]; context.globalAlpha = .85; context.lineWidth = node < 3 ? 1.8 : 1; context.beginPath(); history.forEach((entry, index) => { const px = x(index), py = y(Number(entry.node_state[node]?.[channel] || 0)); if (index) context.lineTo(px, py); else context.moveTo(px, py); }); context.stroke(); }); context.globalAlpha = 1; context.fillStyle = '#cbd6e8'; context.fillText(`channel ${channel} · ${history.length} samples · one line per node`, left, height - 10);
  }
  function renderDemoIndividual(snapshot) {
    const individual = snapshot.individual, network = snapshot.network;
    state.demoIndividual = individual.id;
    if (state.demo) state.demo.lastNetwork = network;
    const options = Array.from({length:network.state_width}, (_, index) => new Option(`channel ${index}`, String(index))), priorColour = ui.demoColorChannel.value, priorHistory = ui.demoChannel.value;
    ui.demoColorChannel.replaceChildren(...options.map((option) => option.cloneNode(true))); ui.demoColorChannel.value = Number(priorColour) < network.state_width ? priorColour : '0'; ui.demoColorChannel.disabled = false;
    ui.demoChannel.replaceChildren(...options); ui.demoChannel.value = Number(priorHistory) < network.state_width ? priorHistory : '0'; ui.demoChannel.disabled = false;
    state.demoHistory.push({tick:snapshot.tick, node_state:network.node_state}); if (state.demoHistory.length > 360) state.demoHistory.shift();
    ui.demoIndividualTitle.textContent = `${individual.species} ${individual.id} · ${network.nodes} nodes · ${network.edges.length} directed edges`;
    ui.demoIndividualSummary.textContent = `Network tick ${network.step}; ray vision occupies channel 0, body/interoceptive signals occupy channel 1, and all other input coordinates are zero. The colour and history controls are independent; history retains ${state.demoHistory.length} observed ecology ticks.`;
    drawDemoNetwork(network); drawDemoNodeChart();
  }
  async function refreshDemoIndividual(individualId = state.demoIndividual) {
    if (!state.demo || !individualId) return;
    try { const response = await fetch(`/api/embodied/sessions/${state.demo.id}/individuals/${encodeURIComponent(individualId)}`), snapshot = await response.json(); if (!response.ok) throw new Error(apiError(snapshot, 'selected organism is unavailable')); renderDemoIndividual(snapshot); } catch (error) { if (individualId === state.demoIndividual) { state.demoIndividual = null; state.demoHistory = []; ui.demoIndividualTitle.textContent = 'Selected organism is no longer alive'; ui.demoIndividualSummary.textContent = 'Choose another organism to inspect its newly sampled network.'; ui.demoColorChannel.disabled = true; ui.demoChannel.disabled = true; ui.demoNetwork.replaceChildren(); drawDemoNodeChart(); } }
  }
  function selectDemoIndividual(snapshot, event) {
    const rect = ui.demoCanvas.getBoundingClientRect(), world = snapshot.state, px = (event.clientX - rect.left) / rect.width * ui.demoCanvas.width, py = (event.clientY - rect.top) / rect.height * ui.demoCanvas.height, x = (value) => value / world.bounds.width * ui.demoCanvas.width, y = (value) => ui.demoCanvas.height - value / world.bounds.height * ui.demoCanvas.height;
    const chosen = world.organisms.map((organism) => ({organism, distance:Math.hypot(px - x(organism.x), py - y(organism.y))})).sort((left, right) => left.distance - right.distance)[0];
    if (!chosen || chosen.distance > 28) return;
    state.demoIndividual = chosen.organism.id; state.demoHistory = []; refreshDemoIndividual(chosen.organism.id);
  }
  function drawDemo(snapshot) {
    const context = ui.demoCanvas.getContext('2d'), width = ui.demoCanvas.width, height = ui.demoCanvas.height, world = snapshot.state, bounds = world.bounds;
    context.clearRect(0, 0, width, height); context.fillStyle = '#101a24'; context.fillRect(0, 0, width, height);
    const x = (value) => value / bounds.width * width, y = (value) => height - value / bounds.height * height;
    world.plants.forEach((plant) => { context.fillStyle = '#74c365'; context.beginPath(); context.arc(x(plant.x), y(plant.y), Math.max(2, plant.radius * 3), 0, 2 * Math.PI); context.fill(); });
    world.organisms.forEach((organism) => { const px = x(organism.x), py = y(organism.y), predator = organism.species === 'predator', radius = predator ? 9 : 7, selected = organism.id === state.demoIndividual; context.fillStyle = predator ? '#f39b72' : '#63d5c2'; context.beginPath(); context.arc(px, py, radius, 0, 2 * Math.PI); context.fill(); context.strokeStyle = selected ? '#fff2a8' : '#e9edf7'; context.lineWidth = selected ? 3 : 1.5; context.beginPath(); context.arc(px, py, radius + (selected ? 4 : 0), 0, 2 * Math.PI); context.stroke(); context.beginPath(); context.moveTo(px, py); context.lineTo(px + Math.cos(organism.heading) * radius * 1.6, py - Math.sin(organism.heading) * radius * 1.6); context.stroke(); context.fillStyle = '#cbd6e8'; context.font = '10px ui-monospace'; context.fillText(`${organism.id} e${organism.energy.toFixed(1)}`, px + radius + 3, py - radius - 2); });
    context.fillStyle = '#9caac2'; context.font = '12px system-ui'; context.fillText('green = plant · cyan = prey · orange = predator · line = heading', 12, 20);
    ui.demoLabel.textContent = `world tick ${snapshot.tick} · prey ${world.population.prey} · predators ${world.population.predator} · plants ${world.plants.length}`;
    const events = snapshot.events || {}; ui.demoEvents.textContent = `Click an organism to inspect its network · births: ${(events.births || []).join(', ') || 'none'} · deaths: ${(events.deaths || []).join(', ') || 'none'} · meals: ${(events.meals || []).map((meal) => `${meal.species}:${Number(meal.interval).toFixed(2)}s`).join(', ') || 'none'}`;
    if (state.demo) state.demo.snapshot = snapshot;
  }
  async function advanceDemo(ticks = 1) {
    if (!state.demo || state.demo.pending) return;
    state.demo.pending = true;
    try { const response = await fetch(`/api/embodied/sessions/${state.demo.id}/step`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ticks})}), snapshot = await response.json(); if (!response.ok) throw new Error(apiError(snapshot, 'demo update failed')); drawDemo(snapshot); if (state.demoIndividual) refreshDemoIndividual(); } catch (error) { state.demo.playing = false; ui.demoPlay.textContent = 'Play'; ui.demoStatus.textContent = `Demo update failed: ${error.message}`; } finally { if (state.demo) state.demo.pending = false; }
  }
  function animateDemo(now) { if (!state.demo?.playing) return; if (now - state.demoLastTick >= 1000 / Number(ui.demoRate.value)) { state.demoLastTick = now; advanceDemo(); } requestAnimationFrame(animateDemo); }
  async function startDemo(event) {
    event.preventDefault(); if (!ui.demoRun.value) return;
    ui.demoStart.disabled = true; ui.demoStatus.textContent = 'Creating a fresh ecology with the selected best rules…';
    try { const response = await fetch('/api/embodied/sessions', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({run_id:ui.demoRun.value, seed:Number(ui.demoSeed.value), prey_count:Number(ui.demoPreyCount.value), predator_count:Number(ui.demoPredatorCount.value), initial_food:Number(ui.demoInitialFood.value), max_food:Number(ui.demoMaxFood.value), food_growth_rate:Number(ui.demoFoodGrowthRate.value)})}), snapshot = await response.json(); if (!response.ok) throw new Error(apiError(snapshot, 'could not create demo')); if (state.demoRecorder?.state === 'recording') state.demoRecorder.stop(); state.demo = {id:snapshot.session_id, pending:false, playing:false, snapshot}; state.demoIndividual = null; state.demoHistory = []; ui.demoColorChannel.replaceChildren(); ui.demoColorChannel.disabled = true; ui.demoChannel.replaceChildren(); ui.demoChannel.disabled = true; ui.demoIndividualTitle.textContent = 'Click a prey or predator in the ecology'; ui.demoIndividualSummary.textContent = 'The selected organism\'s random recurrent graph and live state will appear here.'; ui.demoNetwork.replaceChildren(); drawDemoNodeChart(); ui.demoRecord.disabled = false; drawDemo(snapshot); const source = snapshot.model_source === 'current_checkpoint' ? 'the current training checkpoint' : 'the completed run'; ui.demoStatus.textContent = `Loaded ${source} ${snapshot.run_id.slice(0, 8)} with independent ecology settings. Press Play or Step.`; } catch (error) { ui.demoStatus.textContent = `Could not start demo: ${error.message}`; } finally { ui.demoStart.disabled = false; }
  }
  function toggleDemoRecording() {
    if (!state.demo) return;
    if (state.demoRecorder?.state === 'recording') { state.demoRecorder.stop(); return; }
    if (!window.MediaRecorder || !ui.demoCanvas.captureStream) { ui.demoStatus.textContent = 'This browser cannot record the ecology canvas. Use a current Chromium, Firefox, or Safari version.'; return; }
    const stream = ui.demoCanvas.captureStream(Math.max(12, Number(ui.demoRate.value) || 30));
    const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((type) => MediaRecorder.isTypeSupported(type));
    state.demoRecordingChunks = [];
    const recorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined); state.demoRecorder = recorder;
    recorder.ondataavailable = (event) => { if (event.data.size) state.demoRecordingChunks.push(event.data); };
    recorder.onstop = () => { const blob = new Blob(state.demoRecordingChunks, {type:recorder.mimeType || 'video/webm'}), link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `ecology-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`; link.click(); window.setTimeout(() => URL.revokeObjectURL(link.href), 1000); stream.getTracks().forEach((track) => track.stop()); state.demoRecorder = null; ui.demoRecord.textContent = 'Start recording'; ui.demoStatus.textContent = `Recording downloaded (${(blob.size / 1024 / 1024).toFixed(1)} MB WebM).`; };
    recorder.start(1000); ui.demoRecord.textContent = 'Stop recording'; ui.demoStatus.textContent = 'Recording the ecology canvas. Playback can continue; press Stop recording to download the video.';
  }

  document.querySelectorAll('.nav').forEach((button) => button.addEventListener('click', () => activate(button.dataset.view)));
  window.addEventListener('popstate', () => activate(location.hash.slice(1) || 'survival', false));
  ui.run.addEventListener('change', () => { state.runName = ui.run.value; state.frame = state.batch = state.coordinate = 0; state.selected = null; refreshSelectors(); drawGraph(); update(); updateOverview(); }); ui.batch.addEventListener('change', () => { state.batch = Number(ui.batch.value); update(); }); ui.coordinate.addEventListener('change', () => { state.coordinate = Number(ui.coordinate.value); update(); }); ui.slider.addEventListener('input', () => { state.frame = Number(ui.slider.value); update(); }); ui.prev.addEventListener('click', () => move(-1)); ui.next.addEventListener('click', () => move(1)); ui.play.addEventListener('click', () => { state.playing = !state.playing; ui.play.textContent = state.playing ? 'Pause' : 'Play'; state.lastTick = performance.now(); if (state.playing) requestAnimationFrame(animate); });
  ui.form.addEventListener('submit', async (event) => { event.preventDefault(); const values = Object.fromEntries(new FormData(ui.form)); ['seed','nodes','steps','batch_size'].forEach((key) => { values[key] = Number(values[key]); }); ['mean_degree','dt'].forEach((key) => { values[key] = Number(values[key]); }); ui.runStatus.textContent = 'Running deterministic local experiment…'; try { const response = await fetch('/api/experiment', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(values)}), document = await response.json(); if (!response.ok) throw new Error(apiError(document, 'experiment request failed')); load(document); ui.runStatus.textContent = 'Reference simulation loaded. Open Replay for details.'; } catch (error) { ui.runStatus.textContent = `Could not start: ${error.message}`; } });
  ui.file.addEventListener('change', async () => { const file = ui.file.files[0]; if (!file) return; try { load(JSON.parse(await file.text())); } catch (error) { ui.status.textContent = error.message; } });
  restorePreferences();
  ui.asyncForm.addEventListener('submit', startAsync); ui.asyncDiagnostic.addEventListener('click', startAsyncDiagnostic); ui.asyncRefresh.addEventListener('click', loadLatestAsync); [ui.asyncCandidateBudget, ui.asyncReplicasInput, ui.asyncStablePopulation].forEach((input) => input.addEventListener('input', updateAsyncEstimate)); updateAsyncEstimate(); activate(location.hash.slice(1) || 'survival', false); loadLatestAsync();
  ui.embodiedForm.addEventListener('submit', startEmbodied); ui.embodiedTerminate.addEventListener('click', terminateEmbodied); ui.embodiedRefresh.addEventListener('click', refreshEmbodiedModels); ui.embodiedModel.addEventListener('change', () => { if (ui.embodiedModel.value) { ui.embodiedContinueRun.value = ''; const width = state.embodiedModelWidths[ui.embodiedModel.value]; if (width) ui.embodiedStateWidth.value = String(width); } }); ui.embodiedContinueRun.addEventListener('change', () => { if (ui.embodiedContinueRun.value) { ui.embodiedModel.value = ''; const width = state.embodiedRunWidths[ui.embodiedContinueRun.value]; if (width) ui.embodiedStateWidth.value = String(width); } }); ui.embodiedEnergyScale.addEventListener('input', updateEmbodiedHorizonSuggestion); ui.embodiedSurvivalPressure.addEventListener('change', updateEmbodiedHorizonSuggestion); updateEmbodiedHorizonSuggestion(); refreshEmbodiedModels();
  ui.demoForm.addEventListener('submit', startDemo); ui.demoRefresh.addEventListener('click', refreshDemoRuns); ui.demoPlay.addEventListener('click', () => { if (!state.demo) return; state.demo.playing = !state.demo.playing; ui.demoPlay.textContent = state.demo.playing ? 'Pause' : 'Play'; state.demoLastTick = performance.now(); if (state.demo.playing) requestAnimationFrame(animateDemo); }); ui.demoStep.addEventListener('click', () => advanceDemo()); ui.demoRecord.addEventListener('click', toggleDemoRecording); ui.demoCanvas.addEventListener('click', (event) => { if (state.demo?.snapshot) selectDemoIndividual(state.demo.snapshot, event); }); ui.demoColorChannel.addEventListener('change', () => { if (state.demo?.lastNetwork) drawDemoNetwork(state.demo.lastNetwork); }); ui.demoChannel.addEventListener('change', drawDemoNodeChart); refreshDemoRuns();
  ui.liveRefresh.addEventListener('click', refreshLiveModels); ui.liveModel.addEventListener('change', renderLiveModelDetail); ui.liveForm.addEventListener('submit', launchLive); ui.livePlay.addEventListener('click', () => { if (!state.live) return; state.playing = !state.playing; ui.livePlay.textContent = state.playing ? 'Pause' : 'Play'; state.lastTick = performance.now(); if (state.playing) requestAnimationFrame(animate); }); ui.liveStep.addEventListener('click', () => advanceLive()); refreshLiveModels();
})();
