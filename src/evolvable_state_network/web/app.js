(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const ui = {
    run: $('run-select'), batch: $('batch-select'), coordinate: $('coordinate-select'), rate: $('rate-select'),
    slider: $('frame-slider'), label: $('frame-label'), play: $('play'), prev: $('previous'), next: $('next'), loop: $('loop'),
    svg: $('network'), status: $('load-status'), file: $('file-input'), frameInfo: $('frame-info'), selection: $('selection-info'),
    metrics: $('metrics'), events: $('events'), form: $('experiment-form'), runStatus: $('run-status')
  };
  const state = { data: null, runName: '', frame: 0, batch: 0, coordinate: 0, selected: null, playing: false, lastTick: 0, layout: [] };
  const NS = 'http://www.w3.org/2000/svg';
  const element = (tag, attrs = {}) => { const node = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k,v]) => node.setAttribute(k, v)); return node; };

  function load(data) {
    if (!data || data.schema_version !== 1 || !data.graph || !data.runs) throw new Error('This is not a supported dashboard_data.json file.');
    state.data = data; state.runName = Object.keys(data.runs)[0]; state.frame = state.batch = state.coordinate = 0; state.selected = null;
    ui.run.replaceChildren(...Object.keys(data.runs).map((name) => new Option(name, name)));
    refreshSelectors(); drawGraph(); update(); ui.status.textContent = `Loaded ${data.graph.nodes} nodes, ${data.graph.edges.length} directed edges.`;
  }
  function activeRun() { return state.data.runs[state.runName]; }
  function trajectory() { return activeRun().trajectory; }
  function refreshSelectors() {
    const first = trajectory().node_states[0];
    ui.batch.replaceChildren(...first.map((_, i) => new Option(`batch ${i}`, i)));
    ui.coordinate.replaceChildren(...first[0][0].map((_, i) => new Option(`coordinate ${i}`, i)));
    ui.slider.max = String(trajectory().steps.length - 1); ui.slider.value = String(state.frame);
  }
  function nodePositions(count, edges) {
    // A deterministic, index-sequential force layout. The starting grid keeps
    // disconnected nodes readable in sequence; local graph links then pull
    // related nodes together while pairwise repulsion prevents circular rings.
    if (count === 1) return [{x:360, y:280}];
    const columns = Math.ceil(Math.sqrt(count * 1.25));
    const rows = Math.ceil(count / columns), left = 70, right = 650, top = 64, bottom = 500;
    const anchors = Array.from({length: count}, (_, i) => ({
      x: left + (i % columns) * (right - left) / Math.max(1, columns - 1),
      y: top + Math.floor(i / columns) * (bottom - top) / Math.max(1, rows - 1)
    }));
    const points = anchors.map((point) => ({...point}));
    for (let iteration = 0; iteration < 180; iteration += 1) {
      const force = Array.from({length: count}, () => ({x:0, y:0}));
      for (let a = 0; a < count; a += 1) for (let b = a + 1; b < count; b += 1) {
        const dx = points[a].x - points[b].x, dy = points[a].y - points[b].y, distance = Math.max(1, Math.hypot(dx, dy));
        const amount = 2100 / (distance * distance), x = dx / distance * amount, y = dy / distance * amount;
        force[a].x += x; force[a].y += y; force[b].x -= x; force[b].y -= y;
      }
      edges.forEach((edge) => {
        const a = edge.source, b = edge.target, dx = points[a].x - points[b].x, dy = points[a].y - points[b].y, distance = Math.max(1, Math.hypot(dx, dy));
        const amount = (distance - 118) * 0.025, x = dx / distance * amount, y = dy / distance * amount;
        force[a].x -= x; force[a].y -= y; force[b].x += x; force[b].y += y;
      });
      points.forEach((point, index) => {
        force[index].x += (anchors[index].x - point.x) * 0.055; force[index].y += (anchors[index].y - point.y) * 0.055;
        point.x = Math.max(28, Math.min(692, point.x + Math.max(-6, Math.min(6, force[index].x))));
        point.y = Math.max(28, Math.min(532, point.y + Math.max(-6, Math.min(6, force[index].y))));
      });
    }
    return points;
  }
  function drawGraph() {
    state.layout = nodePositions(state.data.graph.nodes, state.data.graph.edges); ui.svg.replaceChildren();
    const defs = element('defs'); const marker = element('marker', {id:'arrow', markerWidth:'7', markerHeight:'7', refX:'6', refY:'3.5', orient:'auto'}); marker.append(element('path', {d:'M0,0 L7,3.5 L0,7 z', fill:'#71809b'})); defs.append(marker); ui.svg.append(defs);
    const edgeLayer = element('g', {id:'edges'}); const nodeLayer = element('g', {id:'nodes'}); ui.svg.append(edgeLayer, nodeLayer);
    state.data.graph.edges.forEach((edge, index) => {
      const a = state.layout[edge.source], b = state.layout[edge.target]; const line = element('line', {x1:a.x, y1:a.y, x2:b.x, y2:b.y, 'marker-end':'url(#arrow)', class:'edge', 'data-edge':index});
      line.addEventListener('click', () => { state.selected = {kind:'edge', index}; update(); }); edgeLayer.append(line);
    });
    state.layout.forEach((point, index) => {
      const group = element('g'); const circle = element('circle', {cx:point.x, cy:point.y, r:'12', class:'node', 'data-node':index}); const label = element('text', {x:point.x, y:point.y + 4, class:'node-label'}); label.textContent = index;
      circle.addEventListener('click', () => { state.selected = {kind:'node', index}; update(); }); group.append(circle, label); nodeLayer.append(group);
    });
  }
  function currentValues() { return trajectory().node_states[state.frame][state.batch].map((vector) => vector[state.coordinate]); }
  function color(value, scale) { const t = Math.min(1, Math.abs(value) / scale); const base = value >= 0 ? [250, 130, 103] : [90, 169, 255]; const mix = base.map((v) => Math.round(198 + (v - 198) * t)); return `rgb(${mix.join(',')})`; }
  function update() {
    if (!state.data) return; const trace = trajectory(), values = currentValues(); const scale = Math.max(0.15, ...values.map(Math.abs));
    ui.slider.value = String(state.frame); ui.label.value = `step ${trace.steps[state.frame]} | t=${Number(trace.times[state.frame]).toFixed(3)}`;
    ui.svg.querySelectorAll('.node').forEach((node, index) => { const value = values[index]; node.setAttribute('r', String(10 + Math.min(16, Math.abs(value) / scale * 16))); node.setAttribute('fill', color(value, scale)); node.classList.toggle('selected', state.selected?.kind === 'node' && state.selected.index === index); });
    ui.svg.querySelectorAll('.edge').forEach((line, index) => { const weight = state.data.graph.edges[index].weight; line.setAttribute('stroke', weight >= 0 ? '#f39b72' : '#6daafa'); line.setAttribute('stroke-width', String(.7 + Math.min(3.6, Math.abs(weight) * 2.5))); line.setAttribute('opacity', String(.22 + Math.min(.65, Math.abs(weight)))); line.classList.toggle('selected', state.selected?.kind === 'edge' && state.selected.index === index); });
    updateDiagnostics(values); updateSelection(values); updateMetrics(); updateEvents();
  }
  function updateDiagnostics(values) {
    const trace = trajectory(), frame = trace.node_states[state.frame][state.batch], finite = values.every(Number.isFinite), magnitude = values.reduce((sum, x) => sum + Math.abs(x), 0) / values.length;
    const config = state.data.simulation_config || {}; const rows = [['run', state.runName], ['recorded frame', `${state.frame + 1} / ${trace.steps.length}`], ['integration dt', config.dt ?? 'not recorded'], ['state vector width', frame[0].length], ['mean |selected coordinate|', magnitude.toFixed(5)], ['all finite', finite ? 'yes' : 'NO'], ['edge-state width', trace.edge_states[state.frame]?.[state.batch]?.[0]?.length ?? 0]];
    ui.frameInfo.replaceChildren(...rows.flatMap(([term, value]) => { const dt = document.createElement('dt'); dt.textContent = term; const dd = document.createElement('dd'); dd.textContent = value; return [dt, dd]; }));
  }
  function updateSelection(values) {
    if (!state.selected) { ui.selection.textContent = 'Select a node or edge.'; return; }
    const trace = trajectory(), item = state.selected;
    if (item.kind === 'node') {
      const edges = state.data.graph.edges.map((edge, index) => ({edge, index})).filter(({edge}) => edge.source === item.index || edge.target === item.index).map(({edge,index}) => ({edge:index, direction:edge.source === item.index ? 'out' : 'in', other:edge.source === item.index ? edge.target : edge.source, weight:+edge.weight.toFixed(5)}));
      ui.selection.textContent = JSON.stringify({kind:'node', node:item.index, state:trace.node_states[state.frame][state.batch][item.index], external_input:trace.inputs[state.frame][state.batch][item.index], selected_coordinate:values[item.index], incident_edges:edges}, null, 2);
    } else {
      const edge = state.data.graph.edges[item.index], vector = trace.edge_states[state.frame]?.[state.batch]?.[item.index] ?? [];
      ui.selection.textContent = JSON.stringify({kind:'edge', edge:item.index, ...edge, edge_state:vector, source_state:trace.node_states[state.frame][state.batch][edge.source], target_state:trace.node_states[state.frame][state.batch][edge.target]}, null, 2);
    }
  }
  function updateMetrics() { const metrics = activeRun().metrics; ui.metrics.replaceChildren(...Object.entries(metrics).map(([name, value]) => { const row = document.createElement('div'); row.className = 'metric'; const label = document.createElement('span'); label.textContent = name.replaceAll('_', ' '); const outcome = document.createElement('strong'); const status = value.bounded ?? value.non_silent ?? value.diverse ?? value.responsive ?? value.recovered; const inverted = name === 'saturation'; const pass = inverted ? !value.saturated : status; outcome.textContent = pass === undefined ? JSON.stringify(value) : (pass ? 'pass' : 'flag'); row.append(label, outcome); return row; })); }
  function updateEvents() { const step = trajectory().steps[state.frame]; ui.events.replaceChildren(...trajectory().events.map((event) => { const entry = document.createElement('div'); entry.className = `event ${step >= event.start && step <= event.end + 1 ? 'active' : ''}`; entry.textContent = `${event.kind}: ${event.start}-${event.end}`; return entry; })); }
  function move(delta) { const last = trajectory().steps.length - 1; state.frame += delta; if (state.frame > last) state.frame = ui.loop.checked ? 0 : last; if (state.frame < 0) state.frame = ui.loop.checked ? last : 0; update(); }
  function animate(now) { if (!state.playing) return; const interval = 1000 / Number(ui.rate.value); if (now - state.lastTick >= interval) { move(1); state.lastTick = now; if (state.frame === trajectory().steps.length - 1 && !ui.loop.checked) { state.playing = false; ui.play.textContent = 'Play'; return; } } requestAnimationFrame(animate); }
  ui.run.addEventListener('change', () => { state.runName = ui.run.value; state.frame = state.batch = state.coordinate = 0; state.selected = null; refreshSelectors(); drawGraph(); update(); });
  ui.batch.addEventListener('change', () => { state.batch = Number(ui.batch.value); update(); }); ui.coordinate.addEventListener('change', () => { state.coordinate = Number(ui.coordinate.value); update(); }); ui.slider.addEventListener('input', () => { state.frame = Number(ui.slider.value); update(); }); ui.prev.addEventListener('click', () => move(-1)); ui.next.addEventListener('click', () => move(1));
  ui.play.addEventListener('click', () => { state.playing = !state.playing; ui.play.textContent = state.playing ? 'Pause' : 'Play'; state.lastTick = performance.now(); if (state.playing) requestAnimationFrame(animate); });
  ui.form.addEventListener('submit', async (event) => {
    event.preventDefault(); const values = Object.fromEntries(new FormData(ui.form));
    ['seed', 'nodes', 'steps', 'batch_size'].forEach((key) => { values[key] = Number(values[key]); });
    ['mean_degree', 'dt'].forEach((key) => { values[key] = Number(values[key]); });
    ui.runStatus.textContent = 'Running deterministic local experiment...';
    try {
      const response = await fetch('/api/experiment', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(values)});
      const text = await response.text(); let document;
      try { document = JSON.parse(text); } catch (_) { throw new Error('This page is being served by a static or older server. Stop it and run: python -m evolvable_state_network.server --output experiment_output'); }
      if (!response.ok) throw new Error(document.error || 'experiment request failed');
      load(document); ui.runStatus.textContent = 'Fresh experiment loaded into replay and debug view.';
    } catch (error) { ui.runStatus.textContent = `Could not start: ${error.message}`; }
  });
  ui.file.addEventListener('change', async () => { const file = ui.file.files[0]; if (!file) return; try { load(JSON.parse(await file.text())); } catch (error) { ui.status.textContent = error.message; } });
  fetch('../dashboard_data.json').then((response) => { if (!response.ok) throw new Error('Bundled data unavailable; load dashboard_data.json manually.'); return response.json(); }).then(load).catch((error) => { ui.status.textContent = error.message; });
})();
