(() => {
  'use strict';

  const scope = typeof window === 'undefined' ? globalThis : window;
  const MAX_LOCAL_ARTIFACT_BYTES = 50 * 1024 * 1024;
  const COLORS = { prey: '#5fd9c1', predator: '#ff966c', secondary: '#80aaff' };

  class ArtifactLoadError extends Error {
    constructor(message, details = []) {
      super(message);
      this.name = 'ArtifactLoadError';
      this.details = details;
    }
  }

  function isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function genome(document, species) {
    const values = document?.[species]?.best_genome ?? document?.[`${species}_best_genome`];
    return Array.isArray(values) ? values : [];
  }

  function validateArtifact(document, label) {
    if (!isRecord(document)) {
      throw new ArtifactLoadError(`${label} must contain one JSON object.`);
    }
    const missing = [];
    if (!isRecord(document.architecture)) missing.push('architecture');
    if (!isRecord(document.edge_architecture)) missing.push('edge_architecture');
    if (!isRecord(document.task_config)) missing.push('task_config');
    if (!genome(document, 'prey').length && !genome(document, 'predator').length) {
      missing.push('prey_best_genome or predator_best_genome');
    }
    if (missing.length) {
      throw new ArtifactLoadError(
        `${label} is JSON, but not a usable embodied-run artifact.`,
        [`Missing ${missing.join(', ')}.`],
      );
    }
    for (const species of ['prey', 'predator']) {
      const values = genome(document, species);
      if (values.some((value) => !Number.isFinite(Number(value)))) {
        throw new ArtifactLoadError(`${label} contains non-finite ${species} genome values.`);
      }
    }
    return document;
  }

  function compatibilityWarnings(report, checkpoint) {
    if (!report || !checkpoint) return [];
    const warnings = [];
    const reportWidth = Number(report.architecture?.state_width);
    const checkpointWidth = Number(checkpoint.architecture?.state_width);
    if (Number.isFinite(reportWidth) && Number.isFinite(checkpointWidth) && reportWidth !== checkpointWidth) {
      warnings.push(`Report state width ${reportWidth} does not match checkpoint width ${checkpointWidth}.`);
    }
    for (const species of ['prey', 'predator']) {
      const reportLength = genome(report, species).length;
      const checkpointLength = genome(checkpoint, species).length;
      if (reportLength && checkpointLength && reportLength !== checkpointLength) {
        warnings.push(`${species} genome dimensions differ (${reportLength} vs ${checkpointLength}).`);
      }
    }
    return warnings;
  }

  function normalizeBundle({ report = null, checkpoint = null, warnings = [], source = 'unknown' }) {
    const validReport = report === null ? null : validateArtifact(report, 'report.json');
    const validCheckpoint = checkpoint === null ? null : validateArtifact(checkpoint, 'checkpoint.json');
    if (!validReport && !validCheckpoint) {
      throw new ArtifactLoadError('No usable report.json or checkpoint.json was provided.');
    }
    return {
      report: validReport,
      checkpoint: validCheckpoint,
      primary: validReport || validCheckpoint,
      source,
      warnings: [...warnings, ...compatibilityWarnings(validReport, validCheckpoint)],
    };
  }

  async function parseResponse(response, label) {
    const text = await response.text();
    let data;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (error) {
      throw new ArtifactLoadError(`${label} returned malformed JSON.`, [error.message]);
    }
    if (!response.ok) {
      const detail = data?.detail || data?.error || `${response.status} ${response.statusText}`;
      throw new ArtifactLoadError(`${label} could not be loaded: ${detail}`);
    }
    return data;
  }

  async function readLocalArtifact(file, label) {
    if (!file) return null;
    if (file.size > MAX_LOCAL_ARTIFACT_BYTES) {
      throw new ArtifactLoadError(`${label} is too large to inspect safely in the browser.`, [
        `Maximum size is ${MAX_LOCAL_ARTIFACT_BYTES / 1024 / 1024} MB.`,
      ]);
    }
    let document;
    try {
      document = JSON.parse(await file.text());
    } catch (error) {
      throw new ArtifactLoadError(`${file.name || label} is not valid JSON.`, [error.message]);
    }
    return validateArtifact(document, label);
  }

  class DiagnosticLoader {
    constructor(fetchImplementation = scope.fetch?.bind(scope)) {
      this.fetch = fetchImplementation;
      this.controller = null;
      this.sequence = 0;
    }

    begin() {
      this.controller?.abort();
      this.controller = typeof AbortController === 'undefined' ? null : new AbortController();
      this.sequence += 1;
      return { sequence: this.sequence, signal: this.controller?.signal };
    }

    assertCurrent(sequence) {
      if (sequence !== this.sequence) {
        const error = new Error('A newer diagnostic load replaced this request.');
        error.name = 'AbortError';
        throw error;
      }
    }

    async fromServer(runId) {
      if (!this.fetch) throw new ArtifactLoadError('This browser cannot load server artifacts.');
      const request = this.begin();
      const response = await this.fetch(
        `/api/embodied/runs/${encodeURIComponent(runId)}/artifacts`,
        { signal: request.signal },
      );
      const data = await parseResponse(response, `Run ${runId}`);
      this.assertCurrent(request.sequence);
      return normalizeBundle({
        report: data.report,
        checkpoint: data.checkpoint,
        warnings: Array.isArray(data.warnings) ? data.warnings : [],
        source: data.selected_source || 'server',
      });
    }

    async fromFiles(reportFile, checkpointFile) {
      const request = this.begin();
      const [report, checkpoint] = await Promise.all([
        readLocalArtifact(reportFile, 'report.json'),
        readLocalArtifact(checkpointFile, 'checkpoint.json'),
      ]);
      this.assertCurrent(request.sequence);
      return normalizeBundle({ report, checkpoint, source: 'local_files' });
    }
  }

  function abortableDelay(milliseconds, signal) {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        const error = new Error('Diagnostic job polling was cancelled.');
        error.name = 'AbortError';
        reject(error);
        return;
      }
      const timeout = setTimeout(resolve, milliseconds);
      signal?.addEventListener('abort', () => {
        clearTimeout(timeout);
        const error = new Error('Diagnostic job polling was cancelled.');
        error.name = 'AbortError';
        reject(error);
      }, { once: true });
    });
  }

  async function waitForJob(
    fetchImplementation,
    jobId,
    { signal, interval = 500, timeout = 5 * 60 * 1000 } = {},
  ) {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      const response = await fetchImplementation(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
      const job = await parseResponse(response, `Diagnostic job ${jobId}`);
      if (job.status === 'complete') return job.result;
      if (['failed', 'terminated'].includes(job.status)) {
        throw new ArtifactLoadError(job.error || `Diagnostic job ${job.status}.`);
      }
      await abortableDelay(interval, signal);
    }
    throw new ArtifactLoadError(`Diagnostic job ${jobId} did not finish within ${Math.round(timeout / 1000)} seconds.`);
  }

  function prepareCanvas(canvas, aspectRatio = 0.48) {
    const ratio = scope.devicePixelRatio || 1;
    const width = Math.max(320, canvas.clientWidth || canvas.width);
    const height = Math.max(220, width * aspectRatio);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.height = `${height}px`;
    const context = canvas.getContext('2d');
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#0b1119';
    context.fillRect(0, 0, width, height);
    return { context, width, height };
  }

  function emptyPlot(canvas, message) {
    const { context } = prepareCanvas(canvas);
    context.fillStyle = '#8292a5';
    context.font = '13px system-ui';
    context.fillText(message, 22, 34);
  }

  function drawFrame(context, width, height, range, label) {
    const margin = { left: 54, right: 18, top: 24, bottom: 38 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const spread = range.maximum - range.minimum || 1;
    const y = (value) => margin.top + (range.maximum - value) / spread * plotHeight;
    context.font = '10px ui-monospace';
    for (let index = 0; index <= 4; index += 1) {
      const value = range.minimum + spread * index / 4;
      const py = y(value);
      context.strokeStyle = '#253241';
      context.beginPath(); context.moveTo(margin.left, py); context.lineTo(width - margin.right, py); context.stroke();
      context.fillStyle = '#718196'; context.fillText(value.toFixed(2), 5, py + 3);
    }
    context.fillStyle = '#8797aa';
    context.fillText(label, margin.left, 14);
    return { margin, plotWidth, plotHeight, y };
  }

  function finiteSeries(history, key) {
    return history.map((entry, index) => ({
      x: Number(entry.generation ?? entry.tick ?? index + 1),
      y: Number(entry[key]),
    })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  }

  function drawHistoryPlot(canvas, history, metric = 'best_lifetime') {
    const prey = finiteSeries(history, `prey_${metric}`);
    const predator = finiteSeries(history, `predator_${metric}`);
    const all = [...prey, ...predator];
    if (!all.length) {
      emptyPlot(canvas, `No recorded ${metric.replaceAll('_', ' ')} history.`);
      return;
    }
    const { context, width, height } = prepareCanvas(canvas);
    const values = all.map((point) => point.y);
    const padding = Math.max(0.01, (Math.max(...values) - Math.min(...values)) * 0.08);
    const frame = drawFrame(context, width, height, {
      minimum: Math.min(...values) - padding,
      maximum: Math.max(...values) + padding,
    }, metric.replaceAll('_', ' '));
    const minX = Math.min(...all.map((point) => point.x));
    const maxX = Math.max(...all.map((point) => point.x));
    const mapX = (value) => frame.margin.left + (value - minX) / (maxX - minX || 1) * frame.plotWidth;
    [[prey, COLORS.prey, 'prey'], [predator, COLORS.predator, 'predator']].forEach(([series, color, label], labelIndex) => {
      if (!series.length) return;
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.beginPath();
      series.forEach((point, index) => {
        if (index) context.lineTo(mapX(point.x), frame.y(point.y));
        else context.moveTo(mapX(point.x), frame.y(point.y));
      });
      context.stroke();
      context.fillStyle = color;
      context.fillText(label, frame.margin.left + labelIndex * 70, height - 12);
    });
    context.fillStyle = '#718196';
    context.textAlign = 'right';
    context.fillText(`${all.length} recorded points`, width - frame.margin.right, height - 12);
    context.textAlign = 'left';
  }

  function drawGenomeHistogram(canvas, preyGenome, predatorGenome) {
    const prey = preyGenome.map(Number).filter(Number.isFinite);
    const predator = predatorGenome.map(Number).filter(Number.isFinite);
    const values = [...prey, ...predator];
    if (!values.length) {
      emptyPlot(canvas, 'No final genome parameters were recorded.');
      return;
    }
    const { context, width, height } = prepareCanvas(canvas);
    const limit = Math.max(0.1, ...values.map(Math.abs));
    const binCount = 31;
    const bin = (items) => {
      const counts = new Array(binCount).fill(0);
      items.forEach((value) => {
        const index = Math.max(0, Math.min(binCount - 1, Math.floor((value + limit) / (2 * limit) * binCount)));
        counts[index] += 1 / Math.max(1, items.length);
      });
      return counts;
    };
    const preyBins = bin(prey);
    const predatorBins = bin(predator);
    const maximum = Math.max(...preyBins, ...predatorBins, 0.01);
    const margin = { left: 44, right: 16, top: 26, bottom: 38 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    context.strokeStyle = '#334253';
    context.beginPath(); context.moveTo(margin.left, margin.top); context.lineTo(margin.left, height - margin.bottom); context.lineTo(width - margin.right, height - margin.bottom); context.stroke();
    const barWidth = plotWidth / binCount;
    [[preyBins, COLORS.prey, -0.16], [predatorBins, COLORS.predator, 0.16]].forEach(([counts, color, offset]) => {
      context.fillStyle = color;
      context.globalAlpha = 0.62;
      counts.forEach((count, index) => {
        const barHeight = count / maximum * plotHeight;
        context.fillRect(margin.left + index * barWidth + barWidth * offset, height - margin.bottom - barHeight, barWidth * 0.62, barHeight);
      });
    });
    context.globalAlpha = 1;
    context.font = '10px ui-monospace';
    context.fillStyle = '#718196';
    context.fillText(`−${limit.toFixed(2)}`, margin.left, height - 13);
    context.textAlign = 'center'; context.fillText('parameter value', width / 2, height - 13);
    context.textAlign = 'right'; context.fillText(`+${limit.toFixed(2)}`, width - margin.right, height - 13);
    context.textAlign = 'left';
    context.fillStyle = COLORS.prey; context.fillText('prey', margin.left, 15);
    context.fillStyle = COLORS.predator; context.fillText('predator', margin.left + 52, 15);
  }

  function drawReliabilityPlot(canvas, result) {
    const prey = (result?.prey?.fitness_values || []).map(Number).filter(Number.isFinite);
    const predator = (result?.predator?.fitness_values || []).map(Number).filter(Number.isFinite);
    const values = [...prey, ...predator];
    if (!values.length) {
      emptyPlot(canvas, 'Run the fresh-graph test to plot trial reliability.');
      return;
    }
    const { context, width, height } = prepareCanvas(canvas, 0.34);
    const margin = { left: 68, right: 20, top: 28, bottom: 38 };
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const mapX = (value) => margin.left + (value - minimum) / (maximum - minimum || 1) * (width - margin.left - margin.right);
    context.strokeStyle = '#334253';
    context.beginPath(); context.moveTo(margin.left, height - margin.bottom); context.lineTo(width - margin.right, height - margin.bottom); context.stroke();
    [[prey, COLORS.prey, 70, 'prey'], [predator, COLORS.predator, 142, 'predator']].forEach(([series, color, y, label]) => {
      context.fillStyle = '#8392a5'; context.font = '11px ui-monospace'; context.fillText(label, 10, y + 4);
      context.strokeStyle = '#233140'; context.beginPath(); context.moveTo(margin.left, y); context.lineTo(width - margin.right, y); context.stroke();
      context.fillStyle = color;
      series.forEach((value, index) => {
        const jitter = ((index * 17) % 13) - 6;
        context.beginPath(); context.arc(mapX(value), y + jitter, 3.5, 0, 2 * Math.PI); context.fill();
      });
    });
    context.fillStyle = '#718196'; context.fillText(minimum.toFixed(2), margin.left, height - 13);
    context.textAlign = 'right'; context.fillText(maximum.toFixed(2), width - margin.right, height - 13); context.textAlign = 'left';
  }

  function drawSensitivityPlot(canvas, comparison) {
    const results = Object.entries(comparison?.parameter_scaling?.results || {})
      .map(([scale, data]) => ({ scale: Number(scale), fitness: Number(data.fitness?.mean) }))
      .filter((point) => Number.isFinite(point.scale) && Number.isFinite(point.fitness))
      .sort((left, right) => left.scale - right.scale);
    if (!results.length) {
      emptyPlot(canvas, 'Evaluate two checkpoints to plot parameter sensitivity.');
      return;
    }
    const { context, width, height } = prepareCanvas(canvas, 0.4);
    const values = results.map((point) => point.fitness);
    const padding = Math.max(0.01, (Math.max(...values) - Math.min(...values)) * 0.1);
    const frame = drawFrame(context, width, height, {
      minimum: Math.min(...values) - padding,
      maximum: Math.max(...values) + padding,
    }, 'mean fitness under parameter scaling');
    const minScale = Math.min(...results.map((point) => point.scale));
    const maxScale = Math.max(...results.map((point) => point.scale));
    const mapX = (value) => frame.margin.left + (value - minScale) / (maxScale - minScale || 1) * frame.plotWidth;
    context.strokeStyle = COLORS.secondary; context.fillStyle = COLORS.secondary; context.lineWidth = 2; context.beginPath();
    results.forEach((point, index) => {
      const x = mapX(point.scale), y = frame.y(point.fitness);
      if (index) context.lineTo(x, y); else context.moveTo(x, y);
    });
    context.stroke();
    results.forEach((point) => { context.beginPath(); context.arc(mapX(point.scale), frame.y(point.fitness), 4, 0, 2 * Math.PI); context.fill(); });
    context.fillStyle = '#718196'; context.fillText(`×${minScale}`, frame.margin.left, height - 12);
    context.textAlign = 'right'; context.fillText(`×${maxScale}`, width - frame.margin.right, height - 12); context.textAlign = 'left';
  }

  scope.StateNetworkDiagnostics = Object.freeze({
    ArtifactLoadError,
    DiagnosticLoader,
    waitForJob,
    validateArtifact,
    normalizeBundle,
    drawHistoryPlot,
    drawGenomeHistogram,
    drawReliabilityPlot,
    drawSensitivityPlot,
  });
})();
