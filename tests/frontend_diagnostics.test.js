'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

require('../src/evolvable_state_network/web/diagnostics.js');
require('../src/evolvable_state_network/web/ecology.js');

const {
  DiagnosticLoader,
  normalizeBundle,
  validateArtifact,
  waitForJob,
} = global.StateNetworkDiagnostics;
const {HistoryBuffer, drawNodeHistoryChart} = global.StateNetworkEcology;

test('ecology history retains a rolling window and supports allocation-free traversal', () => {
  const history = new HistoryBuffer(3);
  history.push('first');
  history.push('second');
  history.push('third');
  history.push('fourth');

  assert.equal(history.length, 3);
  assert.equal(history.at(0), 'second');
  assert.equal(history.at(-1), 'fourth');
  assert.deepEqual(history.toArray(), ['second', 'third', 'fourth']);
  const visited = [];
  history.forEach((value) => visited.push(value));
  assert.deepEqual(visited, ['second', 'third', 'fourth']);
});

test('node history plot reads a rolling HistoryBuffer without array indexing', () => {
  const history = new HistoryBuffer(2);
  history.push({tick: 4, node_state: [[0.1], [-0.2]]});
  history.push({tick: 5, node_state: [[0.3], [-0.4]]});
  const context = {
    beginPath() {}, clearRect() {}, fill() {}, fillRect() {}, fillText() {}, lineTo() {}, moveTo() {}, stroke() {},
  };
  const canvas = {width: 200, height: 100, getContext: () => context};

  const series = drawNodeHistoryChart(canvas, history, {channel: 0, seriesCount: 1});

  assert.equal(series.length, 1);
  assert.equal(series[0].node, 1);
});

function artifact(overrides = {}) {
  return {
    architecture: {state_width: 2},
    edge_architecture: {latent_width: 2},
    task_config: {embodied_interface: 'ray_image_v3_sparse_multichannel_v1'},
    prey_best_genome: [0, 0.1, -0.1],
    predator_best_genome: [0.2, 0, -0.2],
    ...overrides,
  };
}

function response(data, {ok = true, status = 200, statusText = 'OK'} = {}) {
  return {
    ok,
    status,
    statusText,
    text: async () => JSON.stringify(data),
  };
}

test('artifact validation rejects arrays, missing structure, and non-finite genomes', () => {
  assert.throws(() => validateArtifact([], 'report.json'), /one JSON object/);
  assert.throws(() => validateArtifact({}, 'report.json'), /not a usable embodied-run artifact/);
  assert.throws(
    () => validateArtifact(artifact({prey_best_genome: [Number.NaN]}), 'report.json'),
    /non-finite prey genome/,
  );
});

test('bundle normalization reports incompatible report and checkpoint dimensions', () => {
  const bundle = normalizeBundle({
    report: artifact(),
    checkpoint: artifact({
      architecture: {state_width: 3},
      prey_best_genome: [0, 1],
    }),
    source: 'local_files',
  });

  assert.equal(bundle.primary, bundle.report);
  assert.ok(bundle.warnings.some((warning) => warning.includes('state width')));
  assert.ok(bundle.warnings.some((warning) => warning.includes('genome dimensions')));
});

test('a newer server load suppresses a stale response', async () => {
  let resolveFirst;
  const firstResponse = new Promise((resolve) => { resolveFirst = resolve; });
  const fetch = (url) => url.includes('/first/')
    ? firstResponse
    : Promise.resolve(response({report: artifact(), checkpoint: null, warnings: [], selected_source: 'completed_report'}));
  const loader = new DiagnosticLoader(fetch);

  const stale = loader.fromServer('first');
  const current = loader.fromServer('second');
  resolveFirst(response({report: artifact(), checkpoint: null, warnings: [], selected_source: 'completed_report'}));

  await assert.rejects(stale, (error) => error.name === 'AbortError');
  assert.equal((await current).source, 'completed_report');
});

test('job polling tolerates running states and returns the completed result', async () => {
  const states = [
    {status: 'running'},
    {status: 'running'},
    {status: 'complete', result: {score: 4.2}},
  ];
  const fetch = async () => response(states.shift());

  const result = await waitForJob(fetch, 'job-1', {interval: 1, timeout: 1000});

  assert.deepEqual(result, {score: 4.2});
});

test('job polling surfaces terminal failures', async () => {
  const fetch = async () => response({status: 'failed', error: 'numerical failure'});
  await assert.rejects(
    waitForJob(fetch, 'job-2', {interval: 1, timeout: 1000}),
    /numerical failure/,
  );
});
