# Evolvable State Network

A minimal, deterministic experimental foundation for evolving **generic local
graph dynamics**. It intentionally does not model biological cells or impose
meanings on state-vector coordinates. A node holds a finite vector, edges can
hold another finite vector, and shared local rules transform an incoming
permutation-invariant message sum plus external input.

## Current scope

- Batched synchronous, bounded integration with a continuous-time step `dt`.
- Replaceable `NodeRule` and `EdgeRule` interfaces; edge rules see only their
  current unnamed vector, endpoint vectors, current message, and endpoint
  inputs. Neither rule receives node IDs, global score, future signal, or a
  hidden environment state.
- Runtime-adaptive directed communication channels with fixed topology and unit
  base weights. Edge
  latents take bounded `scale * tanh(local_rule(...))` increments; a smooth
  bounded scalar gate modulates a fixed projected source message.
- Reproducible directed random graphs and index-addressed input/noise streams.
- Trajectory recording plus input-shift, impulse, node-lesion, and weight-noise
  disturbances.
- Metrics for boundedness, non-silence, saturation, activity diversity,
  disturbance response, and recovery.
- Two fixed references: a one-coordinate RNN and a hand-designed stabilizing
  rule. They are comparison points.
- Evolution supports node-only, edge-only, and joint parameter groups through
  a deterministic codec; groups can be exported/restored independently. The
  message projection, aggregation, widths, and rule architectures remain
  experiment configuration rather than genome fields.

Ecological tasks, agents, rewards, biological interpretations, gradients during
simulation, and topology evolution remain deliberately out of scope. Runtime
edge adaptation is generic channel-state dynamics, not a biological analogy.

## Evolve shared node and edge rules asynchronously

The rule architecture, state width, topology, unit base weight, aggregation, and timestep
are experiment configuration—not genome fields. Training scenarios use several
seeds and perturbations; validation and test use disjoint seeds, larger graphs,
stronger disturbances, and longer horizons.

The primary research runner keeps a fixed set of active slots and replaces each
candidate immediately on first-passage death or survival-milestone graduation.
Node and edge rule parameters form one joint genome by default. Runtime node
and edge states are reset and never inherited. Online health accumulators,
paired response probes, right-censor records, common scenario banks, multiple
replicas, buffered CMA-ES updates, and a validated elite archive are written to
the operating system's standard per-user application-data directory.

Run the required short diagnostic before scheduling a larger search:

```powershell
esn-evolve --diagnostic
```

Then, if the diagnostic is satisfactory, start a bounded asynchronous run:

```powershell
esn-evolve --ticks 500 --slots 16 --replicas 3
```

The CLI evolves node and edge rule parameters jointly by default. Use
`--evolve node`, `--evolve edge`, or `--evolve joint` to select a parameter
group explicitly. Newly exported replay data shows each edge's unit base
weight, its current smooth communication strength, and their effective product.

The older fixed-horizon generation runner is retained only for comparison via
`--legacy-generational`. It uses the tensorized Torch backend for the standard MLP
node/edge rules, selecting CUDA when it is available. The generic Python
simulator remains only for custom-rule and intervention experiments.

Every legacy generational run first writes `random_search_smoke.json`. CMA-ES refuses to start
unless the sampled fitness values are non-degenerate. The output also includes
`checkpoint.json`, `best_genome.json`, `evolution_report.json`,
and `analysis/` with trajectory, recovery, state-distribution, correlation,
update-magnitude, train-validation-gap, scale, and long-horizon summaries.

The dashboard's **Phase 1A viability** panel can run the same deterministic
random-search smoke test through the local server. It reports sample count,
fitness range, mean, variability, and viable-scenario fraction; it does not
launch long-running optimization from the browser.

## Run the application

Install the project once into its virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Start the FastAPI application without selecting or preparing an output folder:

```powershell
esn-dashboard
```

Open `http://127.0.0.1:8000/`. Interactive API documentation is available at
`http://127.0.0.1:8000/docs`. The old `/dashboard/` URL redirects to `/`.

FastAPI serves the packaged frontend and artifact files. Uvicorn runs the ASGI
application, Pydantic validates request bodies, and Platformdirs chooses the
correct persistent data location for Windows, macOS, or Linux. Set
`ESN_DATA_DIR` only when an explicit storage override is useful; it is not
required for normal use. `GET /api/health` reports the active storage path.

The dashboard supports wall-clock
playback, stepping, speed and loop controls, baseline/batch/coordinate choices,
and click-to-inspect node and edge state, external input, graph connections,
metrics, and active disturbances. The **New experiment** panel runs a fresh
parameterized simulation through this local server and loads it directly into
the replay view. It can also load any compatible
replay JSON through its file picker.

To generate a fixed-rule comparison from the command line, run `esn-experiment`.
It also stores its metrics, plots, and replay JSON in application data without
requiring an output argument.

## Extending the substrate

Implement `NodeRule` and optionally `EdgeRule` from `rules.py`. Keep each rule
local. Node states use the simulator's numerical safety bound. Edge latent
increments are bounded by the edge rule itself, while the effective channel
strength is smooth-bounded; trajectories expose both so growth is detectable.
`CandidateEvaluator(..., edge_architecture=..., target="node" | "edge" |
"joint")` selects which parameter group evolves. `evaluate_ablations` reports
the four fixed/adaptive node-edge combinations on matched scenario suites.

The metrics currently use coordinate zero as a documented observation
convention. It is not a semantic claim; later experiments may add registered
readouts while keeping rule and simulator interfaces unchanged.

## Verify

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests include dead, saturated, synchronized, and exploding trajectories.
