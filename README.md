# Evolvable State Network

A minimal, deterministic experimental foundation for evolving **generic local
graph dynamics**. It intentionally does not model biological cells or impose
meanings on state-vector coordinates. A node holds a finite vector, edges can
hold another finite vector, and shared local rules transform an incoming
permutation-invariant message sum plus external input.

## Current scope

- Batched synchronous, bounded integration with a continuous-time step `dt`.
- Replaceable `NodeRule` and `EdgeRule` interfaces; rules receive no node IDs,
  global score, future signal, or hidden environment state.
- Reproducible directed random graphs and index-addressed input/noise streams.
- Trajectory recording plus input-shift, impulse, node-lesion, and weight-noise
  disturbances.
- Metrics for boundedness, non-silence, saturation, activity diversity,
  disturbance response, and recovery.
- Two fixed references: a one-coordinate RNN and a hand-designed stabilizing
  rule. They are comparison points, not an evolutionary system.

Evolution and an ecological environment are deliberately out of scope.

## Run the comparison

No third-party runtime dependencies are required. From the repository root:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m evolvable_state_network.cli --output experiment_output
```

The output directory contains `metrics.json`, one SVG trajectory plot per
baseline, `dashboard_data.json`, and a dependency-free browser dashboard.

To use the interactive replay, serve the output directory and open the shown
URL (the dashboard automatically loads the adjacent experiment data):

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m evolvable_state_network.server --output experiment_output
```

Do not use `python -m http.server` when you need **New experiment**: it can
serve replay files but does not provide the local experiment API.

Then open `http://localhost:8000/dashboard/`. The dashboard supports wall-clock
playback, stepping, speed and loop controls, baseline/batch/coordinate choices,
and click-to-inspect node and edge state, external input, graph connections,
metrics, and active disturbances. The **New experiment** panel runs a fresh
parameterized simulation through this local server and loads it directly into
the replay view. It can also load any compatible
`dashboard_data.json` through its file picker, which is useful when opening the
HTML directly rather than running a local server.

## Extending the substrate

Implement `NodeRule` and optionally `EdgeRule` from `rules.py`. Keep the rule
local: it gets its own vector, aggregate, external vector, `dt`, and the update
bound only. The simulator independently enforces per-coordinate delta and
absolute-state bounds as a numerical safety layer.

The metrics currently use coordinate zero as a documented observation
convention. It is not a semantic claim; later experiments may add registered
readouts while keeping rule and simulator interfaces unchanged.

## Verify

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests include dead, saturated, synchronized, and exploding trajectories.
