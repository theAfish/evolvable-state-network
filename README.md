# Evolvable State Network

## Embodied-environment task module

`evolvable_state_network.environments` now contains an optimizer-free,
continuous predator–prey–plant world imported from the former `evo-eco`
prototype.  It owns world state, observations, ecological events, and controller
lifecycle only. Its generic transition reward is always zero; it does **not**
import ESN evolution, CMA-ES, NEAT, or any
selection/mutation code.  This package can therefore be extracted as a
standalone environment library.

`evolvable_state_network.evolution` remains task-agnostic and can likewise be
extracted independently.  The only join is
`evolvable_state_network.tasks.food_web.FoodWebTaskEvaluator`, which decodes a
node-rule genome into a recurrent controller and evaluates it over repeatable
food-web episodes.  New environments and learning tasks should add another
module under `tasks/`, rather than importing environments from `evolution`.

### Embodied rule learning

`evolvable_state_network.embodied` implements the next-stage architecture.
An environment-specific `AgentAdapter` normalises observations onto fixed input
nodes and maps one `[-1, 1]` output per action node into the environment's
action space. `EmbodiedNetwork` then creates a fresh random graph and fresh
random node states for every agent/episode; only the node and edge update-rule
genome is inherited. The food-web implementation in
`tasks.embodied_food_web` evaluates prey and predator update rules on matched
random worlds, and
can start CMA-ES from an exported joint rule genome via
`EmbodiedRuleEvolutionConfig(initial_genome=...)`.

Food-web agents sense hunger, signed energy change, whether they ate on the
previous world tick, and elapsed starvation time as ordinary body channels.
No optimizer reward or previous action is injected into the network; retaining
action or body history is the job of recurrent node and edge state. Immutable
boundary-role tags distinguish sensor and actuator meanings without becoming
heritable parameters. Vision is a nine-pixel ego-relative ray image with
separate plant, prey, and predator proximity channels per pixel; pixel order
encodes left-to-right direction and ray angles are not network inputs. Random
graphs and initial states remain fresh, but batch
candidates are compared on matched scenario realizations independent of their
genomes. Evolution maximizes only the first life's duration, restricted at the
episode horizon; meals, hunger, energy, movement, and ecological events are
diagnostics and never selection terms. One disjoint fixed seed bank selects the
archive; a second seed bank is not touched until that winner is frozen and
supplies the final lifetime estimate.
The final pass also evaluates a neutral zero rule and a causal intervention
that masks every ray pixel. Reports include lifetime gain over neutral,
horizon survival, correctly aggregated completed lifetimes, deaths, and
diagnostic body/behavior measurements.

Default food regrowth occurs around persistent patches, for which a simple
ray-guided policy materially outperforms blind straight-line sweeping. Setting
`plant_cluster_count=0` restores uniform food for deliberate ablations. Batch
requests enforce a horizon of at least three configured no-food lifetimes and
enough theoretical plant regrowth to sustain all prey by default; these guards
can be explicitly disabled for infeasible or short-horizon ablations. The
default embodied network has 64 nodes, leaving 31 anonymous
hidden nodes after the 31 sensor and two actuator boundary nodes.

Embodied training has two independent acceleration controls. The `torch`
execution backend keeps agent node/edge state in persistent tensors and can
run on `cpu`, `cuda`, or `auto`; the reference `python` backend remains
available for comparisons. Batch evolution can also evaluate candidate
genomes in reusable spawned processes with `workers=0` selecting up to eight
CPU workers automatically. CPU workers preserve candidate order, common seed
banks, and frozen opponents. CUDA intentionally uses one process to avoid
duplicating GPU contexts; for the default small 64-node agents, vectorized
Torch CPU plus several candidate workers is normally faster than CUDA.

In the web UI, a completed embodied run can also be selected as a continuation
starting point. Its prey and predator best genomes seed their respective new
online libraries; the newly chosen population, world, and network settings are
then used for the new run. This restarts optimization from the learned rules;
it does not restore an in-progress CMA-ES optimizer state.

Each embodied run writes an atomic `checkpoint.json` when live progress is
reported (the first tick, then every four ticks, and the final tick). It holds
the current best genome for each species and can be selected as a continuation
source even if the original run has not completed.

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
  latents take bounded `scale * tanh(local_rule(...))` increments; smooth
  bounded coordinate-wise gates modulate a fixed projected source message.
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
the project-local `.outputs/` directory by default.

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

Start the FastAPI application. Generated runs are stored in `.outputs/` under
the current directory:

```powershell
esn-dashboard
```

The default port is `8000`, but it can be changed if another application is
using it:

```powershell
esn-dashboard --port 8001
```

For an environment-based setting, use `$env:ESN_PORT = "8001"` before running
`esn-dashboard`. The `--port` argument takes precedence over `ESN_PORT`.
Without an override, use `http://127.0.0.1:8000/` and
`http://127.0.0.1:8000/docs`; with the example override, use port `8001` for
both URLs. The old `/dashboard/` URL redirects to `/`.

To keep results in another local folder, set it once for the server process:

```powershell
esn-dashboard --data-dir .\my-results
```

Alternatively set `$env:ESN_DATA_DIR = "D:\research\state-network-results"`
before starting the dashboard. This is a server-level storage setting, not a
required per-run output argument.

The **Survival** tab now has two distinct actions. `Run 80-tick smoke test`
checks that the asynchronous machinery works. `Start survival training` runs a
configured experiment and stops at either the requested completed candidate-life
budget or the safety tick limit. A candidate life is one genome evaluated across
the selected number of seeded replicas; CMA-ES is updated only after exact death
or graduation outcomes form a comparable result batch. The page explains the
learning state, stop reason, completed replica trajectories, curriculum passage,
death causes, and every archived candidate in plain language.

Every completed run also persists `elite_archive.json`. The **Live graph** tab
lists up to the configured elite count from each survival run, including older
runs whose elites can be reconstructed from their candidate archive. Selecting
an elite decodes the trained joint node-and-edge genome and runs it continuously
on a newly configured graph. Legacy `best_genome.json` exports remain available
as explicitly labelled comparison models.

The Live model chooser defaults to the best trained elite from each run. It can
also show every survival elite or only legacy comparisons. Survival models are
globally ordered by the same lexicographic evidence used during evolution:
curriculum stage, demonstrated function across replicas, lifetime, lower
worst-replica pathology burden, response, propagation, recovery, and paired-probe
distinguishability. The detail panel exposes these values and does not collapse
them into an arbitrary scalar score.

FastAPI serves the packaged frontend and artifact files. Uvicorn runs the ASGI
application and Pydantic validates request bodies. `GET /api/health` reports
the active storage path.

The dashboard's **Survival** archive exposes a `Replay this exact life` control
for every completed candidate replica. It reconstructs the archived genome on
that replica's saved graph, initial state, input stream, probes, and curriculum
disturbances, then stops at the archived death or graduation age. It is not a
fixed 80-step validation window. The older **Legacy comparison** view retains
its generation-based plots but no longer supplies the interactive replay chooser.

The dashboard supports wall-clock playback, stepping, speed and loop controls,
baseline/batch/coordinate choices, and click-to-inspect node and edge state,
external input, graph connections, metrics, and active disturbances. The **New experiment** panel runs a fresh
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

## Development map

The backend is organized by responsibility:

```text
evolvable_state_network/
├── application/       request models, configuration mapping, runtime services
├── evolution/         candidates, genomes, evaluation, CMA-ES, search runners
├── simulation/        reference engine and optimized Torch backend
├── api.py             HTTP route composition
├── experiment.py      fixed-rule experiment use case
└── web/               packaged browser assets
```

New request fields belong in `application/models.py`; their mapping to research
configuration belongs in `application/configuration.py`. New search behavior
belongs under `evolution/`, and numerical stepping belongs under `simulation/`.
Each should receive direct unit tests before it is exposed through the API.

## Verify

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests include dead, saturated, synchronized, and exploding trajectories.
