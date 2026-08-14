#!/usr/bin/env bash
#SBATCH --job-name=esn-embodied
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --signal=B:TERM@120

set -euo pipefail

cd /path/to/evolvable-state-network
source .venv/bin/activate

# Use durable shared or scratch storage; retain it to use a checkpoint as a
# continuation source in a later run.
esn-embodied \
  --config examples/embodied-cpu.yaml \
  --data-dir "${SCRATCH:?SCRATCH must be set}/esn-results" \
  --run-id "$SLURM_JOB_ID"
