#!/bin/bash
#SBATCH --job-name=elongation_ev
#SBATCH --partition=amd_a8_384
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=/publicfs01/fs1-a8/home/a8s002767/elongation_ev
#SBATCH --output=elongation_%j.out
#SBATCH --error=elongation_%j.err

# Supply --time when submitting. The commands below read the partition limit;
# they query Slurm and submit the simulation without a preliminary test run.
#
# ev_time_limit=$(sinfo -h -p amd_a8_384 -o '%l' | head -n 1)
# case "$ev_time_limit" in
#   infinite|INFINITE|unlimited|UNLIMITED) ev_time_limit=0 ;;
# esac
# test -n "$ev_time_limit" && sbatch --time="$ev_time_limit" submit_elongation_ev.sh

set -eo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate base

# The current solver has no chain-level multiprocessing. Keep the BLAS
# backend single-threaded for the many small linear systems in each step.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

printf 'Job: %s\nHost: %s\nStarted: %s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
printf 'Python: %s\nTime limit: %s\n' \
  "$CONDA_PREFIX/bin/python" "${SLURM_TIMELIMIT:-scheduler-controlled}"

srun --ntasks=1 --cpus-per-task=1 \
  "$CONDA_PREFIX/bin/python" -u main_elongation_ev.py

printf 'Finished: %s\n' "$(date -Is)"
