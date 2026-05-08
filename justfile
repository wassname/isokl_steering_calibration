set shell := ["bash", "-cu"]

default:
    @just --list

# Smoke: tiny-random Llama, all 3 methods, asserts nonzero KL + branch-pmass changes with coeff.
smoke:
    BEARTYPE=1 uv run --extra all pytest -q tests/test_smoke.py

test:
    uv run --extra all pytest -q

# Run one (model, method, seed, window) cell end-to-end (calibrate + trajectory + pmass).
cell model="Qwen/Qwen3.5-0.8B" method="mean_diff" seed="0" window="512":
    uv run --extra all python scripts/run_cell.py \
        --model {{model}} --method {{method}} --seed {{seed}} --window {{window}}

# Sweep model x method x seed x window cells (sequential bash).
sweep:
    bash scripts/sweep.sh

# Queue all sweep cells via pueue (one job per cell, priority: small models first).
queue:
    bash scripts/queue_sweep.sh

# Show pueue status and a tail of each running job.
queue-status:
    pueue status
    pueue log -l 15

# Aggregate all outputs/<run_id>/ into figs/figure1.png + figs/table.md.
aggregate:
    uv run --extra all python scripts/aggregate.py --runs-root outputs --out figs
