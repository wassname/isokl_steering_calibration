set shell := ["bash", "-cu"]

default:
    @just --list

# Smoke: tiny-random Llama, all 3 methods, asserts nonzero KL + branch-pmass changes with coeff.
smoke:
    BEARTYPE=1 uv run --extra all pytest -q tests/test_smoke.py

test:
    uv run --extra all pytest -q

# Run one (model, method, seed, window) cell end-to-end (calibrate + trajectory + pmass).
cell model="Qwen/Qwen2.5-0.5B-Instruct" method="mean_diff" seed="0" window="50":
    uv run --extra all python scripts/run_cell.py \
        --model {{model}} --method {{method}} --seed {{seed}} --window {{window}}

# Sweep model x method x seed x window cells.
sweep:
    bash scripts/sweep.sh

# Aggregate all outputs/<run_id>/ into figs/figure1.png + figs/table.md.
aggregate:
    uv run --extra all python scripts/aggregate.py --runs-root outputs --out figs
