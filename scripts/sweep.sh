#!/usr/bin/env bash
# Sweep: model x method x seed x window. Edit the lists to taste.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "meta-llama/Llama-3.2-1B-Instruct")
METHODS=("mean_diff" "directional_ablation" "pca")
SEEDS=(0 1 2)
WINDOWS=(20 50)

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for window in "${WINDOWS[@]}"; do
        run_id="$(basename "$model")_${method}_s${seed}_w${window}"
        if [ -f "outputs/${run_id}/calib.json" ]; then
          echo "skip ${run_id}"; continue
        fi
        echo "=== ${run_id} ==="
        uv run --extra all python scripts/run_cell.py \
          --model "$model" --method "$method" --seed "$seed" --window "$window"
      done
    done
  done
done

uv run --extra all python scripts/aggregate.py --runs-root outputs --out figs
