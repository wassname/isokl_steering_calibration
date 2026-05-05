#!/usr/bin/env bash
# Queue all sweep cells via pueue. Each cell gets its own job with why/resolve.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen3-4B-Instruct-2507"
)
METHODS=("mean_diff" "directional_ablation" "pca")
SEEDS=(0 1 2)
WINDOWS=(20 50)

# Priority: small models first so the figure starts populating; 4B last.
prio_for_model() {
  case "$1" in
    *0.5B*) echo 30 ;;
    *1B*)   echo 20 ;;
    *4B*)   echo 10 ;;
    *)      echo 5  ;;
  esac
}

n=0
for model in "${MODELS[@]}"; do
  prio=$(prio_for_model "$model")
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for window in "${WINDOWS[@]}"; do
        run_id="$(basename "$model")_${method}_s${seed}_w${window}"
        if [ -f "outputs/${run_id}/calib.json" ]; then
          echo "skip ${run_id} (already done)"; continue
        fi
        label="why: stability of iso-KL calib for ${method} on ${model} (seed ${seed}, T=${window}); resolve: include cell in figure1 if it converges, else flag as bracket-pinned"
        pueue add -w "$ROOT" -o "$prio" -l "$label" -- \
          uv run --extra all python scripts/run_cell.py \
            --model "$model" --method "$method" --seed "$seed" --window "$window" \
            --run-id "$run_id"
        n=$((n+1))
      done
    done
  done
done
echo "queued ${n} jobs"
echo "monitor: pueue status | head -40"
echo "aggregate after queue drains: just aggregate"
