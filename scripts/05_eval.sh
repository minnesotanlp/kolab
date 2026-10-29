#!/usr/bin/env bash
#
# Step 5 of 5 -- compute the metrics from the result CSVs, for every experiment and condition found.
#
# Runs code/evaluation/eval_results.py once per (experiment, model): accuracy / F1 for 1a-1d, rating metrics
# for 2a / 2b. With several run files it prints each run and the mean +/- std.
#
# Usage:
#   scripts/05_eval.sh gemini qwen              # models to evaluate (gemini flash gpt qwen qwen32b)
#   scripts/05_eval.sh gemini --only 1a 1b      # restrict to some experiments
#   scripts/05_eval.sh gemini --dry-run         # print the commands only
# Experiments with no result files for a model are reported and skipped.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODELS=()
ONLY=()
mode=models
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --only)    mode=only ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown argument: $arg (see --help)" >&2; exit 2 ;;
        *) if [[ $mode == only ]]; then ONLY+=("$arg"); else MODELS+=("$arg"); fi ;;
    esac
done
[[ ${#MODELS[@]} -gt 0 ]] || { usage; exit 2; }
cd "$CODE_DIR"

EXPERIMENTS=(1a 1b 1c_last1 1c_last2 1c_last3 1c_full 1d experiment2a experiment2b)
[[ ${#ONLY[@]} -gt 0 ]] && EXPERIMENTS=("${ONLY[@]}")

for exp in "${EXPERIMENTS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "== $exp / $model"
        run "$PYTHON" evaluation/eval_results.py --setting "$exp" --model "$model" \
            || echo "   (no results found for $exp / $model, skipped)"
    done
done
