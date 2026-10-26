#!/usr/bin/env bash
#
# Step 4 of 5 -- run the models. A thin wrapper over code/inference/run_experiments.py: every argument is passed
# straight through. Everything runs in ONE process, so a Qwen model is loaded once and reused.
#
#   --exp1a --exp1b --exp1c_last1 --exp1c_last2 --exp1c_last3 --exp1c_full --exp1d --exp2a --exp2b
#                       each takes modes (or `all`); omit a flag to skip that experiment.
#                       Modes: video-only video-graph graph-only video-chat chat-only graph-chat video-graph-chat
#                       (2a / 2b also: video-graph-ts graph-only-ts)
#   --models M [M ...]  gemini flash gpt qwen qwen32b   (required)
#   --runs N [N ...]    run numbers (repeats every job; writes ..._results_run<N>.csv)
#   --all               every matching sample (default: only 1 sample per experiment/mode, a smoke test)
#   --dry-run           show what would be sent; no API calls, no model load, nothing written
#   (also --limit, --p, --session, --exp-id, --max_frames, --frames; see run_experiments.py --help)
#
# Needs 01 to 03 first. Results: <data>/experiment_results/<folder>/<model>_<folder>_results[_run<N>].csv.
# Rows already in a results file are skipped, so an interrupted run can be started again.
# gemini / flash / gpt call a paid API and read AIGATEWAY_API_KEY from the environment.
#
# Examples:
#   scripts/04_run.sh --models flash --exp1a video-only graph-only --dry-run
#   scripts/04_run.sh --models qwen --exp1a all --exp1b all --exp2a all --exp2b all --all --runs 1 2 3
#   scripts/04_run.sh --models gpt --exp1d chat-only --all --runs 1
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
cd "$CODE_DIR"
run "$PYTHON" inference/run_experiments.py "$@"
