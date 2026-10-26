#!/usr/bin/env bash
#
# Step 3 of 5 -- the graph / chat ablation prompts, and experiments 1c and 1d.
#
#   1. 1a / 1b ablations   prompt_maker_setting.py --setting experiment<1a|1b>_<mode>   (6 modes each)
#   2. 2a / 2b ablations   prompt_maker_setting.py --setting experiment<2a|2b>_<mode>   (8 modes each)
#   3. 1d                  build_experiment1d.py: whole workflow, 0:00 to the target AI action
#                          (instances, clips, setup CSV, prompts for every mode)
#   4. 1c                  build_experiment1c.py: context levels (last 1 / 2 / 3 steps, full), clips at 1 fps
#
# Needs 01_prep.sh and 02_chat.sh first. Existing prompt files are skipped. Next: 04_run.sh.
# Experiment 1c cuts new clips; set KOLAB_CLIP_DIR to put them on a disk with room
# (default <data>/clips).
#
# Usage:
#   scripts/03_prompts.sh
#   scripts/03_prompts.sh --dry-run     # print the commands only
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg (see --help)" >&2; exit 2 ;;
    esac
done
cd "$CODE_DIR"

MODES_1=(video_graph graph_only video_chat chat_only graph_chat video_graph_chat)
MODES_2=(video_graph video_graph_ts graph_only graph_only_ts video_graph_chat video_chat graph_chat chat_only)

echo "== 1. experiments 1a / 1b ablation prompts"
for exp in experiment1a experiment1b; do
    for mode in "${MODES_1[@]}"; do
        run "$PYTHON" prompts/prompt_maker_setting.py --setting "${exp}_${mode}"
    done
done

echo "== 2. experiments 2a / 2b ablation prompts"
for exp in experiment2a experiment2b; do
    for mode in "${MODES_2[@]}"; do
        run "$PYTHON" prompts/prompt_maker_setting.py --setting "${exp}_${mode}"
    done
done

echo "== 3. experiment 1d"
run "$PYTHON" labels/build_experiment1d.py

echo "== 4. experiment 1c"
run "$PYTHON" labels/build_experiment1c.py
