#!/usr/bin/env bash
#
# Step 1 of 5 -- labels, video clips and the base (video-only) prompts, plus the 2a / 2b setup CSVs.
#
#   1. 2a / 2b setup      valid_steps_summary.py -> build_experiment2a.py -> build_experiment2b.py
#                         (rewrite the two basic_experiment_setup.csv from scratch; the old ones are backed up)
#   2. 1a / 1b            process_data_pipeline.py --base-only: label CSVs -> truncated clips -> base prompts
#   3. 2a / 2b            truncate videos, base prompts
#   4. graph setup        build_2a_graph_setup.py for 2a and 2b
#
# Next: 02_chat.sh, then 03_prompts.sh.
#
# Inputs it expects in $KOLAB_DATA_DIR (default <repo>/data): new_annotated_graphs/, workflow_docs/,
# session<N>_videos/, final_participants_for_iclr.csv. Needs ffmpeg.
#
# Usage:
#   scripts/01_prep.sh                  # everything
#   scripts/01_prep.sh --dry-run        # print the commands only
#   scripts/01_prep.sh --skip-setup     # skip stage 1 (do not rebuild the 2a / 2b setup CSVs)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SKIP_SETUP=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=1 ;;
        --skip-setup) SKIP_SETUP=1 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown argument: $arg (see --help)" >&2; exit 2 ;;
    esac
done
cd "$CODE_DIR"

if [[ $SKIP_SETUP -eq 0 ]]; then
    echo "== 1. 2a / 2b setup CSVs"
    for exp in experiment2a experiment2b; do
        csv="$KOLAB_DATA_DIR/extracted_timestamps/$exp/basic_experiment_setup.csv"
        [[ -f "$csv" ]] && run cp -p "$csv" "$csv.bak"
    done
    run "$PYTHON" experiment_setup/rating_audit/valid_steps_summary.py
    run "$PYTHON" experiment_setup/build_experiment2a.py
    run "$PYTHON" experiment_setup/build_experiment2b.py
fi

echo "== 2. experiments 1a / 1b: labels, clips, base prompts"
for exp in 1a 1b; do
    run "$PYTHON" process_data_pipeline.py --setting "$exp" --base-only
done

echo "== 3. experiments 2a / 2b: clips and base prompts"
for exp in experiment2a experiment2b; do
    run "$PYTHON" truncation/truncate_videos_for_setting.py --setting "$exp"
    run "$PYTHON" prompts/prompt_maker_setting.py --setting "$exp"
done

echo "== 4. graph setup for 2a / 2b"
for exp in 2a 2b; do
    run "$PYTHON" experiment_setup/build_2a_graph_setup.py --experiment "$exp"
done
