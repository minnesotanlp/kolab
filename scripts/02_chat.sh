#!/usr/bin/env bash
#
# Step 2 of 5 -- add the human <-> AI chat: anchors, alignment to the video clock, per-step snippets.
#
#   1. build_chat_anchors.py    extracted_timestamps/chat_anchors.csv (first human message + video start of the
#                               session). Stops with "nothing to build" when nothing is new. Edit video_first_mmss
#                               by hand to override a value (non-blank values are never overwritten).
#   2. align_chats.py           chat wall-clock -> video time, into session<S>_chats_aligned/
#   3. build_chat_snippets.py   per step, the chat inside the step's graph window -> experiment2a/chat_snippets.csv
#
# Needs 01_prep.sh first (build_chat_snippets.py reads the annotated setup CSV it writes). Next: 03_prompts.sh.
# Re-running is safe: it only builds sessions that are not built yet.
#
# Usage:
#   scripts/02_chat.sh
#   scripts/02_chat.sh --dry-run      # print the commands; only the anchor check actually runs
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg (see --help)" >&2; exit 2 ;;
    esac
done
cd "$CODE_DIR"

echo "== 1. chat anchors"
if [[ $DRY_RUN -eq 1 ]]; then
    run "$PYTHON" chat/build_chat_anchors.py --dry-run
    exit 0
fi
rc=0
run "$PYTHON" chat/build_chat_anchors.py || rc=$?
if [[ $rc -eq 10 ]]; then
    echo "Nothing new to build."; exit 0     # exit code 10 = nothing to build, not an error
elif [[ $rc -ne 0 ]]; then
    exit "$rc"
fi

echo "== 2. align chats to the video clock"
run "$PYTHON" chat/align_chats.py

echo "== 3. cut the chat per step"
run "$PYTHON" chat/build_chat_snippets.py
