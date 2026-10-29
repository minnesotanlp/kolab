"""
Stage 1 of the Experiment 2a chat ablation (see run_chat_pipeline.sh).

Makes/extends extracted_timestamps/chat_anchors.csv: one row per (participant, session)
that has a behavior graph in annotated_graphs/ (every session, not only the ones with a
rating row in Experiment 2a's annotated_graph_experiment_setup.csv -- a session can be
fully annotated for experiment1a/experiment1b without ever being part of 2a's rated-steps data,
e.g. p6 session1), with

  participant_id, session_id, chat_first_text, chat_first_ts, video_first_mmss

  chat_first_text   the session's first Human chat message (whitespace collapsed, cut to
                    200 characters) -- to recognise it in the video.
  chat_first_ts     wall-clock timestamp of that message.
  video_first_mmss  where that first exchange sits in the session video: the START of the
                    earliest AI node of annotated_graphs/session<S>_<p>_behavior_graph*.json,
                    as mm:ss (hh:mm:ss from one hour on). Filled in automatically; edit it by
                    hand if a video shows the message at a different time -- a non-blank
                    value is never overwritten. If the graph has no AI node it stays blank
                    and that session waits for you.

align_chats.py turns (chat_first_ts, video_first_mmss) into a per-session clock offset.

Existing values are never rewritten; only new sessions are appended and blank
video_first_mmss cells are filled. A file written before the chat_first_text column
existed is upgraded in place.

Also the gate for the pipeline: prints what is pending and exits with

    0   at least one session is ready to build (anchor filled, output missing)
    10  nothing to run (every session is done, or only waiting on a blank anchor)

Usage:
    python build_chat_anchors.py             # append new sessions, fill anchors, print status
    python build_chat_anchors.py --dry-run   # print status, write nothing
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import csv
import sys

from chat.chat_common import (
    ANCHOR_FIELDS, CHAT_ANCHORS_CSV, annotated_sessions, chat_files, first_ai_node_start,
    format_hms, read_anchors, read_chat_rows, session_status, warn,
)

TEXT_MAX_CHARS = 200


def first_human_message(participant_id, session_id):
    """(timestamp, text) of the session's first Human row, or None."""
    for row in read_chat_rows(participant_id, session_id):
        if row["speaker"].strip().lower() == "human":
            text = " ".join(row["chat"].split())
            if len(text) > TEXT_MAX_CHARS:
                text = text[:TEXT_MAX_CHARS].rstrip() + "..."
            return row["timestamp"].strip(), text
    return None


def format_anchor(seconds):
    """mm:ss, or hh:mm:ss from one hour on."""
    hms = format_hms(seconds)
    return hms[3:] if hms.startswith("00:") else hms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print status, write nothing")
    args = parser.parse_args()

    pairs = annotated_sessions()

    rows = read_anchors()
    old_layout = bool(rows) and "chat_first_text" not in rows[0]
    known = {(r["participant_id"], r["session_id"]) for r in rows}
    added = 0
    for participant_id, session_id in pairs:
        if (participant_id, session_id) in known:
            continue
        files = chat_files(participant_id, session_id)
        if not files:
            warn(f"no chat file for {participant_id} {session_id}; skipping.")
            continue
        if len(files) > 1:
            warn(f"{participant_id} {session_id}: using {len(files)} split chat files: {[f.name for f in files]}")
        first_message = first_human_message(participant_id, session_id)
        if first_message is None:
            warn(f"no Human row in the chat of {participant_id} {session_id}; skipping.")
            continue
        rows.append(
            {"participant_id": participant_id, "session_id": session_id,
             "chat_first_text": first_message[1], "chat_first_ts": first_message[0], "video_first_mmss": ""}
        )
        added += 1

    filled = 0
    for row in rows:
        if old_layout and not row.get("chat_first_text"):
            message = first_human_message(row["participant_id"], row["session_id"])
            row["chat_first_text"] = message[1] if message else ""
        if (row.get("video_first_mmss") or "").strip():
            continue
        start = first_ai_node_start(row["participant_id"], row["session_id"])
        if start is None:
            warn(f"{row['participant_id']} {row['session_id']}: no AI node in annotated_graphs; video_first_mmss left blank.")
            continue
        row["video_first_mmss"] = format_anchor(start)
        filled += 1

    if (added or filled or old_layout) and not args.dry_run:
        with open(CHAT_ANCHORS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ANCHOR_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    prefix = "would be " if args.dry_run else ""
    print(f"{added} new anchor row(s) and {filled} video_first_mmss value(s) {prefix}written to {CHAT_ANCHORS_CSV.name}.")

    # Status is read from the file, so under --dry-run it reflects the file as it is now.
    status = session_status()
    waiting = [f"{p} {s}" for (p, s), st in status.items() if st == "needs_anchor"]
    ready = [f"{p} {s}" for (p, s), st in status.items() if st == "needs_build"]
    no_chat = [f"{p} {s}" for (p, s), st in status.items() if st == "no_chat"]
    if no_chat:
        print(f"No chat file (skipped): {', '.join(no_chat)}")
    if waiting:
        print(f"Waiting on an anchor (fill video_first_mmss in {CHAT_ANCHORS_CSV}): {', '.join(waiting)}")
    if ready:
        print(f"Ready to build: {', '.join(ready)}")
    if not ready:
        print("Nothing to run.")
        sys.exit(10)


if __name__ == "__main__":
    main()
