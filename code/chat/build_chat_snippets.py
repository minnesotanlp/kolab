"""
Stage 3 of the Experiment 2a chat ablation (see run_chat_pipeline.sh).

For every step of annotated_graph_experiment_setup.csv whose session has an aligned
chat (align_chats.py), cuts the part of the chat that belongs to the step and writes it
to extracted_timestamps/experiment2a/chat_snippets.csv (experiment_id, chat_text).

Window of a step, on the session video's clock (the clock of the aligned chat and of
the node times in 2a_graphs/<experiment_id>.json):
  - each graph segment (a step revisited later has several) gives one window: the
    start of its first node to the end of its last node, strictly, no buffer;
  - the step's window is the union of those, so the gap between revisits is excluded.
A chat row is kept only if its own video_ts falls inside a window. Only rows with chat
text are kept (tool-use rows have none), as just two roles -- Human, and AI (the chat
file's "Claude") -- with no timestamps. Each message is wrapped in a tag so where it starts
and ends is unambiguous even when a reply is long markdown:

    <human>
    [attached: notes.pdf] <message>
    </human>

    <ai>
    <reply>
    </ai>

Rows already in chat_snippets.csv for other steps are kept; steps rebuilt now replace
their row. A report of empty windows / chat outside the window is written to
chat_snippets_report.txt.

Usage:
    python build_chat_snippets.py
    python build_chat_snippets.py --p p1 --session 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import csv
import json

from chat.chat_common import (
    ANNOTATED_SETUP_CSV, CHAT_SNIPPETS_CSV, CHAT_SNIPPETS_REPORT, ROLE_LABEL, aligned_chat_path,
    format_chat, format_hms, warn,
)
from prompts.prompt_maker_setting import GRAPHS_2A_DIR, split_time_range, to_seconds

def node_ranges(node):
    """[(start_s, end_s), ...] of one node's time field (a string or a list of strings)."""
    field = node.get("time")
    if not field:
        return []
    ranges = [field] if isinstance(field, str) else list(field)
    out = []
    for r in ranges:
        try:
            start, end = split_time_range(r)
            out.append((to_seconds(start), to_seconds(end)))
        except ValueError:
            continue
    return out


def step_windows(subgraph):
    """One (start_s, end_s) per graph segment: first node's start .. last node's end."""
    segments = []
    for seg in subgraph.get("segments", []):
        try:
            start, end = split_time_range(seg["time_range"])
            segments.append((to_seconds(start), to_seconds(end)))
        except (KeyError, ValueError):
            continue

    windows = {}
    for node in subgraph.get("nodes", []):
        for start, end in node_ranges(node):
            if segments:
                # the segment containing the node's start, else the nearest one
                idx = min(
                    range(len(segments)),
                    key=lambda i: 0 if segments[i][0] <= start <= segments[i][1]
                    else min(abs(start - segments[i][0]), abs(start - segments[i][1])),
                )
            else:
                idx = 0
            lo, hi = windows.get(idx, (start, end))
            windows[idx] = (min(lo, start), max(hi, end))
    return [windows[i] for i in sorted(windows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", default=None, help="Only this participant id, e.g. p1")
    parser.add_argument("--session", default=None, choices=["1", "2"], help="Only this session")
    args = parser.parse_args()

    with open(ANNOTATED_SETUP_CSV, newline="", encoding="utf-8") as f:
        setup_rows = list(csv.DictReader(f))

    aligned_cache = {}
    built = {}
    report = []
    for row in setup_rows:
        experiment_id = row["experiment_id"].strip()
        participant_id = row["participant_id"].strip().lower()
        session_id = row["session_id"].strip().lower()
        if args.p and participant_id != args.p.strip().lower():
            continue
        if args.session and session_id != f"session{args.session}":
            continue

        path = aligned_chat_path(participant_id, session_id)
        if not path.is_file():
            continue  # session not aligned (yet): nothing to build
        if (participant_id, session_id) not in aligned_cache:
            with open(path, newline="", encoding="utf-8") as f:
                aligned_cache[(participant_id, session_id)] = list(csv.DictReader(f))
        chat_rows = aligned_cache[(participant_id, session_id)]

        graph_path = GRAPHS_2A_DIR / (row.get("graph_filename", "").strip() or f"{experiment_id}.json")
        if not graph_path.is_file():
            warn(f"missing graph {graph_path.name}; skipping {experiment_id}.")
            continue
        windows = step_windows(json.loads(graph_path.read_text(encoding="utf-8")))
        if not windows:
            warn(f"no node times in {graph_path.name}; skipping {experiment_id}.")
            continue

        kept = [
            r for r in chat_rows
            if r["speaker"].strip().lower() in ROLE_LABEL and r["chat"].strip()
            and any(lo <= float(r["video_ts_seconds"]) <= hi for lo, hi in windows)
        ]
        built[experiment_id] = format_chat(kept)
        window_text = ", ".join(f"{format_hms(lo)}-{format_hms(hi)}" for lo, hi in windows)
        line = f"{experiment_id}: {len(kept)} chat row(s) in {window_text}"
        chat_start, chat_end = float(chat_rows[0]["video_ts_seconds"]), float(chat_rows[-1]["video_ts_seconds"])
        if not kept:
            overlaps = any(hi >= chat_start and lo <= chat_end for lo, hi in windows)
            line += "  ** EMPTY" + ("" if overlaps else " (window outside the chat's time span)")
        report.append(line)

    if not built:
        print("No steps to build (no aligned chats yet, or nothing matched the filters).")
        return

    existing = {}
    if CHAT_SNIPPETS_CSV.is_file():
        with open(CHAT_SNIPPETS_CSV, newline="", encoding="utf-8") as f:
            existing = {r["experiment_id"]: r["chat_text"] for r in csv.DictReader(f)}
    existing.update(built)
    with open(CHAT_SNIPPETS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_id", "chat_text"])
        for experiment_id, text in existing.items():
            writer.writerow([experiment_id, text])

    CHAT_SNIPPETS_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    empty = sum(1 for t in built.values() if not t)
    print("\n".join(report))
    print(f"Wrote {len(built)} step(s) ({empty} empty) to {CHAT_SNIPPETS_CSV} ; report: {CHAT_SNIPPETS_REPORT.name}")


if __name__ == "__main__":
    main()
