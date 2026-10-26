"""
Shared helpers for the Experiment 2a chat ablation (build_chat_anchors.py,
align_chats.py, build_chat_snippets.py). See run_chat_pipeline.sh for the flow.

Chat CSVs (extracted_timestamps/session<S>_chats/) carry wall-clock UTC
timestamps for a WHOLE session; steps and graph nodes are on the session
video's clock. chat_anchors.csv holds, per (participant, session), the
first human message's wall-clock time and the mm:ss at which it appears in the
video, which gives the offset between the two clocks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import EXTRACTED_DIR, ROOT_DIR
ANNOTATED_GRAPHS_DIR = ROOT_DIR / "new_annotated_graphs"  # the only graph source (bash_scripts/sync_graphs.sh)
ANNOTATED_SETUP_CSV = EXTRACTED_DIR / "experiment2a" / "annotated_graph_experiment_setup.csv"
CHAT_ANCHORS_CSV = EXTRACTED_DIR / "chat_anchors.csv"
CHAT_SNIPPETS_CSV = EXTRACTED_DIR / "experiment2a" / "chat_snippets.csv"
CHAT_SNIPPETS_REPORT = EXTRACTED_DIR / "experiment2a" / "chat_snippets_report.txt"

CHAT_PROMPT_DIRS = [
    EXTRACTED_DIR / "prompts" / name
    for name in (
        "experiment2a_video_graph_chat", "experiment2a_video_chat", "experiment2a_graph_chat", "experiment2a_chat_only",
        "experiment2b_video_graph_chat", "experiment2b_video_chat", "experiment2b_graph_chat", "experiment2b_chat_only",
    )
]

ROLE_LABEL = {"human": "Human", "claude": "AI"}
ROLE_TAG = {"Human": "human", "AI": "ai"}

ANCHOR_FIELDS = ["participant_id", "session_id", "chat_first_text", "chat_first_ts", "video_first_mmss"]


def chats_dir(session_id):
    return EXTRACTED_DIR / f"{session_id}_chats"


def aligned_chats_dir(session_id):
    return EXTRACTED_DIR / f"{session_id}_chats_aligned"


def aligned_chat_path(participant_id, session_id):
    return aligned_chats_dir(session_id) / f"{participant_id}_{session_id}_chats.csv"


def chat_files(participant_id, session_id):
    """The chat CSV(s) for one session, top-level of session<S>_chats/ only
    (old_format/ is never used). A "<p>_<session>_new_chats.csv" wins over
    "<p>_<session>_chats.csv"; failing both, split files "<p>_<session>_chats_<i>.csv"
    are all returned (to be concatenated). Empty list if none exist."""
    folder = chats_dir(session_id)
    for name in (f"{participant_id}_{session_id}_new_chats.csv", f"{participant_id}_{session_id}_chats.csv"):
        if (folder / name).is_file():
            return [folder / name]
    return sorted(folder.glob(f"{participant_id}_{session_id}_chats_*.csv"))


def parse_wallclock(ts):
    """ISO-8601 chat timestamp ("2026-08-18T20:21:37.633289Z") -> epoch seconds (float)."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    elif ts.upper().endswith(" UTC"):  # "2026-07-20 22:11:47 UTC", as some chat exports write it
        ts = ts[:-4].rstrip() + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_mmss(value):
    """"mm:ss" (or "hh:mm:ss", single-digit fields, stray spaces) -> seconds (int).
    Raises ValueError on anything else."""
    parts = (value or "").strip().split(":")
    if len(parts) not in (2, 3) or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"expected mm:ss (or hh:mm:ss), got {value!r}")
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + int(p)
    return seconds


def format_hms(seconds):
    seconds = max(int(round(seconds)), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def read_chat_rows(participant_id, session_id):
    """All rows of the session's chat file(s), stable-sorted by wall-clock time
    (some files have a tool row a second before the row that triggered it)."""
    rows = []
    for path in chat_files(participant_id, session_id):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: parse_wallclock(r["timestamp"]))
    return rows


def format_chat(rows):
    """Aligned chat rows as tagged messages: <human>/<ai> blocks, no timestamps."""
    blocks = []
    for row in rows:
        role = ROLE_LABEL[row["speaker"].strip().lower()]
        text = row["chat"].strip()
        attached = (row.get("file") or "").strip()
        if role == "Human" and attached:
            text = f"[attached: {attached}] {text}"
        tag = ROLE_TAG[role]
        blocks.append(f"<{tag}>\n{text}\n</{tag}>")
    return "\n\n".join(blocks)


def chat_for_window(participant_id, session_id, begin_s, end_s, rescue=False):
    """The formatted chat of one session inside (begin_s, end_s) -- seconds on the
    session video's clock, both ends exclusive -- from the aligned chat file. Only rows
    with chat text from a human or the AI are kept. Returns None if the session has no
    aligned chat (so a caller can tell "not aligned" from "no messages in the window",
    which is "").

    rescue=True (experiment 1b's own window, which ends at the AI action's own end, not at
    a later user response): if the window's last message isn't from Claude, the window
    likely closed a few seconds before Claude's reply was actually logged -- the
    annotator's video-based end mark and the chat log's message-sent time don't always
    agree exactly. In that case, look at the very next chat message after end_s and
    include it, but ONLY if that next message is itself from Claude: not any later
    Claude message (that risks pulling in an unrelated later exchange -- e.g. the AI
    action used a different tool such as Grok or a non-chat canvas, so no reply exists
    nearby at all), and not if a Human message comes first (that means the gap is real,
    not just lag)."""
    path = aligned_chat_path(participant_id, session_id)
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chat_rows = sorted(
        (r for r in rows if r["speaker"].strip().lower() in ROLE_LABEL and r["chat"].strip()),
        key=lambda r: float(r["video_ts_seconds"]),
    )
    kept = [r for r in chat_rows if begin_s < float(r["video_ts_seconds"]) < end_s]

    if rescue and not (kept and kept[-1]["speaker"].strip().lower() == "claude"):
        next_after = [r for r in chat_rows if float(r["video_ts_seconds"]) > end_s]
        if next_after and next_after[0]["speaker"].strip().lower() == "claude":
            kept = kept + [next_after[0]]

    return format_chat(kept)


def read_anchors():
    if not CHAT_ANCHORS_CSV.is_file():
        return []
    with open(CHAT_ANCHORS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def annotated_sessions():
    """(participant_id, session_id) pairs that have a graph in new_annotated_graphs/,
    e.g. session1_p13_behavior_graph_modified.json -> ("p13", "session1")."""
    pairs = set()
    for path in ANNOTATED_GRAPHS_DIR.glob("session*_[pP]*_*.json"):
        m = re.match(r"^(session\d+)_(p\d+)_", path.name, re.IGNORECASE)
        if m:
            pairs.add((m.group(2).lower(), m.group(1).lower()))
    return sorted(pairs)


def graph_file(participant_id, session_id):
    """The session's behavior-graph file in new_annotated_graphs/; None if it has none."""
    matches = sorted(ANNOTATED_GRAPHS_DIR.glob(f"{session_id}_{participant_id}_*.json"))
    return matches[0] if matches else None


def first_ai_node_start(participant_id, session_id):
    """Start (seconds, on the session video's clock) of the earliest AI node of
    new_annotated_graphs/session<S>_<p>_*.json, or None when there is no graph or
    no AI node. This is where the session's first chat exchange appears in the video."""
    from prompts.prompt_maker_setting import split_time_range, to_seconds  # lazy: heavy import

    path = graph_file(participant_id, session_id)
    if path is None:
        return None
    graph = json.loads(path.read_text(encoding="utf-8"))
    starts = []
    for step in graph.get("steps", []):
        for node in step.get("nodes", []):
            if (node.get("actor_type") or node.get("actor") or "").strip().lower() != "ai":
                continue
            field = node.get("time")
            for time_range in ([field] if isinstance(field, str) else list(field or [])):
                try:
                    starts.append(to_seconds(split_time_range(time_range)[0]))
                except ValueError:
                    continue
    return min(starts) if starts else None


def warn(msg):
    print(f"Warning: {msg}", file=sys.stderr)


def setup_pairs():
    """Unique (participant_id, session_id) pairs of annotated_graph_experiment_setup.csv, in file order."""
    if not ANNOTATED_SETUP_CSV.is_file():
        return []
    seen, pairs = set(), []
    with open(ANNOTATED_SETUP_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pair = (row["participant_id"].strip().lower(), row["session_id"].strip().lower())
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def read_snippet_ids():
    if not CHAT_SNIPPETS_CSV.is_file():
        return set()
    with open(CHAT_SNIPPETS_CSV, newline="", encoding="utf-8") as f:
        return {r["experiment_id"] for r in csv.DictReader(f)}


def session_status():
    """{(participant, session): "needs_anchor" | "needs_build" | "done" | "no_chat"} for every
    annotated session: the union of new_annotated_graphs/ (annotated_sessions()) and the sessions of
    experiment2a's annotated_graph_experiment_setup.csv. A session can be annotated (and used by
    experiment1a/experiment1b, which read the aligned chat directly) without having any 2a rating rows.
      no_chat      -- the session has no chat file (nothing to anchor or build)
      needs_anchor -- no anchor row, or its video_first_mmss is blank (user must fill it)
      needs_build  -- anchor filled, but the aligned chat file is missing or, for a session that
                      has 2a setup rows, a step's snippet row or one of its eight chat prompt
                      files (2a and 2b) is missing
      done         -- nothing left to do (a re-run would change nothing)"""
    anchors = {(r["participant_id"], r["session_id"]): r for r in read_anchors()}
    snippet_ids = read_snippet_ids()
    step_ids = {}
    if ANNOTATED_SETUP_CSV.is_file():
        with open(ANNOTATED_SETUP_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pair = (row["participant_id"].strip().lower(), row["session_id"].strip().lower())
                step_ids.setdefault(pair, []).append(row["experiment_id"].strip())

    status = {}
    for pair in sorted(set(setup_pairs()) | set(annotated_sessions())):
        anchor = anchors.get(pair)
        steps = step_ids.get(pair, [])  # empty: annotated but no 2a rating rows -> only the aligned chat is checked
        if not chat_files(*pair):
            status[pair] = "no_chat"
        elif anchor is None or not (anchor.get("video_first_mmss") or "").strip():
            status[pair] = "needs_anchor"
        elif (
            not aligned_chat_path(*pair).is_file()
            or not set(steps) <= snippet_ids
            or any(not (d / f"{eid}.txt").is_file() for d in CHAT_PROMPT_DIRS for eid in steps)
        ):
            status[pair] = "needs_build"
        else:
            status[pair] = "done"
    return status
