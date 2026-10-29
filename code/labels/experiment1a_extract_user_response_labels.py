"""
Extract user-response labels (adopt / modify / reject) from a behavior graph
JSON and write them out as a CSV.

A behavior graph node N carries a "user_response" field when the user acted
on something Claude produced:

    "user_response": [
        {"source_node": ..., "target_ai": ..., "label": "adopt|modify|reject",
         "source_node_time": ..., "ai_node_time": ..., "user_adoption_time": ...,
         "reasoning": ...}
    ]

Times are "HH:MM:SS-HH:MM:SS" (MM:SS, en-dashes and comma-separated multi-ranges
are also accepted; a multi-range runs from its first start to its last end) and say
which occurrence of the node the response refers to when it was visited more than
once: they replace source_node's / target_ai's own first/last "time" range, and
user_adoption_time replaces node N's last "time" range for end_timestamp. If
source_node_time is empty, source_node's own "time" is used when it has exactly
one range; with several ranges (or an empty source_node) the entry is skipped with
a warning. reasoning is not used.

For each entry, this script emits one CSV row:

    experiment_id      -- "exp<i>_<participant_id>_s<session>", where <i> is a
                           1-based counter over this file's rows (e.g. the
                           first row of session1_p1's CSV is "exp1_p1_s1")
    user_action_node   -- id of node N, the node holding "user_response"
                           (the user's action taken in response to the AI output)
    ai_node            -- id of `claude_node` (the AI node this user action
                           was a response to)
    begin_timestamp    -- start timestamp of `source_node` (its first time range)
    end_timestamp      -- end timestamp of node N itself (its last time range,
                           or the end of user_response_timestamp_range)
    label              -- user_action ("adopt", "modify", or "reject")
    target_ai_timestamp -- "HH:MM:SS-HH:MM:SS" time range of `claude_node`
                            (its first time range's start to its last time
                            range's end)

If --output is omitted, the output path is derived from the input filename:
given an input like ".../session1_p1_behavior_graph.json", the output is
written to "../extracted_timestamps/labels_experiment1a/session1_p1_timestamp_labels.csv"
(relative to this script's directory).

Usage:
    python experiment1a_extract_user_response_labels.py --input ../annotated_graphs/session2_p1_behavior_graph_modified.json
    python experiment1a_extract_user_response_labels.py --input behavior_graph.json --output labels.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder
from config import EXTRACTED_DIR

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from labels.experiment1b_extract_user_response_labels import parse_user_response_entry

DEFAULT_OUTPUT_DIR = EXTRACTED_DIR / "labels_experiment1a"


def walk(obj):
    """Yield every dict nested anywhere inside obj (dicts and lists)."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk(item)


def normalize_timestamp(ts):
    """Normalize a timestamp string to HH:MM:SS."""
    parts = [p.strip() for p in ts.strip().split(":")]
    if len(parts) == 2:
        parts = ["00"] + parts
    if len(parts) != 3:
        raise ValueError(f"Cannot parse timestamp: {ts!r}")
    parts = [p.zfill(2) for p in parts]
    return ":".join(parts)


def split_time_range(range_str):
    """Split a 'HH:MM:SS-HH:MM:SS' style range into (start, end)."""
    for sep in ("–", "—", "-"):
        if sep in range_str:
            start, _, end = range_str.partition(sep)
            if start.strip() and end.strip():
                return start.strip(), end.strip()
    raise ValueError(f"Cannot split time range: {range_str!r}")


def time_ranges(node):
    """Return the node's `time` field as a list of range strings."""
    time_field = node.get("time")
    if time_field is None:
        raise ValueError(f"Node {node.get('id')!r} has no 'time' field")
    if isinstance(time_field, str):
        return [time_field]
    return list(time_field)


def node_begin_timestamp(node):
    start, _ = split_time_range(time_ranges(node)[0])
    return normalize_timestamp(start)


def node_end_timestamp(node):
    _, end = split_time_range(time_ranges(node)[-1])
    return normalize_timestamp(end)


def node_timestamp_range(node):
    return f"{node_begin_timestamp(node)}-{node_end_timestamp(node)}"


def build_node_index(data):
    """Map node id -> node dict, searching the whole graph."""
    index = {}
    for obj in walk(data):
        if isinstance(obj, dict) and "id" in obj and "time" in obj:
            index[obj["id"]] = obj
    return index


def extract_rows(data):
    node_index = build_node_index(data)
    rows = []

    for node in walk(data):
        if not isinstance(node, dict) or "user_response" not in node:
            continue

        node_id = node.get("id")

        for entry in node["user_response"]:
            parsed = parse_user_response_entry(entry, node_id, node_index)
            if parsed is None:
                continue
            (source_node_id, claude_node_id, user_action,
             source_ts_override, claude_ts_override, response_ts_override) = parsed

            source_node = node_index.get(source_node_id)
            if source_node is None:
                print(
                    f"Warning: source_node {source_node_id!r} referenced by "
                    f"node {node_id!r} not found; skipping.",
                    file=sys.stderr,
                )
                continue

            claude_node = node_index.get(claude_node_id)
            if claude_node is None:
                print(
                    f"Warning: claude_node {claude_node_id!r} referenced by "
                    f"node {node_id!r} not found; skipping.",
                    file=sys.stderr,
                )
                continue

            try:
                if source_ts_override is not None:
                    start, _ = split_time_range(source_ts_override)
                    begin_timestamp = normalize_timestamp(start)
                else:
                    begin_timestamp = node_begin_timestamp(source_node)

                if claude_ts_override is not None:
                    ai_start, ai_end = split_time_range(claude_ts_override)
                    target_ai_timestamp = f"{normalize_timestamp(ai_start)}-{normalize_timestamp(ai_end)}"
                else:
                    target_ai_timestamp = node_timestamp_range(claude_node)

                if response_ts_override is not None:
                    _, response_end = split_time_range(response_ts_override)
                    end_timestamp = normalize_timestamp(response_end)
                else:
                    end_timestamp = node_end_timestamp(node)
            except ValueError as exc:
                print(f"Warning: skipping entry for node {node_id!r}: {exc}", file=sys.stderr)
                continue

            rows.append(
                {
                    "user_action_node": node_id,
                    "ai_node": claude_node_id,
                    "begin_timestamp": begin_timestamp,
                    "end_timestamp": end_timestamp,
                    "label": user_action,
                    "target_ai_timestamp": target_ai_timestamp,
                }
            )

    return rows


def add_experiment_ids(rows, data):
    """Add an "experiment_id" field to each row: exp<i>_<participant_id>_s<session>."""
    participant_id = str(data.get("participant_id", "p?")).lower()  # some graphs say "P15"; ids stay lowercase
    # the number after the word "session" ("p17_session2" -> 2); the first number alone would give 17
    session_match = re.search(r"session\D*(\d+)", str(data.get("session_id", "")), re.IGNORECASE) or re.search(r"\d+", str(data.get("session_id", "")))
    session_num = (session_match.group(1) if session_match.re.groups else session_match.group(0)) if session_match else "?"

    for i, row in enumerate(rows, start=1):
        row["experiment_id"] = f"exp{i}_{participant_id}_s{session_num}"


def default_output_path(input_path):
    """Derive '../extracted_timestamps/labels_experiment1a/<session>_<p>_timestamp_labels.csv'
    from an input filename containing a 'session<N>_p<N>' segment."""
    stem = Path(input_path).stem
    match = re.search(r"session\d+_p\d+", stem)
    if match:
        prefix = match.group(0)
    else:
        print(
            f"Warning: could not find 'session<N>_p<N>' in {stem!r}; "
            f"using the full stem as the output filename prefix.",
            file=sys.stderr,
        )
        prefix = stem
    return DEFAULT_OUTPUT_DIR / f"{prefix}_timestamp_labels.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the input behavior graph JSON")
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Path to the output CSV file (default: derived from --input, see module docstring)",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else default_output_path(args.input)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r") as f:
        data = json.load(f)

    rows = extract_rows(data)
    add_experiment_ids(rows, data)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_id",
                "user_action_node",
                "ai_node",
                "begin_timestamp",
                "end_timestamp",
                "label",
                "target_ai_timestamp",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {output_path}")


if __name__ == "__main__":
    main()

