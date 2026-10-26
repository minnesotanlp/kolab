"""
Extract user-response labels (adopt / modify / reject) from a behavior graph
JSON and write them out as a CSV.

A behavior graph node N carries a "user_response" field when the user acted
on something an AI produced:

    "user_response": [
        {"source_node": ..., "target_ai": ..., "label": "adopt|modify|reject",
         "source_node_time": ..., "ai_node_time": ..., "user_adoption_time": ...,
         "reasoning": ...}
    ]

Times are "HH:MM:SS-HH:MM:SS" (MM:SS, en-dashes and comma-separated multi-ranges
are also accepted; a multi-range runs from its first start to its last end).
source_node_time / ai_node_time say which occurrence of the node the response
refers to when it was visited more than once. If source_node_time is empty, the
source node's own "time" is used when it has exactly one range; with several
ranges the entry is skipped with a warning (as it is when source_node is empty).
An empty ai_node_time falls back to the AI node's own first-start..last-end
range. user_adoption_time and reasoning are not used by this setting.

For each entry, this script emits one CSV row:

    experiment_id         -- "exp<i>_<participant_id>_s<session>", where <i> is a
                              1-based counter over this file's rows
    user_action_node      -- id of node N (the node holding "user_response")
    ai_node               -- id of `ai_node` (the AI node this user action was
                              a response to)
    begin_timestamp      -- start timestamp of `source_node` (its first time range)
    end_timestamp        -- end timestamp of `ai_node` (its last time range)
    label                -- user_action ("adopt", "modify", or "reject")
    target_ai_timestamp -- "HH:MM:SS-HH:MM:SS" time range of `ai_node`
                            (its first time range's start to its last time
                            range's end)

If --output is omitted, the output path is derived from the input filename:
given an input like ".../session1_p1_behavior_graph.json", the output is
written to "../extracted_timestamps/labels_experiment1b/session1_p1_timestamp_labels.csv"
(relative to this script's directory).

Usage:
    python experiment1b_extract_user_response_labels.py --input ../annotated_graphs/session1_p6_behavior_graph_modified.json
    python experiment1b_extract_user_response_labels.py --input behavior_graph.json --output labels.csv
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

DEFAULT_OUTPUT_DIR = EXTRACTED_DIR / "labels_experiment1b"


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


def collapse_ranges(value):
    """'a-b' or a comma-separated list 'a-b, c-d' -> 'HH:MM:SS-HH:MM:SS' (first start to
    last end). Empty / None -> None. Raises ValueError on a malformed range."""
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    if value is None or not str(value).strip():
        return None
    # a time that was saved as the text of a list, "['00:01:03-00:02:31', ...]", is read like the plain string
    value = str(value).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    start = split_time_range(parts[0])[0]
    end = split_time_range(parts[-1])[1]
    return f"{normalize_timestamp(start)}-{normalize_timestamp(end)}"


def parse_user_response_entry(entry, node_id, node_index=None):
    """Return (source_node_id, ai_node_id, user_action, source_ts, ai_ts, adoption_ts) for a
    dict user_response entry, or None (and print a warning) if it can't be used.

    source_ts / ai_ts / adoption_ts are "HH:MM:SS-HH:MM:SS" strings or None. A None ai_ts
    / adoption_ts means "use the node's own range". When source_node_time is empty but
    source_node is set and node_index is given, source_ts is the source node's `time` if
    that is a single range; with several ranges the entry is skipped with a warning."""
    if isinstance(entry, (list, tuple)):
        # a graph that still holds an entry in the old list form [source, ai, label(, source_time, ai_time(, adoption_time))]
        if len(entry) in (3, 5, 6):
            keys = ("source_node", "target_ai", "label", "source_node_time", "ai_node_time", "user_adoption_time")
            entry = dict(zip(keys, entry))
        else:
            print(f"Warning: user_response entry on node {node_id!r} has {len(entry)} elements "
                  f"(expected 3, 5 or 6): {entry!r}; skipping.", file=sys.stderr)
            return None
    if not isinstance(entry, dict):
        print(
            f"Warning: user_response entry on node {node_id!r} is not a dict: {entry!r}; skipping.",
            file=sys.stderr,
        )
        return None
    source_node_id = entry.get("source_node")
    ai_node_id = entry.get("target_ai")
    user_action = entry.get("label")
    try:
        source_ts = collapse_ranges(entry.get("source_node_time"))
        ai_ts = collapse_ranges(entry.get("ai_node_time"))
        adoption_ts = collapse_ranges(entry.get("user_adoption_time"))
        if source_ts is None and source_node_id and node_index is not None:
            source_node = node_index.get(source_node_id)
            if source_node is not None:
                ranges = time_ranges(source_node)
                if len(ranges) != 1:
                    print(
                        f"Warning: node {node_id!r}: source_node {source_node_id!r} has no "
                        f"source_node_time and {len(ranges)} time ranges {ranges!r}; skipping.",
                        file=sys.stderr,
                    )
                    return None
                source_ts = collapse_ranges(ranges[0])
    except ValueError as exc:
        print(f"Warning: skipping entry for node {node_id!r}: {exc}", file=sys.stderr)
        return None
    return source_node_id, ai_node_id, user_action, source_ts, ai_ts, adoption_ts


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
            source_node_id, ai_node_id, user_action, source_ts_override, ai_ts_override, _ = parsed

            source_node = node_index.get(source_node_id)
            if source_node is None:
                print(
                    f"Warning: source_node {source_node_id!r} referenced by "
                    f"node {node_id!r} not found; skipping.",
                    file=sys.stderr,
                )
                continue

            ai_node = node_index.get(ai_node_id)
            if ai_node is None:
                print(
                    f"Warning: ai_node {ai_node_id!r} referenced by "
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

                if ai_ts_override is not None:
                    ai_start, ai_end = split_time_range(ai_ts_override)
                    target_ai_timestamp = f"{normalize_timestamp(ai_start)}-{normalize_timestamp(ai_end)}"
                    end_timestamp = normalize_timestamp(ai_end)
                else:
                    end_timestamp = node_end_timestamp(ai_node)
                    target_ai_timestamp = node_timestamp_range(ai_node)
            except ValueError as exc:
                print(f"Warning: skipping entry for node {node_id!r}: {exc}", file=sys.stderr)
                continue

            rows.append(
                {
                    "user_action_node": node_id,
                    "ai_node": ai_node_id,
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
    """Derive '../extracted_timestamps/labels_experiment1b/<session>_<p>_timestamp_labels.csv'
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

