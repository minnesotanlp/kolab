"""
Build the Experiment 2a graph-condition setup from the annotated behavior graphs.

Experiment 2a predicts a participant's 1-5 collaboration rating for one workflow
step. The graph conditions (experiment2a_video_graph / _video_graph_ts /
_graph_only / _graph_only_ts in prompt_maker_setting.py) only make sense for
steps that have been annotated, so this script looks at which sessions have a
graph in annotated_graphs/ (or, when a session has none, graph_clean/) and builds, for
exactly those sessions:

  extracted_timestamps/experiment2a/annotated_graph_experiment_setup.csv
      one row per workflow step -- the same rows as basic_experiment_setup.csv
      (minus its truncated_video_filename column) plus a graph_filename column
      (ratings, mp4_file_name, timestamp_ranges, ..., graph_filename). Ratings etc.
      are copied from that file, so nothing is re-parsed from the workflow docx
      files, and the experiment_ids are unchanged (so truncated clips and prompts
      line up).
  extracted_timestamps/2a_graphs/<experiment_id>.json
      the step's subgraph (nodes + edges, node times still on the FULL session
      video's clock, plus the row's timestamp_ranges so prompt_maker_setting.py
      can convert them to the step clip's clock)
  extracted_timestamps/2a_graphs/BUILD_REPORT.txt
      what was found / added / skipped, multi-range steps, and graph nodes whose
      footage is not (fully) in the step clip -- read it to check the build.

How a step's subgraph is cut: a session graph is a time-ordered list of segments,
each with a badge ("Step 1") and its own nodes/edges. A step the participant
returned to later appears as a SECOND segment with the same badge (e.g. Step 1 at
00:00-19:45 and again at 01:03:47-01:08:13) -- those are the step row's
";"-separated timestamp_ranges, and its clip is those pieces joined end to end. So
the subgraph merges the nodes/edges of EVERY segment whose badge contains the
step's number (the same badge rule as step_name_for_step_id() in
prompt_maker_setting.py), renumbers the nodes n1, n2, ... in that order, and
rewrites the edges to match (edges to a node outside the merged segments are
dropped).

Re-running is safe: rows already in annotated_graph_experiment_setup.csv are never
rewritten, only steps of newly-annotated sessions are appended (and only new rows
get a subgraph JSON unless --force-json). A step with no ratings row in
basic_experiment_setup.csv, or a ratings row with no matching graph step, is
skipped with a note in the report.

Experiment 2b: pass --experiment 2b to build experiment2b/annotated_graph_experiment_setup.csv the
same way from experiment2b/basic_experiment_setup.csv (output_quality_rating column). A newly
annotated session only shows up in 2a or 2b after its rows are in that experiment's
basic_experiment_setup.csv (build_experiment2a.py / build_experiment2b.py). The subgraph JSONs in 2a_graphs/ are shared with 2a.

Usage:
    python build_2a_graph_setup.py --dry-run          # print the report, write nothing
    python build_2a_graph_setup.py                    # build / append
    python build_2a_graph_setup.py --p p4 --session 1 # only one participant/session
    python build_2a_graph_setup.py --force-json       # also rewrite JSONs of rows already in the CSV
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import csv
import json
import re
import sys
from datetime import datetime

from prompts.prompt_maker_setting import (
    ANNOTATED_GRAPHS_DIR,
    ANNOTATED_SETUP_CSV,
    ANNOTATED_SETUP_CSV_2B,
    EXPERIMENT2A_CSV,
    EXPERIMENT2B_CSV,
    GRAPHS_2A_DIR,
    load_graph,
    parse_clip_ranges,
    session_range_to_clip_ranges,
    split_time_range,
    to_seconds,
)

# Which experiment's setup to build (--experiment): where its ratings come from, the
# annotated-steps CSV to append to, and where the report goes. The per-step subgraph
# JSONs are shared (2a_graphs/<experiment_id>.json): a step's graph does not depend on
# the rating, and 2b's 39 steps have the same ids and timestamp_ranges as 2a's.
EXPERIMENTS = {
    "2a": {"source": EXPERIMENT2A_CSV, "output": ANNOTATED_SETUP_CSV, "report": GRAPHS_2A_DIR / "BUILD_REPORT.txt"},
    "2b": {"source": EXPERIMENT2B_CSV, "output": ANNOTATED_SETUP_CSV_2B, "report": GRAPHS_2A_DIR / "BUILD_REPORT_2b.txt"},
}
ANNOTATED_NAME_RE = re.compile(r"^session(\d+)_p(\d+)_", re.IGNORECASE)


def discover_annotated_sessions():
    """[(participant_id, session_id), ...] like [("p1", "session1"), ...] for every
    session with a behavior graph in new_annotated_graphs/, ordered by session then
    participant number."""
    pairs = set()
    for path in ANNOTATED_GRAPHS_DIR.glob("session*_[pP]*_*.json"):
        match = ANNOTATED_NAME_RE.match(path.name)
        if match:
            pairs.add((f"p{match.group(2)}", f"session{match.group(1)}"))
    return sorted(pairs, key=lambda pair: (int(pair[1][len("session"):]), int(pair[0][1:])))


def load_session_graph(session_id, participant_id):
    """(graph, source_folder_name) for a session from new_annotated_graphs/.
    Raises FileNotFoundError if it has none."""
    session_number = int(session_id[len("session"):])
    return load_graph(session_number, participant_id), ANNOTATED_GRAPHS_DIR.name


# Column that truncate_videos_for_setting.py adds to basic_experiment_setup.csv. It is not part of
# annotated_graph_experiment_setup.csv, so it is dropped when copying rows over.
SOURCE_ONLY_COLUMNS = ("truncated_video_filename",)


def load_source_rows(source_csv):
    """(fieldnames, rows) of the experiment's basic_experiment_setup.csv (2a and 2b), in file
    order -- the ratings / mp4_file_name / timestamp_ranges source. Columns that only exist in the
    basic CSV (SOURCE_ONLY_COLUMNS) are dropped."""
    with open(source_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [name for name in reader.fieldnames if name not in SOURCE_ONLY_COLUMNS]
        rows = [{name: row[name] for name in fieldnames} for row in reader]
    return fieldnames, rows


def step_number(step_id):
    match = re.search(r"\d+", step_id)
    return int(match.group(0)) if match else None


def badge_numbers(badge):
    return [int(n) for n in re.findall(r"\d+", badge or "")]


def segments_for_step(graph, number):
    """Every graph segment whose badge contains step `number` ("Step 1", "Step 1-2",
    "Step 1+2+3" all match 1; "custom-*" badges are skipped), in graph order."""
    return [
        segment
        for segment in graph.get("steps", [])
        if "custom" not in (segment.get("badge") or "").lower() and number in badge_numbers(segment.get("badge"))
    ]


def cut_step_subgraph(graph, row):
    """Merge the nodes/edges of every segment for this row's step into one subgraph.
    Returns (subgraph_dict, info) or (None, reason) if the graph has no such step.

    Node ids become n1, n2, ... in segment/list order (original id kept as
    "orig_id"); node "time" stays on the full session video's clock."""
    number = step_number(row["step_id"])
    segments = segments_for_step(graph, number) if number is not None else []
    if not segments:
        return None, f"graph has no segment for step {row['step_id']!r}"

    # Every node is kept, even if two nodes happen to share an original id (a
    # graph_clean file reuses "n1" for two different nodes): each segment gets its
    # own orig-id -> new-id map for its edges, and id_map (first occurrence wins)
    # only serves edges that point at a node in another segment.
    id_map = {}
    segment_maps = []
    nodes = []
    for segment in segments:
        local_map = {}
        for node in segment.get("nodes", []):
            new_id = f"n{len(nodes) + 1}"
            local_map.setdefault(node["id"], new_id)
            id_map.setdefault(node["id"], new_id)
            nodes.append(
                {
                    "id": new_id,
                    "orig_id": node["id"],
                    "actor_type": node.get("actor_type") or node.get("actor") or "",
                    "label": node.get("label", ""),
                    "detail": node.get("detail", ""),
                    "time": node.get("time"),
                }
            )
        segment_maps.append(local_map)

    edges, seen, dropped = [], set(), 0
    for segment, local_map in zip(segments, segment_maps):
        for edge in segment.get("edges", []):
            src = local_map.get(edge.get("from"), id_map.get(edge.get("from")))
            dst = local_map.get(edge.get("to"), id_map.get(edge.get("to")))
            if src is None or dst is None:
                dropped += 1
                continue
            key = (src, dst, edge.get("type", ""))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": src, "to": dst, "type": edge.get("type", "")})

    subgraph = {
        "experiment_id": row["experiment_id"].strip(),
        "step_id": row["step_id"].strip(),
        "step_name": segments[0].get("step_name", ""),
        "timestamp_ranges": row["timestamp_ranges"].strip(),
        "segments": [{"badge": s.get("badge"), "time_range": s.get("time_range")} for s in segments],
        "nodes": nodes,
        "edges": edges,
    }
    return subgraph, {"segments": len(segments), "dropped_edges": dropped}


def nodes_outside_clip(subgraph):
    """[(node_id, orig_id, session_times, kind)] for nodes whose footage is not fully
    in the step clip: kind "gap" = none of it, "partial" = only part of it. Only
    meaningful for steps with a gap between their pieces (multi-range steps) -- a
    single-range step's own nodes normally sit inside its range."""
    clip_ranges = parse_clip_ranges(subgraph["timestamp_ranges"])
    found = []
    for node in subgraph["nodes"]:
        time_field = node.get("time")
        if not time_field:
            continue
        session_ranges = [time_field] if isinstance(time_field, str) else list(time_field)
        wanted, covered = 0, 0
        for session_range in session_ranges:
            try:
                start, end = split_time_range(session_range)
                start_s, end_s = to_seconds(start), to_seconds(end)
            except ValueError:
                continue
            wanted += end_s - start_s
            covered += sum(b - a for a, b in session_range_to_clip_ranges(start_s, end_s, clip_ranges))
        if wanted and covered == 0:
            found.append((node["id"], node["orig_id"], session_ranges, "gap"))
        elif wanted and covered < wanted:
            found.append((node["id"], node["orig_id"], session_ranges, "partial"))
    return found


def read_existing_ids(output_csv):
    if not output_csv.is_file():
        return set()
    with open(output_csv, newline="", encoding="utf-8") as f:
        return {row["experiment_id"].strip() for row in csv.DictReader(f)}


def append_new_rows(output_csv, fieldnames, rows):
    """Append `rows` to the experiment's annotated_graph_experiment_setup.csv (header
    first if the file is new). Existing rows are never touched."""
    write_header = not output_csv.is_file() or output_csv.stat().st_size == 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment", default="2a", choices=sorted(EXPERIMENTS),
        help="Which experiment's setup to build (default 2a). 2b reads experiment2b/basic_experiment_setup.csv "
             "and writes experiment2b/annotated_graph_experiment_setup.csv (its subgraphs are the same "
             "2a_graphs/<experiment_id>.json files; ones that already exist are left alone unless --force-json).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the report; write no CSV, JSON or report file")
    parser.add_argument("--p", default=None, help="Only this participant, e.g. p4")
    parser.add_argument("--session", default=None, choices=["1", "2"], help="Only this session")
    parser.add_argument(
        "--force-json", action="store_true",
        help="Also rewrite the subgraph JSON of steps that are already in the CSV (rows themselves are never rewritten)",
    )
    args = parser.parse_args()

    source_csv = EXPERIMENTS[args.experiment]["source"]
    output_csv = EXPERIMENTS[args.experiment]["output"]
    report_path = EXPERIMENTS[args.experiment]["report"]
    if not source_csv.is_file():
        raise SystemExit(f"No such file: {source_csv} (the ratings source)")

    fieldnames, source_rows = load_source_rows(source_csv)
    if "graph_filename" not in fieldnames:
        fieldnames.append("graph_filename")
    existing_ids = read_existing_ids(output_csv)

    report = [f"Build report (experiment {args.experiment}) -- {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    sessions = discover_annotated_sessions()
    if args.p:
        sessions = [s for s in sessions if s[0] == args.p.strip().lower()]
    if args.session:
        sessions = [s for s in sessions if s[1] == f"session{args.session}"]
    report.append(
        f"Sessions with a graph in {ANNOTATED_GRAPHS_DIR.name}/: {len(sessions)}"
    )

    new_rows, json_jobs = [], []  # json_jobs: (experiment_id, subgraph)
    multi_range, outside, skipped = [], [], []
    for participant_id, session_id in sessions:
        session_rows = [
            r for r in source_rows
            if r["participant_id"].strip().lower() == participant_id and r["session_id"].strip().lower() == session_id
        ]
        if not session_rows:
            skipped.append(f"{participant_id} {session_id}: annotated, but no rows in {source_csv.name} (no ratings)")
            continue
        try:
            graph, graph_source = load_session_graph(session_id, participant_id)
        except FileNotFoundError as exc:
            skipped.append(f"{participant_id} {session_id}: {exc}")
            continue

        row_steps = set()
        for row in session_rows:
            experiment_id = row["experiment_id"].strip()
            row_steps.add(step_number(row["step_id"]))
            subgraph, info = cut_step_subgraph(graph, row)
            if subgraph is None:
                skipped.append(f"{experiment_id}: {info}")
                continue
            if len(parse_clip_ranges(row["timestamp_ranges"])) > 1:
                multi_range.append(
                    f"{experiment_id}: {row['timestamp_ranges']} <- {info['segments']} graph segment(s)"
                )
            for node_id, orig_id, times, kind in nodes_outside_clip(subgraph):
                outside.append(f"{experiment_id}: {node_id} (annotated {orig_id}) {times} -- {kind}")
            if info["dropped_edges"]:
                report.append(f"  note: {experiment_id}: {info['dropped_edges']} edge(s) to nodes outside this step dropped")

            if experiment_id not in existing_ids:
                new_row = dict(row)
                new_row["graph_filename"] = f"{experiment_id}.json"
                new_rows.append(new_row)
                json_jobs.append((experiment_id, subgraph))
            elif args.force_json:
                json_jobs.append((experiment_id, subgraph))

        for segment in graph.get("steps", []):
            numbers = badge_numbers(segment.get("badge"))
            if numbers and "custom" not in (segment.get("badge") or "").lower() and not (set(numbers) & row_steps):
                skipped.append(
                    f"{participant_id} {session_id}: graph step {segment.get('badge')!r} has no ratings row (not built)"
                )

    report.append(f"Steps already in {output_csv.parent.name}/{output_csv.name}: {len(existing_ids)}")
    report.append(f"New steps to add: {len(new_rows)}")
    for row in new_rows:
        report.append(f"  + {row['experiment_id']}")
    report.append("")
    report.append(f"Multi-range steps (clip = pieces joined end to end): {len(multi_range)}")
    report.extend(f"  {line}" for line in multi_range)
    report.append("")
    report.append(f"Graph nodes whose footage is not fully in the step clip: {len(outside)}")
    report.append("  (shown without a time -- or with the covered part only -- in the *_ts prompts)")
    report.extend(f"  {line}" for line in outside)
    report.append("")
    report.append(f"Skipped / not built: {len(skipped)}")
    report.extend(f"  {line}" for line in skipped)

    print("\n".join(report))

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    GRAPHS_2A_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for experiment_id, subgraph in json_jobs:
        json_path = GRAPHS_2A_DIR / f"{experiment_id}.json"
        # The subgraph files are shared with 2a: for 2b leave an existing one alone.
        if args.experiment == "2b" and json_path.exists() and not args.force_json:
            continue
        json_path.write_text(json.dumps(subgraph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
    if new_rows:
        append_new_rows(output_csv, fieldnames, new_rows)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote {written} subgraph JSON(s) to {GRAPHS_2A_DIR}, appended {len(new_rows)} row(s) to "
          f"{output_csv}, report at {report_path}")


if __name__ == "__main__":
    main()
