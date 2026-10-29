"""
Build experiment 1d in one go: select instances, cut their videos, write the setup CSV and the
video-only prompts.

Experiment 1d is experiment 1b with every clip starting at the beginning of the workflow recording
(00:00:00) instead of at the source node: the clip runs from 00:00:00 to the end of the target
AI action, so the model sees the whole workflow up to that action.

Candidates are experiment 1b's instances (same parsing and experiment_ids as
experiment1b_extract_user_response_labels.py, re-derived from annotated_graphs/ so the source node
id is known, and checked against the labels_experiment1b CSVs), with duplicates removed:

    * instances sharing (participant, session, target AI node, target AI time range) get the
      same clip and the same prompt; one of them is kept (lowest exp number)
    * if such a group mixes labels (same input, different answers), the whole group is dropped

Selection: every de-duplicated instance. --ids takes exactly the given experiment_ids instead
(graph_update.py uses it); rows of an existing setup CSV that belong to sessions with no graph
are carried over unchanged.

Outputs:
    extracted_timestamps/experiment1d/basic_setup.csv
        experiment_id, participant_id, session_id, source_node, user_action_node, ai_node,
        label, begin_timestamp, end_timestamp, target_ai_timestamp, clip_length_seconds,
        video_file_name, prompt_file_name
    extracted_timestamps/truncated_videos/experiment1d/<p>_s<S>_exp<i>.mp4
        cut with a stream copy (ffmpeg -t <end> -c copy, no re-encode: every clip starts at
        0, so there is no keyframe problem at the start); existing clips are kept
    extracted_timestamps/prompts/experiment1d/<p>_s<S>_exp<i>.txt
        experiment 1b's video-only prompt (same template), with the target AI action timestamp
        on the full video's timeline since the clip starts at 00:00:00
    extracted_timestamps/prompts/experiment1d_<condition>/<p>_s<S>_exp<i>.txt
        the prompts of the six other conditions (experiment 1b's templates, graph and
        chat cut to 00:00:00 .. end); instances whose session has no aligned chat log get no
        chat prompts

So that the regular runners (run_experiments.py --exp1d, eval_results.py, progress.py) can
use experiment 1d like experiments 1a/1b, it also writes the files they look for:
    extracted_timestamps/labels_experiment1d/session<S>_p<N>_timestamp_labels.csv
        the selected rows only, in the labels_experiment1b format, begin_timestamp 00:00:00
    extracted_timestamps/truncated_videos/experiment1d/session<S>_p<N>_00-00-00_<end>.mp4
        relative symlinks to the <p>_s<S>_exp<i>.mp4 clips

Usage:
    python build_experiment1d.py --dry-run                 # print the instances, write nothing
    python build_experiment1d.py                           # build every de-duplicated instance
    python build_experiment1d.py --ids my_picks.txt        # exactly these experiment_ids (one per line)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder
from config import EXTRACTED_DIR

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from prompts import prompt_maker_setting as pm
from labels.experiment1b_extract_user_response_labels import (
    add_experiment_ids,
    build_node_index,
    node_begin_timestamp,
    node_end_timestamp,
    node_timestamp_range,
    normalize_timestamp,
    parse_user_response_entry,
    split_time_range,
    walk,
)
from truncation.truncate_videos_for_setting import find_source_video

TIMESTAMPS_DIR = EXTRACTED_DIR
EXPERIMENT1B_CSV_DIR = TIMESTAMPS_DIR / "labels_experiment1b"
OUTPUT_CSV = TIMESTAMPS_DIR / "experiment1d" / "basic_setup.csv"
LABELS_DIR = TIMESTAMPS_DIR / "labels_experiment1d"
VIDEO_DIR = TIMESTAMPS_DIR / "truncated_videos" / "experiment1d"
PROMPT_DIR = TIMESTAMPS_DIR / "prompts" / "experiment1d"
# condition prompt folders (prompts/experiment1d_<name>), built with experiment 1b's condition templates
CONDITIONS = ("video_graph", "graph_only", "video_chat", "chat_only", "graph_chat", "video_graph_chat")
LABEL_FIELDNAMES = ["experiment_id", "user_action_node", "ai_node", "begin_timestamp", "end_timestamp",
                    "label", "target_ai_timestamp"]

WORKFLOW_START = "00:00:00"

FIELDNAMES = [
    "experiment_id", "participant_id", "session_id", "source_node", "user_action_node", "ai_node",
    "label", "begin_timestamp", "end_timestamp", "target_ai_timestamp", "clip_length_seconds",
    "video_file_name", "prompt_file_name",
]


def to_seconds(ts):
    h, m, s = (int(p) for p in normalize_timestamp(ts).split(":"))
    return h * 3600 + m * 60 + s


def extract_rows_with_source(data):
    """experiment1b_extract_user_response_labels.extract_rows, plus the source node id: same walk,
    same skips, same order, so add_experiment_ids gives the same experiment_ids."""
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
            ai_node = node_index.get(ai_node_id)
            if source_node is None or ai_node is None:
                continue
            try:
                if source_ts_override is not None:
                    begin_timestamp = normalize_timestamp(split_time_range(source_ts_override)[0])
                else:
                    begin_timestamp = node_begin_timestamp(source_node)
                if ai_ts_override is not None:
                    ai_start, ai_end = split_time_range(ai_ts_override)
                    target_ai_timestamp = f"{normalize_timestamp(ai_start)}-{normalize_timestamp(ai_end)}"
                    end_timestamp = normalize_timestamp(ai_end)
                else:
                    end_timestamp = node_end_timestamp(ai_node)
                    target_ai_timestamp = node_timestamp_range(ai_node)
            except ValueError:
                continue
            rows.append({
                "source_node": source_node_id,
                "user_action_node": node_id,
                "ai_node": ai_node_id,
                "experiment1b_begin_timestamp": begin_timestamp,
                "end_timestamp": end_timestamp,
                "label": user_action,
                "target_ai_timestamp": target_ai_timestamp,
            })
    return rows


def load_candidates():
    """Every experiment 1b instance (with its graph), checked against the labels_experiment1b CSVs."""
    candidates, mismatches, no_graph = [], [], []
    for csv_path, session, participant_id in pm.discover_jobs(None, None, "1b"):
        try:
            graph = pm.load_graph(session, participant_id)
        except FileNotFoundError:
            no_graph.append(f"{participant_id} session{session}")
            continue
        rows = extract_rows_with_source(graph)
        add_experiment_ids(rows, graph)
        with open(csv_path, newline="") as f:
            expected = {r["experiment_id"]: r for r in csv.DictReader(f)}
        for row in rows:
            ref = expected.get(row["experiment_id"])
            same = ref is not None and all(
                ref[k] == row[k2] for k, k2 in (("user_action_node", "user_action_node"), ("ai_node", "ai_node"),
                                                ("label", "label"), ("end_timestamp", "end_timestamp"),
                                                ("begin_timestamp", "experiment1b_begin_timestamp"))
            )
            if not same:
                mismatches.append(row["experiment_id"])
                continue
            candidates.append({
                **row,
                "participant_id": participant_id,
                "session": session,
                "graph": graph,
                "length": to_seconds(row["end_timestamp"]),
                "exp_num": int(row["experiment_id"].split("_")[0][3:]),
            })
    if no_graph:
        print(f"{len(no_graph)} session(s) have no graph in {pm.ANNOTATED_GRAPHS_DIR.name}/ and are not rebuilt "
              f"(their existing experiment 1d rows, clips and prompts are kept as they are).")
    if mismatches:
        print(f"Warning: {len(mismatches)} instance(s) differ from labels_experiment1b and were left out "
              f"(re-run the experiment 1b extraction?): {mismatches}", file=sys.stderr)
    return candidates


def dedupe(candidates):
    """One instance per distinct (participant, session, AI node, AI time range); groups whose
    instances disagree on the label are dropped."""
    groups = defaultdict(list)
    for c in candidates:
        groups[(c["participant_id"], c["session"], c["ai_node"], c["target_ai_timestamp"])].append(c)
    kept, conflicting = [], []
    for group in groups.values():
        if len({c["label"] for c in group}) > 1:
            conflicting.append([(c["experiment_id"], c["label"]) for c in group])
            continue
        kept.append(min(group, key=lambda c: c["exp_num"]))
    print(f"{len(candidates)} instances -> {len(kept)} distinct AI outputs "
          f"({len(conflicting)} group(s) with conflicting labels dropped)")
    for group in conflicting:
        print(f"  dropped (conflicting labels): {group}")
    return kept


def read_ids(path):
    """experiment_ids from a setup CSV (its experiment_id column) or a plain list, one per line."""
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            return [row["experiment_id"] for row in csv.DictReader(f)]
    return [line.strip() for line in open(path) if line.strip() and not line.startswith("#")]


def id_session_has_graph(experiment_id):
    """exp3_p12_s1 -> is there a graph for session1_p12 in new_annotated_graphs/?"""
    m = re.fullmatch(r"exp\d+_(p\d+)_s(\d+)", experiment_id)
    return bool(m) and any(pm.ANNOTATED_GRAPHS_DIR.glob(f"session{m.group(2)}_{m.group(1)}_*.json"))


def select_ids(pool, wanted):
    """Candidates for the wanted ids, plus the ids of sessions with no graph (carried over unchanged)."""
    by_id = {c["experiment_id"]: c for c in pool}
    carried = [i for i in wanted if i not in by_id and not id_session_has_graph(i)]
    wanted = [i for i in wanted if i not in carried]
    missing = [i for i in wanted if i not in by_id]
    if missing:
        sys.exit(f"Not among the de-duplicated experiment 1b instances (unknown id, duplicate AI output "
                 f"or conflicting labels): {missing}")
    return [by_id[i] for i in dict.fromkeys(wanted)], carried


def print_summary(selected):
    print("\nlabel    n   min  median   mean    max (minutes)  participants")
    for label in ("reject", "adopt", "modify"):
        mins = [c["length"] / 60 for c in selected if c["label"] == label]
        if not mins:
            continue
        people = Counter(c["participant_id"] for c in selected if c["label"] == label)
        print(f"{label:7s} {len(mins):2d} {min(mins):5.1f} {statistics.median(mins):7.1f} {statistics.mean(mins):6.1f} "
              f"{max(mins):6.1f}   {len(people)} (max {max(people.values())} each)")
    for label in ("reject", "adopt", "modify"):
        picks = sorted((c for c in selected if c["label"] == label), key=lambda c: c["length"])
        print(f"\n{label}: " + "  ".join(f"{c['experiment_id']} {c['length'] / 60:.1f}" for c in picks))


def cut_video(candidate, output_path):
    if output_path.exists():
        return True
    source = find_source_video(candidate["session"], candidate["participant_id"][1:])
    if source is None:
        print(f"Warning: no source video for {candidate['experiment_id']}; skipping.", file=sys.stderr)
        return False
    part_path = output_path.with_name(output_path.name + ".part")  # renamed when complete
    cmd = ["ffmpeg", "-y", "-i", str(source), "-t", str(candidate["length"]), "-c", "copy", "-f", "mp4", str(part_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: ffmpeg failed for {output_path.name}:\n{result.stderr[-1000:]}", file=sys.stderr)
        part_path.unlink(missing_ok=True)
        return False
    os.replace(part_path, output_path)
    return True


def write_prompt(candidate, output_path, occupations, condition=None):
    """condition None: the video-only prompt; else a experiment 1b condition name, e.g. "chat_only"."""
    graph = candidate["graph"]
    result = pm.build_prompt(
        graph, pm.build_node_index(graph), candidate["user_action_node"], candidate["target_ai_timestamp"],
        WORKFLOW_START, candidate["end_timestamp"], candidate["participant_id"], candidate["session"],
        "1b", occupations, condition=f"experiment1b_{condition}" if condition else None,
    )
    if result is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result[1] + "\n")
    return True


def runner_video_name(candidate):
    """The clip name run_*_setting.py's collect_samples() looks for under truncated_videos/experiment1d/."""
    return (f"session{candidate['session']}_{candidate['participant_id']}_"
            f"{WORKFLOW_START.replace(':', '-')}_{candidate['end_timestamp'].replace(':', '-')}.mp4")


def write_runner_files(candidates):
    """labels_experiment1d/ label CSVs and the runner-named video links, for the selected candidates."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    # only label files of sessions that have a graph are rebuilt; the others keep theirs
    for old in LABELS_DIR.glob("*.csv"):
        m = re.fullmatch(r"session(\d+)_(p\d+)_timestamp_labels", old.stem)
        if m and any(pm.ANNOTATED_GRAPHS_DIR.glob(f"session{m.group(1)}_{m.group(2)}_*.json")):
            old.unlink()
    by_session = defaultdict(list)
    for c in candidates:
        by_session[(c["session"], c["participant_id"])].append(c)
    for (session, participant_id), group in by_session.items():
        with open(LABELS_DIR / f"session{session}_{participant_id}_timestamp_labels.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_FIELDNAMES)
            writer.writeheader()
            for c in sorted(group, key=lambda c: c["exp_num"]):
                writer.writerow({"experiment_id": c["experiment_id"], "user_action_node": c["user_action_node"],
                                 "ai_node": c["ai_node"], "begin_timestamp": WORKFLOW_START,
                                 "end_timestamp": c["end_timestamp"], "label": c["label"],
                                 "target_ai_timestamp": c["target_ai_timestamp"]})
        for c in group:
            link = VIDEO_DIR / runner_video_name(c)
            if not link.is_symlink() and not link.exists():
                stem = Path(pm.prompt_filename(c["experiment_id"], c["participant_id"], c["session"])).stem
                link.symlink_to(f"{stem}.mp4")
    print(f"Wrote {len(by_session)} label CSV(s) to {LABELS_DIR.resolve()} and the runner video links")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", type=Path, default=None, help="file of experiment_ids to use instead of every instance")
    parser.add_argument("--dry-run", action="store_true", help="print the selection; cut and write nothing")
    args = parser.parse_args()

    pool = dedupe(load_candidates())
    carried = []
    if args.ids:
        selected, carried = select_ids(pool, read_ids(args.ids))
    else:
        selected = list(pool)
        if OUTPUT_CSV.exists():
            in_pool = {c["experiment_id"] for c in pool}
            carried = [i for i in read_ids(OUTPUT_CSV) if i not in in_pool and not id_session_has_graph(i)]
    selected.sort(key=lambda c: (int(c["participant_id"][1:]), c["session"], c["exp_num"]))
    print_summary(selected)
    if args.dry_run:
        return

    for folder in (OUTPUT_CSV.parent, VIDEO_DIR, PROMPT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    occupations = pm.load_occupations()
    rows, built, failed = [], [], []
    condition_counts = Counter()
    for i, c in enumerate(selected, 1):
        stem = Path(pm.prompt_filename(c["experiment_id"], c["participant_id"], c["session"])).stem
        video_path, prompt_path = VIDEO_DIR / f"{stem}.mp4", PROMPT_DIR / f"{stem}.txt"
        print(f"[{i}/{len(selected)}] {c['experiment_id']} ({c['length'] / 60:.1f} min)")
        if not (cut_video(c, video_path) and write_prompt(c, prompt_path, occupations)):
            failed.append(c["experiment_id"])
            continue
        for condition in CONDITIONS:
            # None (skipped) for the chat conditions when the session has no aligned chat log
            condition_prompt = PROMPT_DIR.parent / f"experiment1d_{condition}" / f"{stem}.txt"
            condition_counts[condition] += write_prompt(c, condition_prompt, occupations, condition)
        built.append(c)
        rows.append({
            "experiment_id": c["experiment_id"],
            "participant_id": c["participant_id"],
            "session_id": f"session{c['session']}",
            "source_node": c["source_node"],
            "user_action_node": c["user_action_node"],
            "ai_node": c["ai_node"],
            "label": c["label"],
            "begin_timestamp": WORKFLOW_START,
            "end_timestamp": c["end_timestamp"],
            "target_ai_timestamp": c["target_ai_timestamp"],
            "clip_length_seconds": c["length"],
            "video_file_name": video_path.name,
            "prompt_file_name": prompt_path.name,
        })

    if carried:
        with open(OUTPUT_CSV, newline="") as f:
            old_rows = {r["experiment_id"] + "|" + r["participant_id"] + "|" + r["session_id"]: r for r in csv.DictReader(f)}
        carried_rows = [r for r in old_rows.values()
                        if any(r["experiment_id"] == i for i in carried)]
        rows = rows + carried_rows
        rows.sort(key=lambda r: (int(r["participant_id"][1:]), r["session_id"], int(r["experiment_id"].split("_")[0][3:])))
        print(f"Kept {len(carried_rows)} existing row(s) of sessions with no graph.")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} row(s) to {OUTPUT_CSV.resolve()}")
    print(f"Videos in {VIDEO_DIR.resolve()}, prompts in {PROMPT_DIR.resolve()}")
    print("Condition prompts: " + ", ".join(f"experiment1d_{c} {condition_counts[c]}" for c in CONDITIONS))
    write_runner_files(built)
    if failed:
        print(f"Failed (left out of the CSV): {failed}", file=sys.stderr)


if __name__ == "__main__":
    main()

