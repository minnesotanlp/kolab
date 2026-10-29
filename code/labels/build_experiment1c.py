"""
Experiment 1c (context levels): does seeing more of the workflow before an AI action help predict how
the user responds to it?

Only experiment 1b instances whose source node is in the session's final step are used, and only
those whose four clips start at four different times (a session with too few steps would repeat
a window, so it is left out). Each one gets
four clips that all end where the target AI action ends and start progressively earlier:

    level   clip begins at                          same clip as
    last1   start of the final step                 (cut fresh, or experiment 1b's clip when the source node
                                                    is the first node of the final step)
    last2   start of the second-to-last step
    last3   start of the third-to-last step
    full    00:00:00                                build_experiment1d.py's experiment 1d clips (full workflow)

Steps come from the annotated graph's "steps" (start_seconds); the source node's step is found as
(the last step starting at or before the source node).
Candidates are build_experiment1d.py's: experiment 1b's instances re-derived from annotated_graphs/,
checked against labels_experiment1b, one per distinct AI output, conflicting-label groups dropped.

Selection: every eligible instance. --ids takes exactly the given experiment_ids instead.

Clips: an existing clip with the same window (experiment 1b clip for last1, build_experiment1d.py
clip for full) is symlinked (if already 1 fps); anything else
is re-encoded from the source video (truncate_videos_for_setting.truncate_clip), --jobs at a time.
Every clip is 1 fps (--fps): existing clips are only reused or kept when they are already at that rate,
so the old 30 fps / stream-copied clips get re-cut.
The clip folders live under --video-root (the root disk is nearly full) and are symlinked from
extracted_timestamps/truncated_videos/experiment1c_<level>.

Outputs, per level (named so run_*_setting.py can treat "5_<level>" as a setting):
    extracted_timestamps/truncated_videos/experiment1c_<level>/session<S>_p<N>_<begin>_<end>.mp4
    extracted_timestamps/prompts/experiment1c_<level>/<p>_s<S>_exp<i>.txt             video-only
    extracted_timestamps/prompts/experiment1c_<level>_chat_only/<p>_s<S>_exp<i>.txt   chat-only (only for
                                                                                sessions with aligned chat)
    extracted_timestamps/prompts/experiment1c_<level>_graph_only/<p>_s<S>_exp<i>.txt  graph-only (every instance)
    extracted_timestamps/labels_experiment1c_<level>/session<S>_p<N>_timestamp_labels.csv
and one row per (instance, level) in
    extracted_timestamps/experiment1c/basic_setup.csv

Usage:
    python build_experiment1c.py --dry-run                # print the instances, write nothing
    python build_experiment1c.py                          # build every eligible instance
    python build_experiment1c.py --graph-only-prompts      # just the graph-only prompts (no clip work)
    python build_experiment1c.py --ids my_picks.txt       # exactly these experiment_ids
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder
from config import CLIP_DIR

import argparse
import csv
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from labels import build_experiment1d as s5
from prompts import prompt_maker_setting as pm
from chat.chat_common import aligned_chat_path
from truncation.truncate_videos_for_setting import find_source_video, truncate_clip

TIMESTAMPS_DIR = s5.TIMESTAMPS_DIR
OUTPUT_CSV = TIMESTAMPS_DIR / "experiment1c" / "basic_setup.csv"
LEVELS = ("last1", "last2", "last3", "full")
FPS = 1  # every clip is cut at this frame rate (--fps)
CONDITION = "chat_only"

FIELDNAMES = [
    "experiment_id", "participant_id", "session_id", "label", "n_steps", "level", "steps_included",
    "begin_timestamp", "end_timestamp", "target_ai_timestamp", "clip_length_seconds",
    "video_file_name", "video_source", "prompt_file_name", "has_chat_prompt",
]


def step_starts(graph):
    return sorted(s["start_seconds"] for s in graph.get("steps", []) if s.get("start_seconds") is not None)


def final_step_candidates():
    """build_experiment1d's de-duplicated experiment 1b instances whose source node is in the final step,
    with each level's clip start (seconds) added."""
    out = []
    for c in s5.dedupe(s5.load_candidates()):
        starts = step_starts(c["graph"])
        source = s5.to_seconds(c["experiment1b_begin_timestamp"])
        before = [i for i, s in enumerate(starts) if s <= source]
        if not before or before[-1] != len(starts) - 1:
            continue
        begins = {"full": 0}
        for k in (1, 2, 3):
            if len(starts) >= k:
                begins[f"last{k}"] = starts[-k]
        if len(set(begins.values())) < len(LEVELS):  # fewer than 4 distinct clip windows
            continue
        out.append({**c, "n_steps": len(starts), "begins": begins,
                    "stem": Path(pm.prompt_filename(c["experiment_id"], c["participant_id"], c["session"])).stem,
                    "has_chat": aligned_chat_path(c["participant_id"], f"session{c['session']}").exists()})
    return out


def clip_name(c, begin_seconds):
    begin = pm.to_timestamp(begin_seconds).replace(":", "-")
    return f"session{c['session']}_{c['participant_id']}_{begin}_{c['end_timestamp'].replace(':', '-')}.mp4"


def clip_fps(path):
    """The video stream's frame rate, or None if it can't be read."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def has_fps(path):
    return clip_fps(path) == FPS


def existing_clip(c, level):
    """An already-cut clip with exactly this level's window and the target frame rate, or None."""
    candidates = []
    if level == "full":
        candidates.append(s5.VIDEO_DIR / f"{c['stem']}.mp4")
    elif level == "last1":
        candidates += [TIMESTAMPS_DIR / "truncated_videos" / f"experiment{setting}" / clip_name(c, c["begins"]["last1"])
                       for setting in ("1b",)]
    return next((p for p in candidates if p.exists() and has_fps(p)), None)


def level_dirs(level, video_root):
    video_dir = TIMESTAMPS_DIR / "truncated_videos" / f"experiment1c_{level}"
    if not video_dir.exists():
        target = video_root / f"experiment1c_{level}"
        target.mkdir(parents=True, exist_ok=True)
        video_dir.symlink_to(target)
    return video_dir


def make_clip(job):
    """(candidate, level, output path) -> how it was made, or None on failure."""
    c, level, output = job
    if output.exists():
        if has_fps(output):
            return "kept"
        output.unlink()  # wrong frame rate (or an unreadable file / symlink to one): re-make it
    reuse = existing_clip(c, level)
    if reuse is not None:
        output.symlink_to(reuse.resolve())
        return f"link:{reuse.parent.name}"
    source = find_source_video(c["session"], c["participant_id"][1:])
    if source is None:
        print(f"Warning: no source video for {c['experiment_id']}", file=sys.stderr)
        return None
    begin = c["begins"][level]
    ok = truncate_clip(source, pm.to_timestamp(begin), c["end_timestamp"], output, fps=FPS)
    return f"cut:{FPS}fps" if ok else None


def write_prompt(c, level, condition, output):
    result = pm.build_prompt(
        c["graph"], pm.build_node_index(c["graph"]), c["user_action_node"], c["target_ai_timestamp"],
        pm.to_timestamp(c["begins"][level]), c["end_timestamp"], c["participant_id"], c["session"],
        "1b", OCCUPATIONS, condition=f"experiment1b_{condition}" if condition else None,
    )
    if result is None:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result[1] + "\n")
    return True


def write_label_csvs(level, selected):
    labels_dir = TIMESTAMPS_DIR / f"labels_experiment1c_{level}"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for old in labels_dir.glob("*.csv"):
        old.unlink()
    groups = {}
    for c in selected:
        if level in c["begins"]:
            groups.setdefault((c["session"], c["participant_id"]), []).append(c)
    for (session, participant_id), group in groups.items():
        with open(labels_dir / f"session{session}_{participant_id}_timestamp_labels.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=s5.LABEL_FIELDNAMES)
            writer.writeheader()
            for c in sorted(group, key=lambda c: c["exp_num"]):
                writer.writerow({"experiment_id": c["experiment_id"], "user_action_node": c["user_action_node"],
                                 "ai_node": c["ai_node"], "begin_timestamp": pm.to_timestamp(c["begins"][level]),
                                 "end_timestamp": c["end_timestamp"], "label": c["label"],
                                 "target_ai_timestamp": c["target_ai_timestamp"]})


def print_plan(selected):
    print(f"\n{len(selected)} instance(s): " + ", ".join(f"{k} {v}" for k, v in Counter(c['label'] for c in selected).items()))
    print(f"{'experiment_id':16s} {'label':7s} steps chat  " + "  ".join(f"{lv:>12s}" for lv in LEVELS))
    for c in selected:
        cells = []
        for lv in LEVELS:
            if lv not in c["begins"]:
                cells.append(f"{'-':>12s}")
                continue
            minutes = (c["length"] - c["begins"][lv]) / 60
            cells.append(f"{minutes:6.1f}m {'reuse' if existing_clip(c, lv) else 'enc':>5s}")
        print(f"{c['experiment_id']:16s} {c['label']:7s} {c['n_steps']:5d} {'yes' if c['has_chat'] else 'no':4s}  " + "  ".join(cells))


def main():
    global OCCUPATIONS, FPS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", type=Path, default=None, help="file of experiment_ids (one per line, or a CSV)")
    parser.add_argument("--jobs", type=int, default=3, help="clips cut in parallel (default 3)")
    parser.add_argument("--video-root", type=Path, default=CLIP_DIR,
                        help="where new experiment1c_<level> clip folders are created (default: $KOLAB_CLIP_DIR, else <data>/clips)")
    parser.add_argument("--fps", type=float, default=FPS,
                        help="frame rate of every clip (default 1); clips already at another rate are re-made")
    parser.add_argument("--graph-only-prompts", action="store_true",
                        help="only (re)write the graph-only prompts; no clips, no setup CSV, no label CSVs")
    parser.add_argument("--dry-run", action="store_true", help="print the instances; cut and write nothing")
    args = parser.parse_args()
    FPS = args.fps

    pool = final_step_candidates()
    print(f"{len(pool)} final-step instance(s): " + ", ".join(f"{k} {v}" for k, v in Counter(c['label'] for c in pool).items()))
    by_id = {c["experiment_id"]: c for c in pool}
    wanted = None
    if args.ids:
        wanted = s5.read_ids(args.ids)
    if wanted is not None:
        missing = [i for i in wanted if i not in by_id]
        if missing:
            sys.exit(f"Not final-step instances (or not in the de-duplicated experiment 1b set): {missing}")
        selected = [by_id[i] for i in dict.fromkeys(wanted)]
    else:
        selected = list(pool)
    selected.sort(key=lambda c: (int(c["participant_id"][1:]), c["session"], c["exp_num"]))
    print_plan(selected)
    if args.dry_run:
        return

    OCCUPATIONS = pm.load_occupations()
    if args.graph_only_prompts:
        made = sum(write_prompt(c, level, "graph_only", TIMESTAMPS_DIR / "prompts" / f"experiment1c_{level}_graph_only" / f"{c['stem']}.txt")
                   for c in selected for level in LEVELS if level in c["begins"])
        print(f"Wrote {made} graph-only prompt(s)")
        return
    jobs = []
    for level in LEVELS:
        video_dir = level_dirs(level, args.video_root)
        jobs += [(c, level, video_dir / clip_name(c, c["begins"][level])) for c in selected if level in c["begins"]]
    print(f"\nMaking {len(jobs)} clip(s), {args.jobs} at a time ...")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool_ex:
        made = list(pool_ex.map(make_clip, jobs))
    print("Clips: " + ", ".join(f"{k} {v}" for k, v in Counter(m or "FAILED" for m in made).items()))

    rows = []
    for (c, level, video_path), how in zip(jobs, made):
        prompt_dir = TIMESTAMPS_DIR / "prompts" / f"experiment1c_{level}"
        video_ok = how is not None and write_prompt(c, level, None, prompt_dir / f"{c['stem']}.txt")
        chat_ok = c["has_chat"] and write_prompt(
            c, level, CONDITION, prompt_dir.parent / f"experiment1c_{level}_{CONDITION}" / f"{c['stem']}.txt")
        write_prompt(c, level, "graph_only", prompt_dir.parent / f"experiment1c_{level}_graph_only" / f"{c['stem']}.txt")
        begin = c["begins"][level]
        rows.append({
            "experiment_id": c["experiment_id"], "participant_id": c["participant_id"],
            "session_id": f"session{c['session']}", "label": c["label"], "n_steps": c["n_steps"],
            "level": level, "steps_included": c["n_steps"] if level == "full" else int(level[-1]),
            "begin_timestamp": pm.to_timestamp(begin), "end_timestamp": c["end_timestamp"],
            "target_ai_timestamp": c["target_ai_timestamp"], "clip_length_seconds": c["length"] - begin,
            "video_file_name": video_path.name if video_ok else "", "video_source": how or "FAILED",
            "prompt_file_name": f"{c['stem']}.txt" if video_ok else "", "has_chat_prompt": chat_ok,
        })
    for level in LEVELS:
        write_label_csvs(level, selected)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s) to {OUTPUT_CSV}")
    failed = [(r["experiment_id"], r["level"]) for r in rows if not r["prompt_file_name"]]
    if failed:
        print(f"Failed (no clip or prompt): {failed}", file=sys.stderr)


OCCUPATIONS = {}

if __name__ == "__main__":
    main()

