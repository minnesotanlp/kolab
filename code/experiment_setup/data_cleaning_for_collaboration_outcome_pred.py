"""
Prepare data for the "predict collaboration outcome" experiment.

For each step (in the annotated behavior graphs) with >= 2 human->AI edges,
cuts two video clips -- one covering the first ceil(N/2) AI interactions
("50%"), one covering the whole step ("100%") -- and builds a CSV of that
step's collaboration-outcome ground truth, parsed from the participant's
workflow docx.

Scope: only annotated_graphs/*.json (the manually-curated graphs with real
nodes/edges/steps). Not every participant has been through that annotation
pass yet -- this only covers whoever is in that folder.

Human-AI edge definition (same as the earlier statistics): within a step, an
edge whose "from" node has actor_type "user" and "to" node has actor_type
"ai". For a step with N such edges (ordered by the AI node's own start
time), the "50%" clip runs from the step's own start to the end of the
ceil(N/2)-th AI node; the "100%" clip runs from the step's start to the
step's own end. Steps with N < 2 are skipped (no distinct 50/100 split
possible).

Video clips are written to:
    extracted_timestamps/truncated_videos/collab_exp/
        p<participant_num>_session<S>_50percent_step<stepID>.mp4
        p<participant_num>_session<S>_100percent_step<stepID>.mp4
(<stepID> is the graph's own step badge number, e.g. "step3".)

Ground truth is pulled from the matching SESSION's workflow docx (ratings are
session-specific, so a same-participant different-session docx cannot
substitute the way occupation could):
    - ai_completed_task / ai_completed_task_reason: parsed from "Did the AI
      Agent successfully complete the task?" and the accompanying "Why?"
    - rating1..rating7: the 7 "Collaboration satisfaction" statement scores,
      in the order they appear in the doc
    - collab_rating: "Rate the overall collaboration experience from 1-5"

Docx formatting varies across participants -- observed variants handled:
  - bracketed ratings: "[5] The agent understood..."
  - bare-number ratings: "5 The agent understood..."
  - unfilled-placeholder ratings with the score appended at the end:
    "[1-5 rating] The agent understood...5"
  - Yes/No answered inline ("...task? Yes"), left as the unfilled template
    ("...task? Yes/No" -- the real answer is then in the next "Why?" line),
    or embedded together in one paragraph via a literal newline.

Because some docs have missing/malformed "Step N" headers (e.g. p21 session1
implies a "Step 6" between its Step 5 and Step 7 headers, and its graph has
8 step-segments against the 7 step-blocks found in the docx), a graph is
only processed if the number of step-blocks found in its docx EXACTLY
matches the number of steps in the graph. Pairing is then positional (Nth
docx step-block <-> Nth graph step, by chronological/list order) -- NOT by
the graph's own "badge" number, since badges can be non-monotonic or
duplicated (also observed in p21 session1). A count mismatch, or no matching
docx at all (e.g. p6 has no session1 docx), skips that whole graph with a
warning rather than risk a wrong pairing.

Usage:
    python data_cleaning_for_collaboration_outcome_pred.py
    python data_cleaning_for_collaboration_outcome_pred.py --output custom.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

import docx
from tqdm import tqdm

from prompts.prompt_maker_setting import split_time_range, to_seconds, to_timestamp

from config import ROOT_DIR
ANNOTATED_GRAPHS_DIR = ROOT_DIR / "annotated_graphs"
WORKFLOW_DOCS_DIR = ROOT_DIR / "workflow_docs"
VIDEOS_OUT_DIR = ROOT_DIR / "extracted_timestamps" / "truncated_videos" / "collab_exp"
DEFAULT_OUTPUT = ROOT_DIR / "extracted_timestamps" / "collab_exp" / "collab_outcome_labels.csv"

NUM_RATINGS = 7


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def graph_steps(graph):
    """Return the graph's steps in their own (chronological) list order, each
    as {step_id, step_name, start_seconds, end_seconds, nodes, edges}."""
    steps = []
    for position, step in enumerate(graph.get("steps", []), start=1):
        badge_match = re.search(r"\d+", step.get("badge", ""))
        step_id = int(badge_match.group(0)) if badge_match else position
        steps.append(
            {
                "step_id": step_id,
                "step_name": step.get("step_name", ""),
                "start_seconds": step["start_seconds"],
                "end_seconds": step["end_seconds"],
                "nodes": step.get("nodes", []),
                "edges": step.get("edges", []),
            }
        )
    return steps


def node_time_bounds(node):
    time_field = node.get("time")
    ranges = [time_field] if isinstance(time_field, str) else list(time_field or [])
    if not ranges:
        return None
    start_str, _ = split_time_range(ranges[0])
    _, end_str = split_time_range(ranges[-1])
    return to_seconds(start_str), to_seconds(end_str)


def user_to_ai_nodes_in_order(step):
    """AI nodes that are the target of a user->ai edge in this step, ordered
    by the AI node's own start time (chronological)."""
    actor_type = {n["id"]: n.get("actor_type") for n in step["nodes"]}
    nodes_by_id = {n["id"]: n for n in step["nodes"]}

    ai_nodes = []
    for edge in step["edges"]:
        if actor_type.get(edge.get("from")) == "user" and actor_type.get(edge.get("to")) == "ai":
            ai_node = nodes_by_id.get(edge.get("to"))
            if ai_node is not None:
                ai_nodes.append(ai_node)

    def sort_key(node):
        bounds = node_time_bounds(node)
        return bounds[0] if bounds else 0

    ai_nodes.sort(key=sort_key)
    return ai_nodes


# ---------------------------------------------------------------------------
# Workflow docx parsing
# ---------------------------------------------------------------------------

STEP_HEADER_RE = re.compile(r"^Step[\s\-]*\d+.*?:", re.IGNORECASE)
AI_TASK_Q_RE = re.compile(r"successfully complete the task", re.IGNORECASE)
COLLAB_HEADER_RE = re.compile(r"collaboration satisfaction", re.IGNORECASE)
OVERALL_RATING_RE = re.compile(r"rate the overall collaboration experience", re.IGNORECASE)


def parse_rating_line(text):
    """Return (rating:int|None, statement:str) trying the observed formats."""
    text = text.strip()

    m = re.match(r"^\[(\d)\]\s*(.+)$", text)
    if m:
        return int(m.group(1)), m.group(2).strip()

    m = re.match(r"^\[1-5 rating\]\s*(.+?)\s*(\d)\s*$", text, re.IGNORECASE)
    if m:
        return int(m.group(2)), m.group(1).strip()

    m = re.match(r"^(\d)\s+(.+)$", text)
    if m:
        return int(m.group(1)), m.group(2).strip()

    return None, text


def parse_yes_no_and_reason(anchor_text, next_text):
    after_q = anchor_text.split("?", 1)[1].strip() if "?" in anchor_text else ""
    is_placeholder = bool(re.match(r"^\[?\s*Yes\s*/\s*No\s*\]?\s*$", after_q, re.IGNORECASE))

    answer = None
    if after_q and not is_placeholder:
        m = re.match(r"(Yes|No)\b", after_q, re.IGNORECASE)
        if m:
            answer = m.group(1).capitalize()

    search_text = anchor_text + "\n" + (next_text or "")
    reason = ""
    m = re.search(r"Why[^\n?]*\?\s*(.*)", search_text, re.IGNORECASE | re.DOTALL)
    if m:
        reason = m.group(1).strip()

    if answer is None and reason:
        m2 = re.match(r"(Yes|No)\b", reason, re.IGNORECASE)
        if m2:
            answer = m2.group(1).capitalize()

    return answer, reason


def parse_docx_steps(docx_path):
    """Return a list of step-blocks found in the docx, in document order, each
    as {ai_completed_task, ai_completed_task_reason, ratings (list of
    NUM_RATINGS ints or None), collab_rating (int or None)}."""
    document = docx.Document(str(docx_path))
    paragraphs = [p.text for p in document.paragraphs]

    anchor_indices = [i for i, t in enumerate(paragraphs) if AI_TASK_Q_RE.search(t)]

    blocks = []
    for anchor_i in anchor_indices:
        next_text = paragraphs[anchor_i + 1] if anchor_i + 1 < len(paragraphs) else ""
        ai_completed_task, reason = parse_yes_no_and_reason(paragraphs[anchor_i], next_text)

        # Find the "Collaboration satisfaction" header searching forward from
        # this anchor (bounded, so we don't wander into the next step's block).
        collab_header_i = None
        for j in range(anchor_i, min(anchor_i + 15, len(paragraphs))):
            if COLLAB_HEADER_RE.search(paragraphs[j]):
                collab_header_i = j
                break

        ratings = [None] * NUM_RATINGS
        collab_rating = None
        if collab_header_i is not None:
            found = 0
            j = collab_header_i + 1
            while j < len(paragraphs) and found < NUM_RATINGS:
                text = paragraphs[j].strip()
                if not text:
                    j += 1
                    continue
                if OVERALL_RATING_RE.search(text) or STEP_HEADER_RE.match(text):
                    break
                value, _statement = parse_rating_line(text)
                if value is not None:
                    ratings[found] = value
                    found += 1
                j += 1

            for k in range(collab_header_i, min(collab_header_i + NUM_RATINGS + 5, len(paragraphs))):
                m = OVERALL_RATING_RE.search(paragraphs[k])
                if m:
                    num_match = re.search(r":\s*(\d)\D*$", paragraphs[k].strip())
                    if num_match:
                        collab_rating = int(num_match.group(1))
                    break

        blocks.append(
            {
                "ai_completed_task": ai_completed_task,
                "ai_completed_task_reason": reason,
                "ratings": ratings,
                "collab_rating": collab_rating,
            }
        )

    return blocks


def find_workflow_docx(participant_id, session):
    """Find the workflow docx for this EXACT participant+session (no
    cross-session fallback -- collaboration ratings are session-specific)."""
    num_match = re.search(r"\d+", participant_id)
    num = num_match.group(0)
    p_re = re.compile(rf"^p0*{num}(?:[^0-9]|$)", re.IGNORECASE)
    session_re = re.compile(rf"session[^0-9]*0*{session}(?:[^0-9]|$)", re.IGNORECASE)

    candidates = [
        f for f in WORKFLOW_DOCS_DIR.glob("*.docx")
        if p_re.match(f.stem) and session_re.search(f.stem)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda f: len(f.name))[0]


# ---------------------------------------------------------------------------
# Video truncation
# ---------------------------------------------------------------------------

def find_source_video(session, participant_num):
    path = ROOT_DIR / f"session{session}_videos" / "real_videos" / f"session{session}_p{participant_num}.mp4"
    return path if path.exists() else None


def truncate_clip(source_video, start_seconds, end_seconds, output_path):
    if output_path.exists():
        return "skipped"

    duration = end_seconds - start_seconds
    if duration <= 0:
        print(f"Warning: non-positive duration for {output_path.name}; skipping.", file=sys.stderr)
        return "error"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", to_timestamp(start_seconds),
        "-i", str(source_video),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: ffmpeg failed for {output_path.name}:\n{result.stderr[-1500:]}", file=sys.stderr)
        output_path.unlink(missing_ok=True)
        return "error"
    return "created"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_graph(graph_path, results, stats):
    m = re.match(r"session(\d+)_p(\d+)_behavior_graph_modified\.json", graph_path.name)
    if not m:
        return
    session, p_num = m.group(1), m.group(2)
    participant_id = f"p{p_num}"

    import json
    graph = json.load(open(graph_path))
    steps = graph_steps(graph)

    docx_path = find_workflow_docx(participant_id, session)
    if docx_path is None:
        print(f"SKIP {graph_path.name}: no workflow docx found for {participant_id} session {session}.")
        stats["skipped_graphs"] += 1
        return

    docx_blocks = parse_docx_steps(docx_path)
    if len(docx_blocks) != len(steps):
        print(
            f"SKIP {graph_path.name}: graph has {len(steps)} step(s) but "
            f"{docx_path.name} has {len(docx_blocks)} step-block(s) -- counts must match "
            f"to trust positional pairing."
        )
        stats["skipped_graphs"] += 1
        return

    source_video = find_source_video(session, p_num)
    if source_video is None:
        print(f"SKIP {graph_path.name}: source video not found for session{session}_p{p_num}.")
        stats["skipped_graphs"] += 1
        return

    stats["included_graphs"] += 1

    for step, docx_block in zip(steps, docx_blocks):
        ai_nodes = user_to_ai_nodes_in_order(step)
        n_interactions = len(ai_nodes)

        if n_interactions < 2:
            stats["skipped_steps_low_n"] += 1
            continue

        k = max(1, math.ceil(n_interactions / 2))
        cutoff_50_bounds = node_time_bounds(ai_nodes[k - 1])
        if cutoff_50_bounds is None:
            stats["skipped_steps_error"] += 1
            continue
        cutoff_50 = cutoff_50_bounds[1]
        start = step["start_seconds"]
        cutoff_100 = step["end_seconds"]

        video_50_name = f"p{p_num}_session{session}_50percent_step{step['step_id']}.mp4"
        video_100_name = f"p{p_num}_session{session}_100percent_step{step['step_id']}.mp4"
        video_50_path = VIDEOS_OUT_DIR / video_50_name
        video_100_path = VIDEOS_OUT_DIR / video_100_name

        status_50 = truncate_clip(source_video, start, cutoff_50, video_50_path)
        status_100 = truncate_clip(source_video, start, cutoff_100, video_100_path)
        stats[f"clip_{status_50}"] += 1
        stats[f"clip_{status_100}"] += 1

        row = {
            "exp_id": f"p{p_num}_s{session}_step{step['step_id']}",
            "participant_id": participant_id,
            "session_id": session,
            "step_id": step["step_id"],
            "num_interactions": n_interactions,
            "interactions_50pct": k,
            "video_50pct": video_50_name,
            "video_100pct": video_100_name,
            "ai_completed_task": docx_block["ai_completed_task"],
            "ai_completed_task_reason": docx_block["ai_completed_task_reason"],
            **{f"rating{i+1}": docx_block["ratings"][i] for i in range(NUM_RATINGS)},
            "collab_rating": docx_block["collab_rating"],
        }
        results.append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=False, default=str(DEFAULT_OUTPUT), help="CSV output path")
    args = parser.parse_args()

    graph_paths = sorted(ANNOTATED_GRAPHS_DIR.glob("*.json"))
    results = []
    stats = {
        "included_graphs": 0, "skipped_graphs": 0,
        "skipped_steps_low_n": 0, "skipped_steps_error": 0,
        "clip_created": 0, "clip_skipped": 0, "clip_error": 0,
    }

    for graph_path in tqdm(graph_paths, desc="Processing annotated graphs"):
        process_graph(graph_path, results, stats)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["exp_id", "participant_id", "session_id", "step_id", "num_interactions", "interactions_50pct",
         "video_50pct", "video_100pct", "ai_completed_task", "ai_completed_task_reason"]
        + [f"rating{i+1}" for i in range(NUM_RATINGS)]
        + ["collab_rating"]
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} row(s) to {output_path}")
    print(
        f"Graphs: {stats['included_graphs']} included, {stats['skipped_graphs']} skipped "
        f"(no docx / count mismatch / no source video)"
    )
    print(
        f"Steps: {len(results)} included, {stats['skipped_steps_low_n']} skipped (N<2), "
        f"{stats['skipped_steps_error']} skipped (error)"
    )
    print(
        f"Clips: {stats['clip_created']} created, {stats['clip_skipped']} already existed, "
        f"{stats['clip_error']} failed"
    )


if __name__ == "__main__":
    main()
