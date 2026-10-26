"""Shared by build_experiment2a.py / build_experiment2b.py: rewrite a setup CSV without losing the
rows of sessions that have no graph in new_annotated_graphs/ (their graph is gone, so they can't
be rebuilt, but their prepared prompts / clips / results still refer to those rows)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the code/ folder
import csv
import glob
import os


def has_graph(graph_dir, participant_id, session_id):
    """participant_id like 'p12', session_id like 'session1' (or 'P12' / '1')."""
    num = str(participant_id).lower().lstrip("p")
    session = str(session_id).lower().replace("session", "")
    return bool(glob.glob(os.path.join(graph_dir, f"session{session}_p{num}_*.json")))


def read_rows(path):
    if not os.path.isfile(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def merge_keep_old_only(new_rows, out_path, fieldnames, graph_dir):
    """Return (rows, fieldnames) to write: existing rows of sessions with no graph are kept
    verbatim; existing rows of sessions with a graph are replaced by the freshly built row with
    the same experiment_id (and dropped when the build no longer produces it); new ids are
    appended. Columns the build doesn't produce (e.g. truncated_video_filename) are kept for a
    replaced row when its timestamp_ranges did not change."""
    old_rows, old_fields = read_rows(out_path)
    fields = list(fieldnames) + [c for c in old_fields if c not in fieldnames]
    new_by_id = {r["experiment_id"]: r for r in new_rows}
    merged, kept, used = [], 0, set()
    for old in old_rows:
        if not has_graph(graph_dir, old["participant_id"], old["session_id"]):
            merged.append(old)
            kept += 1
            continue
        new = new_by_id.get(old["experiment_id"])
        if new is None:
            continue
        used.add(old["experiment_id"])
        row = dict(new)
        if (old.get("timestamp_ranges") or "") == (new.get("timestamp_ranges") or ""):
            for col in old_fields:
                if col not in fieldnames and col in old:
                    row[col] = old[col]
        merged.append(row)
    for r in new_rows:
        if r["experiment_id"] in used:
            continue
        if not has_graph(graph_dir, r["participant_id"], r["session_id"]):
            continue  # no graph -> no timestamp ranges; the session's existing rows (above) stay as they are
        merged.append(r)
    print(f"Kept {kept} existing row(s) of sessions with no graph in {os.path.basename(graph_dir)}/.")
    return merged, fields
