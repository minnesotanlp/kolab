"""
Paths shared by every script.

The scripts read and write one data folder (the "root"): extracted_timestamps/ (labels, setup CSVs, clips, prompts),
experiment_results/, new_annotated_graphs/, workflow_docs/, session<N>_videos/ and final_participants_for_iclr.csv.
It is <repo>/data by default; set KOLAB_DATA_DIR to put it somewhere else (e.g. a bigger disk).

KOLAB_CLIP_DIR is where experiment 1c's clip folders are created (they are linked from extracted_timestamps/
truncated_videos/); it defaults to <root>/clips.
"""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent

ROOT_DIR = Path(os.environ.get("KOLAB_DATA_DIR", REPO_DIR / "data")).expanduser().resolve()
EXTRACTED_DIR = ROOT_DIR / "extracted_timestamps"
RESULTS_DIR = ROOT_DIR / "experiment_results"
ANNOTATED_GRAPHS_DIR = ROOT_DIR / "new_annotated_graphs"
RATING_AUDIT_DIR = ROOT_DIR / "rating_audit"
CLIP_DIR = Path(os.environ.get("KOLAB_CLIP_DIR", ROOT_DIR / "clips")).expanduser()
