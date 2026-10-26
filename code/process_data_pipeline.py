"""
Run the local data-processing pipeline for experiment1a, experiment1b, or
experiment2a.

For experiment1a / experiment1b:

  1. Extract user-response labels from any new_annotated_graphs/*.json that
     doesn't have a CSV yet in labels_experiment<N>/ -- via
     experiment1a_extract_user_response_labels.py,
     or experiment1b_extract_user_response_labels.py (chosen by --setting). Graphs
     that already have a CSV there are left untouched.
  2. Truncate videos for every row across all of that setting's CSVs, via
     truncate_videos_for_setting.py -- it already skips any clip that
     already exists on disk, so this is safe (and cheap) to re-run.
  3. Generate experiment prompts, via prompt_maker_setting.py --setting <N>
     -- it already skips any prompt file that already exists.

For experiment2a there is no label-extraction step. Instead:

  1. Rebuild extracted_timestamps/experiment2a/basic_experiment_setup.csv, via
     experiment_setup/build_experiment2a.py -- only if that script exists
     next to this checkout (it needs workflow_docs/,
     new_annotated_graphs/, and final_participants_for_iclr.csv, which is a
     local-only data-prep step; on a server checkout that only has
     code/, this step is skipped with a note, and whatever
     basic_experiment_setup.csv was copied over is used as-is).
  2. Truncate videos for every row in that CSV, via
     truncate_videos_for_setting.py --setting experiment2a (also writes the
     truncated_video_filename column back into the CSV).
  3. Generate prompts, via prompt_maker_setting.py --setting experiment2a.

Before generating prompts (for all three settings), this also deletes any
already-generated prompt file that got stuck with the "a professional"
placeholder occupation (from before prompt_maker_setting.py's
final_participants_for_iclr.csv fallback existed, or from a participant
missing there too) -- since prompt_maker_setting.py otherwise leaves an
existing prompt file untouched, that placeholder would never get corrected
on a re-run. Deleting it here lets that same re-run regenerate it with the
right occupation.

For experiment1a / experiment1b, unless --base-only is given, the graph/chat ablation
steps follow the base steps:

  4. Chat alignment: build_chat_anchors.py -> align_chats.py. align_chats.py
     always runs (it aligns every anchor row with a filled video_first_mmss,
     including sessions that have no experiment2a rating rows);
     build_chat_anchors.py's exit code 10 ("nothing to run") is gated on
     experiment2a's setup CSV, so it is not treated as "skip the rest".
     build_chat_snippets.py is experiment2a-only and is not run here.
  5. Ablation prompts, one prompt_maker_setting.py call per condition:
     experiment<ID>_{video_graph, graph_only, video_chat, chat_only, graph_chat,
     video_graph_chat}. Existing prompt files are skipped (--force-prompts
     overwrites the ablation prompts only, never the base ones). The graph
     conditions need no separate setup step (that is experiment2a only).
     Note: these prompts can be generated but not yet sent to a model
     (run_qwen/gemini/gpt_setting.py have no --condition flag).

The experiment2a path is unchanged and has no graph/chat steps here (use
bash_scripts/run_prep_pipeline.sh for those).

This never calls a model API (Gemini/GPT/Kimi) -- it only prepares local
data: CSVs, truncated video clips, and prompt text files.

Usage:
    python process_data_pipeline.py --setting 1a
    python process_data_pipeline.py --setting 1b
    python process_data_pipeline.py --setting experiment2a
    python process_data_pipeline.py --setting 1a --base-only        # skip the chat/ablation steps
    python process_data_pipeline.py --setting 1a --force-prompts    # overwrite existing ablation prompts
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from config import ANNOTATED_GRAPHS_DIR, CODE_DIR, EXTRACTED_DIR, ROOT_DIR

PROMPTS_DIR = EXTRACTED_DIR / "prompts"
BUILD_EXPERIMENT2A_SCRIPT = CODE_DIR / "experiment_setup" / "build_experiment2a.py"

SESSION_P_RE = re.compile(r"session\d+_p\d+")
# prompt file names: p6_s1_exp4.txt (settings) or p1_session1_step2_basic_setup.txt (2a/2b)
PROMPT_SESSION_RE = re.compile(r"^(p\d+)_s(?:ession)?(\d+)_")

# Experiment 1a/1b ablation conditions, same order as bash_scripts/run_prep_pipeline.sh.
SETTING_MODES = ("video_graph", "graph_only", "video_chat", "chat_only", "graph_chat", "video_graph_chat")
CHAT_NOTHING_TO_RUN = 10  # build_chat_anchors.py's "nothing to build" exit code

# The exact fragment prompt_maker_setting.py's templates produce when a
# participant's occupation couldn't be resolved (from the behavior graph or
# from final_participants_for_iclr.csv) -- see build_prompt()'s and
# build_experiment2a_prompt()'s "professional" fallback.
PLACEHOLDER_SNIPPET = "of a professional working with an AI agent"


def profession_line(text):
    """The single line naming the profession, e.g. 'You are given a video
    snippet of an Attorney working with an AI agent toward the following
    goal:' -- for printing an old-vs-new comparison."""
    for line in text.splitlines():
        if "video snippet of" in line:
            return line.strip()
    return None


def remove_placeholder_prompts(prompts_dir):
    """Delete any already-generated prompt .txt file that still has the
    generic "a professional" placeholder occupation, so prompt_maker_setting.py
    (which otherwise leaves existing prompt files untouched) regenerates it
    with the correct occupation on this same run. Returns the list of removed
    paths, so the caller can print each one's new (regenerated) line too.

    A prompt whose session has no graph in new_annotated_graphs/ is kept: it could not be
    regenerated, so deleting it would lose it."""
    if not prompts_dir.is_dir():
        return []
    removed_paths = []
    for prompt_path in sorted(prompts_dir.glob("*.txt")):
        text = prompt_path.read_text(encoding="utf-8")
        if PLACEHOLDER_SNIPPET in text:
            name_match = PROMPT_SESSION_RE.match(prompt_path.name)
            if name_match and not any(
                ANNOTATED_GRAPHS_DIR.glob(f"session{name_match.group(2)}_{name_match.group(1)}_*.json")
            ):
                print(f"Keeping placeholder-occupation prompt (no graph to regenerate it): {prompt_path.name}")
                continue
            print(f"Removing placeholder-occupation prompt: {prompt_path.name}")
            print(f"  old: {profession_line(text)}")
            prompt_path.unlink()
            removed_paths.append(prompt_path)
    return removed_paths


def print_regenerated_lines(removed_paths):
    """After prompt_maker_setting.py has re-run, print each just-regenerated
    file's new profession line, so the fix is visible line-by-line rather
    than just a count."""
    for prompt_path in removed_paths:
        if prompt_path.exists():
            print(f"  new ({prompt_path.name}): {profession_line(prompt_path.read_text(encoding='utf-8'))}")
        else:
            print(f"  new ({prompt_path.name}): MISSING -- was not regenerated")


def find_new_graphs(csv_dir):
    """Graphs in new_annotated_graphs/ whose 'session<N>_p<M>' id has no CSV yet in csv_dir."""
    existing_ids = set()
    if csv_dir.is_dir():
        for csv_path in csv_dir.glob("*.csv"):
            match = SESSION_P_RE.search(csv_path.stem)
            if match:
                existing_ids.add(match.group(0))

    new_graphs = []
    for graph_path in sorted(ANNOTATED_GRAPHS_DIR.glob("*.json")):
        match = SESSION_P_RE.search(graph_path.stem)
        if not match:
            print(f"Warning: could not find 'session<N>_p<M>' in {graph_path.name!r}; skipping.", file=sys.stderr)
            continue
        if match.group(0) not in existing_ids:
            new_graphs.append(graph_path)
    return new_graphs


def run_chat_prep():
    """build_chat_anchors.py -> align_chats.py, for the experiment1a/experiment1b chat conditions.

    build_chat_anchors.py appends/fills anchors for every session in new_annotated_graphs/, but its
    exit code (10 = "nothing to run") is gated on experiment2a's annotated_graph_experiment_setup.csv,
    so it says 10 for a session that is annotated yet has no 2a rating rows even though its anchor
    was just filled and it has no aligned chat. align_chats.py aligns every anchor row with a
    filled video_first_mmss regardless, so it always runs; exit 10 is not an error.
    build_chat_snippets.py (experiment2a's per-step chat_snippets.csv) is not needed here:
    experiment1a/experiment1b chat prompts read the aligned chat files directly."""
    cmd = [sys.executable, str(CODE_DIR / "chat" / "build_chat_anchors.py")]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode not in (0, CHAT_NOTHING_TO_RUN):
        raise SystemExit(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    run([sys.executable, str(CODE_DIR / "chat" / "align_chats.py")])


def run_ablation_prompts(setting_name, force):
    """One prompt_maker_setting.py call per condition: --setting <setting_name>_<mode>."""
    for mode in SETTING_MODES:
        cmd = [sys.executable, str(CODE_DIR / "prompts" / "prompt_maker_setting.py"), "--setting", f"{setting_name}_{mode}"]
        if force:
            cmd.append("--force")
        run(cmd)


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--setting", required=True, choices=["1a", "1b", "experiment2a"], help="Which setting's pipeline to run"
    )
    parser.add_argument(
        "--base-only", action="store_true",
        help="experiment1a/3: only the base (video-only) steps; skip chat alignment and ablation prompts",
    )
    parser.add_argument(
        "--force-prompts", action="store_true",
        help="experiment1a/3: overwrite already-generated ablation prompt files (base prompts are never overwritten)",
    )
    args = parser.parse_args()

    if args.setting == "experiment2a":
        print("=== Step 0: rebuild basic_experiment_setup.csv (experiment2a) ===")
        if BUILD_EXPERIMENT2A_SCRIPT.is_file():
            run([sys.executable, str(BUILD_EXPERIMENT2A_SCRIPT)])
        else:
            print(
                f"Note: {BUILD_EXPERIMENT2A_SCRIPT} not found (this looks like a server checkout without "
                f"code/); skipping rebuild and using basic_experiment_setup.csv as-is."
            )

        print("\n=== Step 1: truncate videos (experiment2a) ===")
        run([sys.executable, str(CODE_DIR / "truncation" / "truncate_videos_for_setting.py"), "--setting", "experiment2a"])

        removed_paths = remove_placeholder_prompts(PROMPTS_DIR / "experiment2a")
        print(f"\nRemoved {len(removed_paths)} placeholder-occupation prompt(s) (experiment2a).")

        print("\n=== Step 2: generate prompts (experiment2a) ===")
        run([sys.executable, str(CODE_DIR / "prompts" / "prompt_maker_setting.py"), "--setting", "experiment2a"])

        if removed_paths:
            print("\n=== Regenerated occupations (experiment2a) ===")
            print_regenerated_lines(removed_paths)
        return

    setting_name = f"experiment{args.setting}"
    extract_script = CODE_DIR / "labels" / f"{setting_name}_extract_user_response_labels.py"
    csv_dir = ROOT_DIR / "extracted_timestamps" / f"labels_{setting_name}"

    print(f"=== Step 1: extract labels ({setting_name}) ===")
    new_graphs = find_new_graphs(csv_dir)
    if not new_graphs:
        print("No new graphs to extract; every new_annotated_graphs/*.json already has a CSV here.")
    for graph_path in new_graphs:
        run([sys.executable, str(extract_script), "--input", str(graph_path)])

    print(f"\n=== Step 2: truncate videos ({setting_name}) ===")
    run([sys.executable, str(CODE_DIR / "truncation" / "truncate_videos_for_setting.py"), "--setting", setting_name])

    removed_paths = remove_placeholder_prompts(PROMPTS_DIR / setting_name)
    print(f"\nRemoved {len(removed_paths)} placeholder-occupation prompt(s) ({setting_name}).")

    print(f"\n=== Step 3: generate prompts ({setting_name}) ===")
    run([sys.executable, str(CODE_DIR / "prompts" / "prompt_maker_setting.py"), "--setting", args.setting])

    if removed_paths:
        print(f"\n=== Regenerated occupations ({setting_name}) ===")
        print_regenerated_lines(removed_paths)

    if not args.base_only:
        print("\n=== Step 4: chat alignment ===")
        run_chat_prep()
        print(f"\n=== Step 5: ablation prompts ({setting_name}) ===")
        run_ablation_prompts(setting_name, args.force_prompts)


if __name__ == "__main__":
    main()
