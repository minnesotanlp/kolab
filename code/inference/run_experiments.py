"""
Run inference for experiment1a, experiment1b, experiment1d, experiment2a and experiment2b, over any mix of input modes
("conditions"), models and run numbers, in ONE process -- so a Qwen model is loaded only once and
reused for everything that needs it.

Each experiment is its own flag and takes the modes to run for it (space separated, or `all`):

    --exp1a MODE [MODE ...]     video-only video-graph graph-only video-chat chat-only graph-chat video-graph-chat
    --exp1b, --exp1c_last1, --exp1c_last2, --exp1c_last3, --exp1c_full, --exp1d   (same seven modes)
    --exp2a    MODE [MODE ...]     the seven above plus video-graph-ts graph-only-ts
    --exp2b    MODE [MODE ...]     (same nine modes)

`all` means the seven modes; the -ts modes run only when named. An experiment whose flag is omitted is not run. `video` is accepted as a shorthand for `video-only`,
and underscores work as well as hyphens (graph_only == graph-only). Modes are the ones the prompts were
prepared for by bash_scripts/run_prep_pipeline.sh (prompt_maker_setting.py); graph-only, chat-only and
graph-chat send no video.

    --models MODEL [MODEL ...]     gemini flash gpt qwen qwen32b   (required)
    --runs N [N ...]               run numbers; every job is repeated per number, writing
                                   ..._results_run<N>.csv (default: one run, no suffix)
    --all                          run every matching sample (default: only --limit, 1, as a smoke test)
    --limit N, --p, --session, --exp-id, --dry-run, --max_frames, --frames

Results go where eval_results.py looks for them:
    experiment_results/<folder>/<model>_<folder>_results[_run<N>].csv
where <folder> is experiment1a, experiment1a_graph_only, ..., experiment2a, experiment2a_video_graph, ...
(model slug for flash is `gemini_flash` for experiment1a/3/5 and `flash` for 2a/2b, as their scripts already do).
Rows already in a results file (by experiment_id) are skipped, so an interrupted run can be re-invoked.

Qwen: one model per name (qwen, qwen32b) is loaded lazily the first time a real (non --dry-run) job for it
comes up and kept for all of that model's jobs across settings, conditions and runs; a second Qwen model
is loaded only after the first is released.

Usage:
    python run_experiments.py --models flash --exp1a video-only graph-only --dry-run
    python run_experiments.py --models qwen --exp1a all --exp1b all --exp2a all --exp2b all --all --runs 1 2 3
    python run_experiments.py --models flash gpt qwen --exp2a video-graph chat-only --all --runs 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import csv
import os
import sys
from argparse import Namespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from inference import run_experiment2a as e2a
from inference import run_experiment2b as e2b
from inference import run_gemini_setting as gemini_setting
from inference import run_gpt_setting as gpt_setting
from inference import run_qwen_setting as qwen_setting
from experiments import CONDITION_CHOICES as SETTING_CONDITION_CHOICES
from experiments import CLASSIFICATION_EXPERIMENTS, QWEN_MAX_FRAMES, condition_folder, condition_results_path

MODELS = ("gemini", "flash", "gpt", "qwen", "qwen32b")
QWEN_MODELS = ("qwen", "qwen32b")
QWEN_LABEL = {"qwen": "Qwen3-VL", "qwen32b": "Qwen3-VL-32B"}
# model slug in the results file name, for experiment1a/3/5 (2a/2b use the model name as is)
SETTING_SLUG = {"gemini": "gemini", "flash": "gemini_flash", "gpt": "gpt", "qwen": "qwen", "qwen32b": "qwen32b"}

# experiment key -> (kind, id, valid conditions)
EXPERIMENTS = {
    **{f"exp{ident}": ("setting", ident, SETTING_CONDITION_CHOICES) for ident in CLASSIFICATION_EXPERIMENTS},
    "exp2a": ("2a", None, tuple(e2a.CONDITION_CHOICES)),
    "exp2b": ("2b", None, tuple(e2b.CONDITION_CHOICES)),
}


def normalize_modes(experiment, raw_modes):
    """['all'] -> every non-timestamp mode (the seven; -ts modes only when named); 'video' -> 'video-only';
    graph_only -> graph-only; de-duplicated, in order."""
    valid = EXPERIMENTS[experiment][2]
    if len(raw_modes) == 1 and raw_modes[0].lower() == "all":
        return [mode for mode in valid if not mode.endswith("-ts")]
    modes = []
    for raw in raw_modes:
        mode = raw.strip().lower().replace("_", "-")
        if mode == "video":
            mode = "video-only"
        if mode not in valid:
            raise SystemExit(f"--{experiment}: unknown mode {raw!r}. Valid: {', '.join(valid)} (or 'all').")
        if mode not in modes:
            modes.append(mode)
    return modes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for experiment in EXPERIMENTS:
        parser.add_argument(f"--{experiment}", nargs="+", default=None, metavar="MODE",
                            help=f"modes to run for {experiment} (see above), or 'all'")
    parser.add_argument("--models", nargs="+", required=True, choices=MODELS, help="which model(s) to run")
    parser.add_argument("--runs", type=int, nargs="+", default=None, metavar="N",
                        help="run number(s); default: a single run written without a _run<N> suffix")
    parser.add_argument("--all", action="store_true", help="run every matching sample (default: only --limit)")
    parser.add_argument("--limit", type=int, default=1, help="samples per (experiment, mode) unless --all (default 1)")
    parser.add_argument("--p", default=None, help="only this participant, e.g. p2")
    parser.add_argument("--session", default=None, choices=["1", "2"], help="only this session")
    parser.add_argument("--exp-id", dest="exp_id", default=None, help="only this one experiment_id")
    parser.add_argument("--dry-run", action="store_true", help="print what would be sent; no API calls, no model load")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="qwen only: cap on sampled video frames per clip (default: "
                             f"{e2a.DEFAULT_MAX_VIDEO_FRAMES}, or experiments.QWEN_MAX_FRAMES for that setting)")
    parser.add_argument("--frames", type=int, default=None, help="gpt only: fixed number of frames per clip")
    args = parser.parse_args()

    args.plan_modes = {}
    for experiment in EXPERIMENTS:
        raw = getattr(args, experiment)
        if raw is not None:
            args.plan_modes[experiment] = normalize_modes(experiment, raw)
    if not args.plan_modes:
        parser.error("nothing to run: give at least one of --exp1a / --exp1b / --exp1c_last1 ... / --exp1d / --exp2a / --exp2b")
    args.models = list(dict.fromkeys(args.models))
    args.runs = list(dict.fromkeys(args.runs)) if args.runs else [None]
    return args


def collect_jobs(args):
    """{(experiment, mode): selected samples}, collected once and reused for every model / run."""
    participant_filter = args.p.strip().lower() if args.p else None
    session_filter = int(args.session) if args.session else None
    jobs = {}
    for experiment, modes in args.plan_modes.items():
        kind, ident, _ = EXPERIMENTS[experiment]
        for mode in modes:
            if kind == "setting":
                label = condition_folder(ident, mode)
                samples = qwen_setting.collect_samples(participant_filter, session_filter, ident, mode)
            else:
                label = f"experiment{kind}" if mode == "video-only" else f"experiment{kind}_{mode.replace('-', '_')}"
                module = e2a if kind == "2a" else e2b
                samples = module.collect_samples(participant_filter, session_filter, mode)
            if not samples:
                print(f"[{label}] no matching samples; skipping.", file=sys.stderr)
                continue
            if args.exp_id:
                selected = [s for s in samples if s["experiment_id"] == args.exp_id]
                if not selected:
                    print(f"[{label}] no sample with experiment_id={args.exp_id!r}; skipping.", file=sys.stderr)
                    continue
            else:
                selected = samples if args.all else samples[: args.limit]
            print(f"[{label}] {len(samples)} sample(s) matched; running {len(selected)} of them.")
            jobs[(experiment, mode)] = selected
    return jobs


csv.field_size_limit(sys.maxsize)  # some answers carry long reasoning


def done_ids(output):
    """experiment_ids already written to a results CSV (the same check the runners do)."""
    if not os.path.exists(output):
        return set()
    with open(output, newline="") as f:
        return {row["experiment_id"] for row in csv.DictReader(f)}


def release_qwen(model):
    del model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    print(f"Models: {', '.join(args.models)}   runs: {', '.join('-' if r is None else str(r) for r in args.runs)}")
    for experiment, modes in args.plan_modes.items():
        print(f"  {experiment}: {', '.join(modes)}")
    print()

    jobs = collect_jobs(args)
    if not jobs:
        print("Nothing to run.", file=sys.stderr)
        return

    for model_name in args.models:
        print(f"\n################ model: {model_name} ################")
        qwen_model = qwen_processor = None
        client = None
        is_qwen = model_name in QWEN_MODELS

        # Resolve every job's output file and drop samples already in it *before* loading the
        # model, so finished jobs cost nothing (no model load, no per-sample skip loop).
        pending_jobs = []
        for run in args.runs:
            for (experiment, mode), selected in jobs.items():
                kind, ident, _ = EXPERIMENTS[experiment]
                if kind == "setting":
                    output = str(condition_results_path(ident, mode, SETTING_SLUG[model_name], run))
                    label = condition_folder(ident, mode)
                else:
                    module = e2a if kind == "2a" else e2b
                    output = module.resolve_output(Namespace(model=model_name, condition=mode, output=None), run)
                    label = f"experiment{kind}" if mode == "video-only" else f"experiment{kind}_{mode.replace('-', '_')}"
                done = done_ids(output)
                pending = [s for s in selected if s["experiment_id"] not in done]
                run_note = "" if run is None else f" | run {run}"
                print(f"  {label}{run_note}: {len(pending)} to run, {len(selected) - len(pending)} already done")
                if pending:
                    pending_jobs.append((run, experiment, mode, pending, output, label))

        if not pending_jobs:
            print(f"Nothing left to run for {model_name}; not loading the model.")
            continue

        try:
            if not args.dry_run:
                if is_qwen:
                    e2a.print_gpu_info()
                    print(f"Loading {QWEN_LABEL[model_name]} once for all jobs of this model ...")
                    qwen_model, qwen_processor = e2a.get_qwen_model_and_processor(e2a.QWEN_MODEL_NAMES[model_name])
                else:
                    client = e2a.get_client()

            for run, experiment, mode, selected, output, label in pending_jobs:
                kind, ident, _ = EXPERIMENTS[experiment]
                run_note = "" if run is None else f" | run {run}"
                print(f"\n===== {model_name} | {label}{run_note} -> {output} =====")
                if kind == "setting":
                    max_frames = args.max_frames or QWEN_MAX_FRAMES.get(ident, e2a.DEFAULT_MAX_VIDEO_FRAMES)
                    ns = Namespace(dry_run=args.dry_run, frames=args.frames, max_frames=max_frames,
                                   use_video_chat3=False)
                    if is_qwen:
                        qwen_setting.run_setting(ident, selected, output, ns, qwen_model, qwen_processor,
                                                 QWEN_LABEL[model_name], mode)
                    elif model_name == "gpt":
                        gpt_setting.run_batch(ident, mode, selected, output, ns, client)
                    else:  # gemini / flash
                        gem = gemini_setting
                        name = gem.FLASH_MODEL if model_name == "flash" else gem.DEFAULT_MODEL
                        slug = gem.FLASH_MODEL_SLUG if model_name == "flash" else gem.DEFAULT_MODEL_SLUG
                        gem.run_batch(ident, mode, selected, output, ns, client, name, slug)
                else:
                    module = e2a if kind == "2a" else e2b
                    ns = Namespace(model=model_name, condition=mode, output=output, dry_run=args.dry_run,
                                   frames=args.frames, max_frames=args.max_frames or e2a.DEFAULT_MAX_VIDEO_FRAMES)
                    module.run_batch(ns, selected, client, qwen_model, qwen_processor)
        finally:
            if qwen_model is not None:
                release_qwen(qwen_model)
                qwen_model = qwen_processor = None

    print("\nAll requested jobs finished." if not args.dry_run else "\n(dry run -- nothing was sent or written)")


if __name__ == "__main__":
    main()
