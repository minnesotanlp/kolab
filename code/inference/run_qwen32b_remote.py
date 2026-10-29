"""
Run every condition of experiment 1a, 1b, 2a and experiment 2b --
the text-only ones (graph-only, chat-only, graph-chat, graph-only-ts) and the video
ones (video-only, video-graph, video-chat, video-graph-chat, video-graph-ts) --
through an already-hosted Qwen3-VL-32B served
behind an OpenAI-compatible API (e.g. `vllm serve Qwen/Qwen3-VL-32B-Instruct`),
instead of loading the weights locally like run_qwen_setting.py /
run_experiment2a.py / run_experiment2b.py do.

Video conditions send the clip as a base64 `video_url` part (data:video/mp4), placed
before the text like the local runners do. The clip is sent as-is, so it should already
be the small ~32-frame version that make_remote_bundle.py writes (the local runners cap
sampling at fps=1, max 32 frames); the full truncated clips are tens of GB.

Samples are collected by those scripts' own collect_samples(), and results go to
the exact same files they would write:

    experiment_results/<folder>/qwen32b_<folder>_results_run<N>.csv

with the same columns, so eval_results.py reads them unchanged. Rows already in
a results file (by experiment_id) are skipped -- copy existing result files over
and only the missing samples are sent.

The server is read from --base-url / --model-name / --api-key, or the env vars
QWEN_BASE_URL, QWEN_MODEL, QWEN_API_KEY. Sampling parameters are not set, so
the server's defaults apply (vLLM uses the model's generation_config, as the
local transformers runs do); max_tokens matches the local MAX_NEW_TOKENS.

Usage:
    python run_qwen32b_remote.py --dry-run                               # counts only, no requests
    python run_qwen32b_remote.py --base-url http://HOST:8000/v1 --runs 1 2 3
    python run_qwen32b_remote.py --experiments 2a --condition graph-only --runs 3 --workers 8
    python run_qwen32b_remote.py --experiments 1 3 --condition video-only video-graph --runs 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import base64
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from inference import run_experiment2a
from inference import run_experiment2b
from inference import run_qwen_setting
from experiments import CONDITION_CHOICES, condition_results_path

MODEL_SLUG = "qwen32b"
EXPERIMENTS = ("1a", "1b", "2a", "2b")
SETTINGS = ("1a", "1b")
# 2a/2b also have the "-ts" variants; "legacy-graph" is the old 126-step run, not offered.
ALL_CONDITIONS = tuple(dict.fromkeys(CONDITION_CHOICES + tuple(c for c in run_experiment2a.CONDITIONS if c != "legacy-graph")))
MAX_TOKENS = run_qwen_setting.MAX_NEW_TOKENS

SETTING_FIELDS = ["participant_id", "experiment_id", "real_answer", "predicted_answer", "video_clip_length", "step_jump"]
EXP2_FIELDS = ["participant_id", "experiment_id", "real_answer", "predicted_answer", "video_clip_length",
               "num_timestamp_ranges"]


def conditions_for(experiment):
    if experiment in SETTINGS:
        return CONDITION_CHOICES
    module = run_experiment2a if experiment == "2a" else run_experiment2b
    return tuple(c for c in module.CONDITIONS if c != "legacy-graph")


def plan_for(experiment, condition):
    """(samples, output path for a run number, CSV fieldnames, extra column name) for one experiment/condition."""
    if experiment in SETTINGS:
        samples = run_qwen_setting.collect_samples(None, None, experiment, condition)
        return samples, (lambda run: condition_results_path(experiment, condition, MODEL_SLUG, run)), SETTING_FIELDS, "step_jump"
    module = run_experiment2a if experiment == "2a" else run_experiment2b
    folder = module.CONDITIONS[condition]["folder"]
    samples = module.collect_samples(None, None, condition)
    output = lambda run: module.ROOT_DIR / "experiment_results" / folder / f"{MODEL_SLUG}_{folder}_results_run{run}.csv"
    return samples, output, EXP2_FIELDS, "num_timestamp_ranges"


def pending(samples, output):
    """(samples whose experiment_id is not yet in the results file, number already done)."""
    done = set()
    if Path(output).exists():
        with open(output, newline="") as f:
            done = {row["experiment_id"] for row in csv.DictReader(f)}
    return [s for s in samples if s["experiment_id"] not in done], len(done)


def ask(client, model_name, prompt_text, video_path=None):
    content = [{"type": "text", "text": prompt_text}]
    if video_path is not None:
        video_b64 = base64.b64encode(Path(video_path).read_bytes()).decode()
        content.insert(0, {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content or ""
    if not answer.strip():
        raise ValueError("empty reply")
    return answer


def run_one(client, args, label, samples, output, fieldnames, extra_field):
    output = Path(output)
    todo, n_done = pending(samples, output)
    print(f"[{label}] {len(samples)} samples, {n_done} already in {output.name}, {len(todo)} to run")
    if args.dry_run or not todo:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    failed = 0
    with open(output, "a", newline="") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        futures = {pool.submit(ask, client, args.model_name, s["prompt_path"].read_text(), s.get("video_path")): s for s in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc=label, unit="prompt"):
            sample = futures[future]
            try:
                answer = future.result()
            except Exception as exc:
                tqdm.write(f"ERROR {sample['experiment_id']}: {exc}")
                failed += 1
                continue
            writer.writerow({
                "participant_id": sample["participant_id"],
                "experiment_id": sample["experiment_id"],
                "real_answer": sample["ground_truth_label"],
                "predicted_answer": answer,
                "video_clip_length": sample["video_clip_length"],
                extra_field: sample[extra_field],
            })
            f.flush()
    print(f"[{label}] wrote {len(todo) - failed}, failed {failed} (rerun to retry failures)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    parser.add_argument("--condition", nargs="+", default=list(ALL_CONDITIONS), choices=ALL_CONDITIONS,
                        help="default: every condition the experiment has")
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL"),
                        help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1 (env QWEN_BASE_URL)")
    parser.add_argument("--model-name", default=os.environ.get("QWEN_MODEL", run_qwen_setting.QWEN_32B_MODEL_NAME),
                        help="Model name the server exposes (env QWEN_MODEL)")
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", "EMPTY"), help="env QWEN_API_KEY")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests to the server (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what is left to run")
    args = parser.parse_args()

    client = None
    if not args.dry_run:
        if not args.base_url:
            parser.error("--base-url (or QWEN_BASE_URL) is required unless --dry-run")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    plans = []
    for experiment in args.experiments:
        for condition in args.condition:
            if condition not in conditions_for(experiment):
                continue
            samples, output_for, fieldnames, extra_field = plan_for(experiment, condition)
            if not samples:
                print(f"No samples for experiment {experiment} {condition}", file=sys.stderr)
                continue
            plans.append((experiment, condition, samples, output_for, fieldnames, extra_field))

    for run in args.runs:
        for experiment, condition, samples, output_for, fieldnames, extra_field in plans:
            label = f"{experiment} {condition} run{run}"
            run_one(client, args, label, samples, output_for(run), fieldnames, extra_field)


if __name__ == "__main__":
    main()
