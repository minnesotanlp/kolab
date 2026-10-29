"""
Send (video, prompt) pairs to GPT through the UMN AI gateway and print/save
the model's predicted label, for experiments 1a-1d (--setting).

Matching video <-> prompt
-------------------------
A video snippet in extracted_timestamps/truncated_videos/experiment<ID>/ is named:

    session<S>_p<N>_<begin_timestamp>_<end_timestamp>.mp4   (timestamps use "-" not ":")

A prompt file in extracted_timestamps/prompts/experiment<ID>/ is named:

    <participant_id>_s<session>_exp<i>.txt   (e.g. p6_s1_exp4.txt)

Neither filename alone tells you which video goes with which prompt: the link
is the CSV row in extracted_timestamps/labels_experiment<N>/session<S>_p<N>_timestamp_labels.csv,
whose (begin_timestamp, end_timestamp) gives the video filename and whose
(node_id, matched AI node id) gives the prompt filename -- exactly the same
matching logic prompt_maker_setting.py already uses to name its output files.
This script reuses that logic directly (imported from prompt_maker_setting.py)
so the pairing is guaranteed consistent with what's on disk.

NOTE on video input: GPT models have no video modality at all (confirmed by
the gateway rejecting a "video_url" content part: "Invalid value: 'video_url'.
Supported values are: 'text', 'image_url', 'input_audio', 'refusal', 'audio',
and 'file'"). So instead of sending the video file, this script samples
evenly-spaced frames out of the clip with ffmpeg and sends those as a
sequence of "image_url" content parts alongside the prompt text -- the
standard way to give a GPT vision model "video understanding".

By default the number of frames is computed per clip at roughly one frame
every FRAME_INTERVAL_SECONDS (10s), clamped to [MIN_FRAMES, MAX_FRAMES] =
[4, 32] -- e.g. an 80s clip gets 8 frames, a 6-minute clip gets ~37 -> capped
to 32, a 21-minute clip also caps to 32 rather than the ~128 a flat 1/10s
rate would otherwise produce. This avoids both under-sampling short clips and
sending huge, expensive frame counts for the long outliers. Pass --frames to
override with a fixed count for every clip instead of the automatic one.

This means GPT never sees motion between frames or the audio track, only a
handful of still frames; if a clip's important content depends on something
that happens between sampled frames, it may miss it.

IMPORTANT -- this calls a paid API. By default it only runs ONE (video, prompt)
pair so you can test before spending real API credits. Pass --limit N to run
more, or --all to run every pair found. Use --dry-run to see what would be
sent (including which frames, without calling the API at all).

Results (--output, default experiment_results/gpt_experiment<ID>_results.csv) are
appended as CSV rows, one at a time as each completes (so a crash or an API
error partway through never loses already-completed, already-paid-for
results): participant_id, experiment_id, real_answer, predicted_answer,
video_clip_length, step_jump.
  - video_clip_length: the clip's duration in seconds (end_timestamp minus
    begin_timestamp from the labels_experiment<N> CSV row).
  - step_jump: how many workflow steps apart the AI action and the user's
    response to it are (their "Step N" segment numbers), e.g. 0 if the AI
    action and the user's response both happen within the same step, 1 if
    the AI action was in step 1 and the user's response came in step 2.
Rows already present (by experiment_id) are skipped on a re-run, so an
interrupted --all run -- or one you extend later with new participants/videos
-- can just be re-invoked and it will only add rows for what's missing,
never re-spend API credits re-running experiment_ids already in the CSV.

Usage:
    python run_gpt_setting.py --setting 1a --dry-run                     # preview the 1 default sample
    python run_gpt_setting.py --setting 1a                                # actually call GPT once
    python run_gpt_setting.py --setting 1b --p p1 --session 1 --limit 1   # a specific, deterministic pick
    python run_gpt_setting.py --setting 1a --exp exp1_p1_s1               # run one specific experiment_id
    python run_gpt_setting.py --setting 1a --frames 20 --exp exp1_p1_s1   # fixed 20 frames instead of automatic
    python run_gpt_setting.py --setting 1a --all                          # run everything, save to the default CSV
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import base64
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from experiments import (
    CLASSIFICATION_EXPERIMENTS,
    CONDITION_CHOICES, condition_folder, condition_prompts_dir, condition_results_path, needs_video,
    video_too_large,
)
from prompts.prompt_maker_setting import (
    ROOT_DIR,
    build_node_index,
    discover_jobs,
    find_ai_action_for_row,
    load_graph,
    prompt_filename,
    to_seconds,
)

MODEL_SLUG = "gpt"
MODEL = "gpt-6-astra"
FRAME_INTERVAL_SECONDS = 10
MIN_FRAMES = 4
MAX_FRAMES = 32

VALID_LABELS = ("ADOPT", "MODIFY", "REJECT/BYPASS")


def truncated_videos_dir(setting):
    return ROOT_DIR / "extracted_timestamps" / "truncated_videos" / f"experiment{setting}"


def prompts_dir(setting, condition="video-only"):
    return condition_prompts_dir(setting, condition)


def default_output(setting, condition="video-only"):
    return condition_results_path(setting, condition, MODEL_SLUG)


def ffprobe_duration_seconds(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {Path(video_path).name}:\n{result.stderr[-1000:]}")
    return float(result.stdout.strip())


def auto_frame_count(duration_seconds):
    """~1 frame every FRAME_INTERVAL_SECONDS, clamped to [MIN_FRAMES, MAX_FRAMES]."""
    return max(MIN_FRAMES, min(MAX_FRAMES, round(duration_seconds / FRAME_INTERVAL_SECONDS)))


def get_client():
    # Lazy import so --dry-run works even if the `openai` package/gateway key isn't set up yet.
    from openai import OpenAI

    api_key = os.environ.get("AIGATEWAY_API_KEY", "")
    return OpenAI(api_key=api_key, base_url="https://api.aigateway.umn.edu")


def extract_frame_b64_list(video_path, num_frames):
    """
    Sample `num_frames` evenly-spaced frames from video_path and return them as
    a list of base64-encoded JPEG strings, in chronological order.
    """
    duration = ffprobe_duration_seconds(video_path)
    interval = max(duration / num_frames, 0.1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"fps=1/{interval:.6f}",
                "-qscale:v", "3",
                str(tmp / "frame_%03d.jpg"),
            ],
            capture_output=True, text=True,
        )
        frame_paths = sorted(tmp.glob("frame_*.jpg"))
        if result.returncode != 0 or not frame_paths:
            raise RuntimeError(f"ffmpeg frame extraction failed for {video_path.name}:\n{result.stderr[-1000:]}")

        frame_paths = frame_paths[:num_frames]
        return [base64.b64encode(p.read_bytes()).decode("ascii") for p in frame_paths]


def collect_samples(participant_filter, session_filter, setting, condition="video-only"):
    """
    Walk the same CSVs prompt_maker_setting.py uses, and for every row produce
    a sample dict pointing at the matching video file and prompt file. For the
    conditions that send no video (graph-only, chat-only, graph-chat) the video
    clip is not required and sample["video_path"] is None.
    """
    jobs = discover_jobs(participant_filter, session_filter, setting)
    samples = []
    videos_dir = truncated_videos_dir(setting)
    prompts_dir_path = prompts_dir(setting, condition)
    with_video = needs_video(condition)

    for csv_path, session, participant_id in jobs:
        try:
            graph = load_graph(session, participant_id)
        except FileNotFoundError:
            # no graph in new_annotated_graphs/ (step_jump needs it): leave this session's existing rows as they are
            print(f"Warning: no graph for {participant_id} session {session}; skipping it.", file=sys.stderr)
            continue
        node_index = build_node_index(graph)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row["user_action_node"]
                target_ai_timestamp = row["target_ai_timestamp"]
                try:
                    ai_node_id, _ai_step, _user_step, _step_name = find_ai_action_for_row(
                        node_index, node_id, target_ai_timestamp
                    )
                except (KeyError, ValueError) as exc:
                    print(f"Warning: skipping node {node_id!r} in {csv_path.name}: {exc}", file=sys.stderr)
                    continue

                begin_dash = row["begin_timestamp"].replace(":", "-")
                end_dash = row["end_timestamp"].replace(":", "-")
                video_path = videos_dir / f"session{session}_{participant_id}_{begin_dash}_{end_dash}.mp4"
                prompt_path = prompts_dir_path / prompt_filename(row["experiment_id"], participant_id, session)

                if with_video and not video_path.exists():
                    print(f"Warning: missing video {video_path.name}; skipping.", file=sys.stderr)
                    continue
                if with_video and video_too_large(setting, video_path):
                    print(f"Skipping {video_path.name}: over the experiment{setting} video size limit.", file=sys.stderr)
                    continue
                if not prompt_path.exists():
                    print(f"Warning: missing prompt {prompt_path.name}; skipping.", file=sys.stderr)
                    continue

                video_clip_length = to_seconds(row["end_timestamp"]) - to_seconds(row["begin_timestamp"])
                # step_jump: how many steps apart the AI action and the user's
                # response to it are (0 = same step/segment of the workflow).
                user_step_number = node_index[node_id][2]
                ai_step_number = node_index[ai_node_id][2]
                step_jump = user_step_number - ai_step_number

                samples.append(
                    {
                        "experiment_id": row.get("experiment_id", ""),
                        "participant_id": participant_id,
                        "session": session,
                        "node_id": node_id,
                        "ai_node_id": ai_node_id,
                        "video_path": video_path if with_video else None,
                        "prompt_path": prompt_path,
                        "ground_truth_label": row["label"],
                        "video_clip_length": video_clip_length,
                        "step_jump": step_jump,
                    }
                )

    return samples


def nonempty_answer(response):
    """The reply text; raises on an empty reply (e.g. a blocked or cut-off video request) so the
    caller skips the sample instead of writing a blank row that a rerun would never redo."""
    answer = response.choices[0].message.content
    if not (answer or "").strip():
        finish = getattr(response.choices[0], "finish_reason", None)
        raise RuntimeError(f"empty answer from the API (finish_reason={finish!r})")
    return answer


def call_gpt(client, prompt_text, video_path, num_frames):
    content = [{"type": "text", "text": prompt_text}]
    if video_path is not None:  # None for the graph-only / chat-only / graph-chat conditions: text only
        for frame_b64 in extract_frame_b64_list(video_path, num_frames):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    return nonempty_answer(response), response


def run_batch(setting, condition, selected, output, args, client):
    """Send `selected` samples of one (setting, condition) to GPT, appending to `output`
    (rows already in it are skipped)."""
    already_done = set()
    if output and Path(output).exists():
        with open(output, newline="") as f:
            for row in csv.DictReader(f):
                already_done.add(row["experiment_id"])

    fieldnames = ["participant_id", "experiment_id", "real_answer", "predicted_answer", "video_clip_length", "step_jump"]
    output_file = None
    writer = None
    if output and not args.dry_run:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.exists() or output_path.stat().st_size == 0
        output_file = open(output, "a", newline="")
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

    succeeded = 0
    failed = 0
    skipped = 0
    try:
        for sample in tqdm(selected, desc=f"Running GPT {condition_folder(setting, condition)} experiments", unit="clip"):
            if sample["experiment_id"] in already_done:
                tqdm.write(f"Skipping (already in {output}): {sample['prompt_path'].name}")
                skipped += 1
                continue

            prompt_text = sample["prompt_path"].read_text()
            video_path = sample["video_path"]  # None for the graph-only / chat-only / graph-chat conditions
            size_bytes = video_path.stat().st_size if video_path is not None else 0
            num_frames = None  # only used when there is a video to sample
            if video_path is not None:
                num_frames = args.frames if args.frames is not None else auto_frame_count(sample["video_clip_length"])

            tqdm.write("=" * 70)
            tqdm.write(f"Experiment: {sample['experiment_id']}  Condition: {condition}")
            tqdm.write(f"Participant: {sample['participant_id']}  Session: {sample['session']}")
            if video_path is not None:
                tqdm.write(
                    f"Video:  {video_path.name}  ({size_bytes / 1_000_000:.1f} MB, "
                    f"{sample['video_clip_length']}s)  -> sampling {num_frames} frames"
                )
            else:
                tqdm.write("Video:  (none -- text-only condition, prompt text only)")
            tqdm.write(f"Prompt: {sample['prompt_path'].name}")
            tqdm.write(f"Ground-truth label (not sent to the model): {sample['ground_truth_label']}")

            if args.dry_run:
                continue

            try:
                answer, response = call_gpt(client, prompt_text, sample["video_path"], num_frames)
            except Exception as exc:
                tqdm.write(f"ERROR calling GPT for {sample['experiment_id']}: {exc}")
                tqdm.write("Skipping this sample and continuing with the rest.")
                failed += 1
                continue

            tqdm.write(f"GPT answer: {answer}")
            usage = getattr(response, "usage", None)
            if usage is not None:
                details = getattr(usage, "completion_tokens_details", None)
                reasoning = getattr(details, "reasoning_tokens", None)
                tqdm.write(
                    f"Usage: prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens} "
                    f"reasoning_tokens={reasoning} frames={num_frames}"
                )

            row = {
                "participant_id": sample["participant_id"],
                "experiment_id": sample["experiment_id"],
                "real_answer": sample["ground_truth_label"],
                "predicted_answer": answer,
                "video_clip_length": sample["video_clip_length"],
                "step_jump": sample["step_jump"],
            }
            if writer:
                writer.writerow(row)
                output_file.flush()
            succeeded += 1
    finally:
        if output_file:
            output_file.close()

    if not args.dry_run:
        print(
            f"[{condition_folder(setting, condition)}] Wrote {succeeded} result(s) to {output}; "
            f"{failed} sample(s) failed; {skipped} already-done sample(s) skipped."
        )



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--setting", required=True, nargs="+", choices=list(CLASSIFICATION_EXPERIMENTS),
        help="Which setting(s) to run, e.g. --setting 1a or --setting 1a 1b",
    )
    parser.add_argument(
        "--condition", nargs="+", default=["video-only"], choices=CONDITION_CHOICES,
        help=(
            "Input mode(s), e.g. --condition video-only graph-only chat-only (default: video-only, the plain "
            "experiment<ID> run). Prompts come from prompts/experiment<ID>[_<mode>]/, results go to "
            "experiment_results/experiment<ID>[_<mode>]/. graph-only / chat-only / graph-chat send no video."
        ),
    )
    parser.add_argument("--p", required=False, default=None, help="Participant id, e.g. p1")
    parser.add_argument("--session", required=False, default=None, choices=["1", "2"], help="Session number: 1 or 2")
    parser.add_argument(
        "--exp", required=False, default=None,
        help="Run only this one experiment_id (e.g. exp1_p1_s1), from the labels_experiment<N> CSV column",
    )
    parser.add_argument(
        "--limit", type=int, default=1, help="Max number of samples to actually send to GPT (default: 1)"
    )
    parser.add_argument("--all", action="store_true", help="Ignore --limit and run every matching sample")
    parser.add_argument(
        "--frames", type=int, default=None,
        help=(
            "Fixed number of evenly-spaced frames to sample per clip. If omitted, it's computed "
            f"automatically per clip (~1 frame every {FRAME_INTERVAL_SECONDS}s, clamped to "
            f"[{MIN_FRAMES}, {MAX_FRAMES}])."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be sent (paths, sizes) without calling the API"
    )
    parser.add_argument(
        "--output", required=False, default=None,
        help="CSV file to append results to (default: experiment_results/experiment<ID>/gpt_experiment<ID>_results.csv, or see --run)",
    )
    parser.add_argument(
        "--run", type=int, nargs="+", default=None,
        help=(
            "Run number for this batch, e.g. --run 1 writes to "
            "experiment_results/experiment<ID>/gpt_experiment<ID>_results_run1.csv. "
            "Ignored if --output is also given."
        ),
    )
    args = parser.parse_args()

    settings = list(dict.fromkeys(args.setting))
    conditions = list(dict.fromkeys(args.condition))
    runs = list(dict.fromkeys(args.run)) if args.run else [None]
    if args.output and (len(settings) > 1 or len(conditions) > 1 or len(runs) > 1):
        parser.error("--output can only be used with a single --setting, --condition and --run "
                     "(each needs its own results file)")

    participant_filter = args.p.strip().lower() if args.p else None
    session_filter = int(args.session) if args.session else None

    plan = []
    for setting in settings:
        for condition in conditions:
            label = condition_folder(setting, condition)
            samples = collect_samples(participant_filter, session_filter, setting, condition)
            if not samples:
                print(f"No matching (video, prompt) pairs found for {label}.", file=sys.stderr)
                continue
            if args.exp:
                selected = [s for s in samples if s["experiment_id"] == args.exp]
                if not selected:
                    print(f"No sample found with experiment_id={args.exp!r} in {label}.", file=sys.stderr)
                    continue
            else:
                selected = samples if args.all else samples[: args.limit]
            print(f"[{label}] {len(samples)} sample(s) matched; running {len(selected)} of them.")
            plan.append((setting, condition, selected))
    if not plan:
        return

    client = None if args.dry_run else get_client()
    for run in runs:
        if run is not None:
            print(f"=== run {run} ===")
        for setting, condition, selected in plan:
            output = args.output or str(condition_results_path(setting, condition, MODEL_SLUG, run))
            run_batch(setting, condition, selected, output, args, client)


if __name__ == "__main__":
    main()
