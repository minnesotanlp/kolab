"""
Run (video, prompt) pairs through a chosen model and print/save its predicted
output-quality rating (1-5), for Experiment 2b (output-quality-rating
prediction). --model selects the backend:

  gemini  - gemini-3.1-pro-preview, via the UMN AI gateway (paid API).
            Sends the raw video inline (base64 "video_url" content part).
  flash   - gemini-3.8-flash, via the UMN AI gateway (paid API). Same
            video-input path as gemini, just a cheaper/faster model.
  gpt     - gpt-6-astra, via the UMN AI gateway (paid API). GPT has no video
            modality at all, so this samples evenly-spaced frames from the
            clip with ffmpeg instead and sends those as a sequence of
            "image_url" parts -- see extract_frame_b64_list(). Pass --frames
            to use a fixed count instead of the automatic ~1-per-10s one.
  qwen    - Qwen/Qwen3-VL-30B-A3B-Instruct, run locally (GPU, via
            transformers + qwen_vl_utils) -- no API cost, but needs the
            model weights (downloaded once, cached locally) and a GPU.
  qwen32b - Qwen/Qwen3-VL-32B-Instruct, run locally the same way as qwen
            (same code path, different model weights).

Matching video <-> prompt
-------------------------
extracted_timestamps/experiment2b/basic_experiment_setup.csv has one row per
workflow step, so no per-participant graph lookup is needed to pair a video
with its prompt:

  - video: extracted_timestamps/truncated_videos/experiment2a/<experiment_id>.mp4
    -- NOTE: this reuses Experiment 2a's truncated-videos folder, NOT a
    separate experiment2b one. The underlying video clips for a given
    experiment_id/step are byte-identical between 2a and 2b (same source
    video, same timestamp ranges), so 2b just reuses whatever 2a already
    truncated rather than needing its own truncation step. If a needed clip
    is genuinely missing there, run
    `truncate_videos_for_setting.py --setting experiment2a` (there is no
    2b-specific truncation command, since none exists/is needed) to produce
    it.
  - prompt: extracted_timestamps/prompts/experiment2b/<experiment_id>.txt
    (written by prompt_maker_setting.py --setting experiment2b)
  - ground truth: the row's output_quality_rating column (1-5)

IMPORTANT -- gemini/flash/gpt call a paid API; qwen loads a large local model
(slow the first time). By default this only runs ONE (video, prompt) pair so
you can test before spending real API credits / GPU time. Pass --limit N to
run more, or --all to run every pair found. Use --dry-run to see what would
be sent (including video/frame sizes) without calling the API or loading the
model at all.

Results are appended as CSV rows to --output if given, else to
experiment_results/experiment2b/<model>_experiment2b_results_run<R>.csv if
--run R is given, else to
experiment_results/experiment2b/<model>_experiment2b_results.csv. Columns:
participant_id, experiment_id, real_answer, predicted_answer,
video_clip_length, num_timestamp_ranges.
  - video_clip_length: total seconds actually shown to the model, summed
    across every range in the row's timestamp_ranges (more than one range
    when the step's footage is disjoint -- the participant revisited it
    later in the session).
  - num_timestamp_ranges: how many disjoint ranges make up that clip (1 for
    a normal contiguous step, 2+ for a revisit).
Rows already present (by experiment_id) are skipped on a re-run, so an
interrupted --all run -- or one you extend later with new rows -- can just
be re-invoked and it will only add rows for what's missing, never re-spend
API credits/GPU time re-running experiment_ids already in the CSV.

Usage:
    python run_experiment2b.py --model gemini --dry-run                     # preview the 1 default sample
    python run_experiment2b.py --model flash                                 # actually call gemini-3.8-flash once
    python run_experiment2b.py --model gpt --p p2 --session 1 --limit 1      # a specific, deterministic pick
    python run_experiment2b.py --model qwen --exp p2_session1_step1_basic_setup  # run one specific experiment_id
    python run_experiment2b.py --model gemini --all --run 1                  # run everything, save to results_run1.csv
    python run_experiment2b.py --model flash --condition graph-only-ts --all --run 1 2 3   # graph only (no video), with node timestamps
    python run_experiment2b.py --model qwen32b --condition video-graph video-graph-ts --all --run 1 2 3
        # several conditions x several runs in ONE process: a qwen model is loaded once

Conditions (--condition), same as Experiment 2a: video-only (default), video-graph, video-graph-ts,
graph-only, graph-only-ts, plus the chat ablation video-graph-chat, video-chat, graph-chat, chat-only
(prompts built by run_chat_pipeline.sh). The graph ones cover the annotated steps and need
`build_2a_graph_setup.py --experiment 2b` and `prompt_maker_setting.py --setting experiment2b_<...>` first;
results go to experiment_results/experiment2b_<video_graph|video_graph_ts|graph_only|graph_only_ts>/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import base64
import csv
import os

# Must be set before torch initializes its CUDA allocator (first `import torch`,
# even a lazy/transitive one via transformers below) to take effect. Reduces
# the "OOM despite enough total free memory" failure mode from allocator
# fragmentation over a long --all run of variable-length video clips.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from config import ROOT_DIR
EXPERIMENT2B_CSV = ROOT_DIR / "extracted_timestamps" / "experiment2b" / "basic_experiment_setup.csv"
# Deliberately points at 2a's truncated-videos folder, not a new experiment2b
# one -- see module docstring. If a clip is missing here, run
# truncate_videos_for_setting.py --setting experiment2a to produce it.
TRUNCATED_VIDEOS_DIR = ROOT_DIR / "extracted_timestamps" / "truncated_videos" / "experiment2a"
PROMPTS_DIR = ROOT_DIR / "extracted_timestamps" / "prompts" / "experiment2b"

# Conditions built from annotated_graphs/ by build_2a_graph_setup.py --experiment 2b and
# prompt_maker_setting.py's experiment2b_* GRAPH_CONDITIONS: the same steps as video-only,
# each with or without the step's graph (and, with "-ts", the graph's node timestamps on
# the step clip's clock), and with or without the video. Same conditions as Experiment 2a.
ANNOTATED_SETUP_CSV = ROOT_DIR / "extracted_timestamps" / "experiment2b" / "annotated_graph_experiment_setup.csv"


def _prompts_dir(folder):
    return ROOT_DIR / "extracted_timestamps" / "prompts" / folder


# condition -> setup CSV, prompts dir, whether the video is attached, and the folder its
# results go to (experiment_results/<folder>/<model>_<folder>_results*.csv).
CONDITIONS = {
    "video-only":      {"csv": EXPERIMENT2B_CSV,    "prompts": PROMPTS_DIR,                                 "video": True,  "folder": "experiment2b"},
    "video-graph":     {"csv": ANNOTATED_SETUP_CSV, "prompts": _prompts_dir("experiment2b_video_graph"),    "video": True,  "folder": "experiment2b_video_graph"},
    "video-graph-ts":  {"csv": ANNOTATED_SETUP_CSV, "prompts": _prompts_dir("experiment2b_video_graph_ts"), "video": True,  "folder": "experiment2b_video_graph_ts"},
    "graph-only":      {"csv": ANNOTATED_SETUP_CSV, "prompts": _prompts_dir("experiment2b_graph_only"),     "video": False, "folder": "experiment2b_graph_only"},
    "graph-only-ts":   {"csv": ANNOTATED_SETUP_CSV, "prompts": _prompts_dir("experiment2b_graph_only_ts"),  "video": False, "folder": "experiment2b_graph_only_ts"},
    # Chat ablation (run_chat_pipeline.sh), same conditions as Experiment 2a.
    "video-graph-chat": {"csv": ANNOTATED_SETUP_CSV, "prompts": _prompts_dir("experiment2b_video_graph_chat"), "video": True,  "folder": "experiment2b_video_graph_chat"},
    "video-chat":      {"csv": ANNOTATED_SETUP_CSV, "prompts": _prompts_dir("experiment2b_video_chat"),       "video": True,  "folder": "experiment2b_video_chat"},
    "graph-chat":      {"csv": ANNOTATED_SETUP_CSV, "prompts": _prompts_dir("experiment2b_graph_chat"),       "video": False, "folder": "experiment2b_graph_chat"},
    "chat-only":       {"csv": ANNOTATED_SETUP_CSV, "prompts": _prompts_dir("experiment2b_chat_only"),        "video": False, "folder": "experiment2b_chat_only"},
}
CONDITION_CHOICES = tuple(CONDITIONS)

MODEL_CHOICES = ("qwen", "qwen32b", "gemini", "flash", "gpt")

API_MODEL_NAME = {
    "gemini": "gemini-3.1-pro-preview",
    "flash": "gemini-3.8-flash",
    "gpt": "gpt-6-astra",
}
QWEN_MODEL_NAMES = {
    "qwen": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen32b": "Qwen/Qwen3-VL-32B-Instruct",
}
MAX_NEW_TOKENS = 128

# Gemini/flash's inline (base64) request payload is capped around 20MB; anything
# bigger needs a real upload/Files API, which this simple inline approach does
# not do. Warn loudly instead of silently sending a request likely to fail.
SIZE_WARNING_BYTES = 20 * 1024 * 1024

FRAME_INTERVAL_SECONDS = 10
MIN_FRAMES = 4
MAX_FRAMES = 32

# qwen/qwen32b only: qwen_vl_utils samples at `fps` up to its own default
# max_frames=768 -- fine for short clips, but at fps=1.0 a long step (e.g. a
# 28-minute clip) hits that cap and floods the forward pass with hundreds of
# video-token frames, which is what actually OOMs generate() (max_memory only
# bounds where WEIGHTS get placed at load time, not this per-call
# activation/KV-cache cost). Capping max_frames bounds token count regardless
# of clip length.
DEFAULT_MAX_VIDEO_FRAMES = 32

VALID_LABELS = ("1", "2", "3", "4", "5")


def default_output(model, condition="video-only"):
    folder = CONDITIONS[condition]["folder"]
    return ROOT_DIR / "experiment_results" / folder / f"{model}_{folder}_results.csv"


def get_client():
    # Lazy import so --dry-run works even if the `openai` package/gateway key isn't set up yet.
    from openai import OpenAI

    api_key = os.environ.get("AIGATEWAY_API_KEY", "")
    return OpenAI(api_key=api_key, base_url="https://api.aigateway.umn.edu")


def print_gpu_info():
    """Report what torch/CUDA actually sees before loading a multi-GiB model,
    so a lopsided OOM (e.g. one GPU packed near its limit while a second sits
    idle) is diagnosable from the run's own output instead of guessing --
    device_map="auto" only shards across GPUs it can see, and a job's
    CUDA_VISIBLE_DEVICES can silently expose fewer than nvidia-smi shows."""
    import torch

    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False -- no GPU visible to this process.", file=sys.stderr)
        return
    n = torch.cuda.device_count()
    print(f"torch sees {n} GPU(s):", file=sys.stderr)
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        print(
            f"  cuda:{i} {props.name}  total={total_bytes / 1024**3:.1f}GiB  free={free_bytes / 1024**3:.1f}GiB",
            file=sys.stderr,
        )


def get_qwen_model_and_processor(model_name):
    # Lazy import so --dry-run works even if torch/transformers aren't installed yet.
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Cap each GPU's model-weight budget a few GiB below its physical capacity
    # (device_map="auto" otherwise packs weights right up to the edge), so
    # there's still room left for a long video clip's activation/KV-cache
    # memory without OOMing -- see the CUDACachingAllocator OOM this was added
    # for: a dense 32B model in bf16 needs ~64GB, so it's already sharded
    # across multiple GPUs, and the default packing left ~1.4GB free on one
    # card, not enough for some clips' visual-token activations.
    max_memory = None
    if torch.cuda.is_available():
        headroom_gib = 6
        max_memory = {
            i: f"{max(int(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)) - headroom_gib, 1)}GiB"
            for i in range(torch.cuda.device_count())
        }

    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", max_memory=max_memory
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def timestamp_to_seconds(ts):
    parts = [int(p) for p in ts.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def total_clip_length(timestamp_ranges):
    """Sum of durations (seconds) across every semicolon-separated
    "begin-end" range in a timestamp_ranges cell."""
    total = 0
    for part in timestamp_ranges.split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = re.split(r"[-–]", part, maxsplit=1)
        if len(pieces) != 2 or not pieces[0].strip() or not pieces[1].strip():
            continue
        total += timestamp_to_seconds(pieces[1]) - timestamp_to_seconds(pieces[0])
    return total


def ffprobe_duration_seconds(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {video_path.name}:\n{result.stderr[-1000:]}")
    return float(result.stdout.strip())


def auto_frame_count(duration_seconds):
    """~1 frame every FRAME_INTERVAL_SECONDS, clamped to [MIN_FRAMES, MAX_FRAMES]."""
    return max(MIN_FRAMES, min(MAX_FRAMES, round(duration_seconds / FRAME_INTERVAL_SECONDS)))


def extract_frame_b64_list(video_path, num_frames):
    """Sample `num_frames` evenly-spaced frames from video_path and return them
    as a list of base64-encoded JPEG strings, in chronological order."""
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


def collect_samples(participant_filter, session_filter, condition="video-only"):
    """Read the condition's setup CSV and, for every row, produce a sample
    dict pointing at the matching truncated video and prompt file. The condition
    (see CONDITIONS) picks the setup CSV and prompts dir; for the graph-only
    conditions no video is sent, so none is looked up (sample["video_path"] is None).

    experiment2b's own CSV (as built by build_experiment2b.py) has no
    truncated_video_filename column at all, so the fallback lookup directly
    in TRUNCATED_VIDEOS_DIR (2a's folder) by experiment_id is the path that
    actually fires in practice; the truncated_video_filename check is kept
    only for forward-compatibility/parity with run_experiment2a.py's shape.
    """
    cfg = CONDITIONS[condition]
    csv_path = cfg["csv"]
    prompts_dir = cfg["prompts"]
    needs_video = cfg["video"]

    if not csv_path.is_file():
        print(f"No such file: {csv_path}", file=sys.stderr)
        return []

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    samples = []
    for row in rows:
        participant_id = row.get("participant_id", "").strip().lower()
        session_id = row.get("session_id", "").strip().lower()
        experiment_id = row.get("experiment_id", "").strip()

        if participant_filter and participant_id != participant_filter:
            continue
        if session_filter and session_id != f"session{session_filter}":
            continue

        # Prefer a truncated_video_filename column when the CSV has it and
        # it's non-empty; otherwise (the normal case for experiment2b's own
        # CSV, which doesn't have this column), fall back to the same-named
        # clip in TRUNCATED_VIDEOS_DIR (2a's truncated-videos folder, reused
        # as-is -- see module docstring).
        truncated_video_filename = row.get("truncated_video_filename", "").strip()
        video_path = None
        if truncated_video_filename and needs_video:
            candidate = TRUNCATED_VIDEOS_DIR / truncated_video_filename
            if candidate.exists():
                video_path = candidate

        if video_path is None and needs_video:
            fallback_candidate = TRUNCATED_VIDEOS_DIR / f"{experiment_id}.mp4"
            if fallback_candidate.exists():
                video_path = fallback_candidate

        if video_path is None and needs_video:
            if not truncated_video_filename:
                print(
                    f"Warning: no truncated video found for {experiment_id!r} in {TRUNCATED_VIDEOS_DIR} "
                    f"(run truncate_videos_for_setting.py --setting experiment2a first); skipping.",
                    file=sys.stderr,
                )
            else:
                print(f"Warning: missing video {truncated_video_filename}; skipping.", file=sys.stderr)
            continue

        prompt_path = prompts_dir / f"{experiment_id}.txt"

        if not prompt_path.exists():
            print(f"Warning: missing prompt {prompt_path.name}; skipping.", file=sys.stderr)
            continue

        timestamp_ranges = row.get("timestamp_ranges", "")
        num_ranges = len([p for p in timestamp_ranges.split(";") if p.strip()])

        samples.append(
            {
                "experiment_id": experiment_id,
                "participant_id": participant_id,
                "session_id": session_id,
                "video_path": video_path,
                "prompt_path": prompt_path,
                "ground_truth_label": row.get("output_quality_rating", "").strip(),
                "video_clip_length": total_clip_length(timestamp_ranges),
                "num_timestamp_ranges": num_ranges,
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


def call_gemini_like(client, model_name, prompt_text, video_path):
    """Used for both gemini and flash -- same gateway, same inline-video shape.
    video_path=None (the graph-only conditions) sends the prompt text alone."""
    content = [{"type": "text", "text": prompt_text}]
    if video_path is not None:
        video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
        content.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})

    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(model=model_name, messages=messages)
    return nonempty_answer(response)


def call_gpt(client, prompt_text, video_path, num_frames):
    """video_path=None (the graph-only conditions) sends the prompt text alone."""
    content = [{"type": "text", "text": prompt_text}]
    if video_path is not None:
        for frame_b64 in extract_frame_b64_list(video_path, num_frames):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(model=API_MODEL_NAME["gpt"], messages=messages)
    return nonempty_answer(response)


def call_qwen(model, processor, prompt_text, video_path, max_frames=DEFAULT_MAX_VIDEO_FRAMES):
    """video_path=None (the graph-only conditions) sends the prompt text alone."""
    from qwen_vl_utils import process_vision_info

    content = [{"type": "text", "text": prompt_text}]
    if video_path is not None:
        video_uri = f"file://{os.path.abspath(video_path)}"
        content.insert(0, {"type": "video", "video": video_uri, "fps": 1.0, "max_frames": max_frames})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def resolve_output(args, run):
    if args.output:
        return args.output
    if run is not None:
        folder = CONDITIONS[args.condition]["folder"]
        return str(ROOT_DIR / "experiment_results" / folder / f"{args.model}_{folder}_results_run{run}.csv")
    return str(default_output(args.model, args.condition))


def run_batch(args, selected, client, qwen_model, qwen_processor):
    already_done = set()
    if args.output and Path(args.output).exists():
        with open(args.output, newline="") as f:
            for row in csv.DictReader(f):
                already_done.add(row["experiment_id"])

    fieldnames = [
        "participant_id", "experiment_id", "real_answer", "predicted_answer",
        "video_clip_length", "num_timestamp_ranges",
    ]
    output_file = None
    writer = None
    if args.output and not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.exists() or output_path.stat().st_size == 0
        output_file = open(args.output, "a", newline="")
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

    succeeded = 0
    failed = 0
    skipped = 0
    try:
        for sample in tqdm(selected, desc=f"Running {args.model} experiment2b experiments", unit="clip"):
            if sample["experiment_id"] in already_done:
                tqdm.write(f"Skipping (already in {args.output}): {sample['prompt_path'].name}")
                skipped += 1
                continue

            prompt_text = sample["prompt_path"].read_text()
            video_path = sample["video_path"]  # None for the graph-only conditions: prompt text only
            size_bytes = video_path.stat().st_size if video_path is not None else 0

            tqdm.write("=" * 70)
            tqdm.write(f"Experiment: {sample['experiment_id']}")
            tqdm.write(f"Participant: {sample['participant_id']}  Session: {sample['session_id']}")
            if video_path is not None:
                tqdm.write(f"Video:  {video_path.name}  ({size_bytes / 1_000_000:.1f} MB)")
            else:
                tqdm.write("Video:  (none -- graph-only condition, prompt text only)")
            tqdm.write(f"Prompt: {sample['prompt_path'].name}")
            tqdm.write(f"Ground-truth rating (not sent to the model): {sample['ground_truth_label']}")

            if args.model in ("gemini", "flash") and size_bytes > SIZE_WARNING_BYTES:
                tqdm.write(
                    f"WARNING: video is {size_bytes / 1_000_000:.1f} MB, over the ~20MB inline "
                    f"request size this gateway typically allows without a Files-API upload. "
                    f"This request may fail."
                )
            num_frames = None  # only used by gpt, and only when there is a video to sample
            if args.model == "gpt" and video_path is not None:
                num_frames = args.frames if args.frames is not None else auto_frame_count(sample["video_clip_length"])
                tqdm.write(f"Clip length: {sample['video_clip_length']}s -> sampling {num_frames} frame(s)")

            if args.dry_run:
                continue

            try:
                if args.model in ("gemini", "flash"):
                    answer = call_gemini_like(client, API_MODEL_NAME[args.model], prompt_text, video_path)
                elif args.model == "gpt":
                    answer = call_gpt(client, prompt_text, video_path, num_frames)
                else:
                    answer = call_qwen(qwen_model, qwen_processor, prompt_text, video_path, args.max_frames)
            except Exception as exc:
                tqdm.write(f"ERROR running {args.model} for {sample['experiment_id']}: {exc}")
                tqdm.write("Skipping this sample and continuing with the rest.")
                failed += 1
                continue
            finally:
                # Release cached-but-unallocated CUDA memory after every clip (not
                # just on failure) so fragmentation doesn't accumulate over a long
                # --all run of variable-length videos and OOM a later clip.
                if args.model in ("qwen", "qwen32b"):
                    import torch
                    torch.cuda.empty_cache()

            tqdm.write(f"Answer [{args.model}]: {answer}")

            row = {
                "participant_id": sample["participant_id"],
                "experiment_id": sample["experiment_id"],
                "real_answer": sample["ground_truth_label"],
                "predicted_answer": answer,
                "video_clip_length": sample["video_clip_length"],
                "num_timestamp_ranges": sample["num_timestamp_ranges"],
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
            f"Wrote {succeeded} result(s) to {args.output}; "
            f"{failed} sample(s) failed; {skipped} already-done sample(s) skipped."
        )

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES, help="Which model to run")
    parser.add_argument("--p", required=False, default=None, help="Participant id, e.g. p2")
    parser.add_argument(
        "--session", required=False, default=None, choices=["1", "2"], help="Session number: 1 or 2"
    )
    parser.add_argument(
        "--exp", required=False, default=None,
        help="Run only this one experiment_id (e.g. p2_session1_step1_basic_setup)",
    )
    parser.add_argument(
        "--limit", type=int, default=1, help="Max number of samples to actually run (default: 1)"
    )
    parser.add_argument("--all", action="store_true", help="Ignore --limit and run every matching sample")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be sent (paths, sizes) without calling the API or loading the model",
    )
    parser.add_argument(
        "--condition", nargs="+", default=["video-only"], choices=CONDITION_CHOICES,
        help=(
            "One or more of the conditions below (e.g. --condition video-graph video-graph-ts): they run one "
            "after another in ONE process, so a qwen model is loaded only once. "
            "video-only (default): the step clip + the plain prompt. video-graph / video-graph-ts: the "
            "clip + a prompt that also contains the step's graph, without / with each node's timestamp "
            "range. graph-only / graph-only-ts: the graph prompt alone, NO video sent. The graph "
            "conditions read extracted_timestamps/experiment2b/annotated_graph_experiment_setup.csv "
            "(build_2a_graph_setup.py --experiment 2b) and prompts/experiment2b_<video_graph|graph_only>[_ts]/ "
            "(prompt_maker_setting.py); results go to experiment_results/experiment2b_<...>/"
        ),
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="--model gpt only: fixed number of evenly-spaced frames to sample per clip (default: automatic, "
             f"~1 every {FRAME_INTERVAL_SECONDS}s, clamped to [{MIN_FRAMES}, {MAX_FRAMES}])",
    )
    parser.add_argument(
        "--max_frames", type=int, default=DEFAULT_MAX_VIDEO_FRAMES,
        help=(
            f"--model qwen/qwen32b only: cap on sampled video frames per clip (default: "
            f"{DEFAULT_MAX_VIDEO_FRAMES}). Long clips at fps=1.0 can otherwise reach qwen_vl_utils' own "
            "default cap of 768 frames, which is what actually drives OOM during generate() (not "
            "model-weight placement) -- lower this further if OOMs persist, especially with --model qwen32b."
        ),
    )
    parser.add_argument(
        "--output", required=False, default=None,
        help="CSV file to append results to (default: experiment_results/experiment2b/<model>_experiment2b_results.csv, or see --run)",
    )
    parser.add_argument(
        "--run", type=int, nargs="+", default=None,
        help=(
            "Run number(s), e.g. --run 1 writes to "
            "experiment_results/experiment2b/<model>_experiment2b_results_run1.csv. "
            "Pass several (--run 1 2 3) to load the model ONCE and do every run in this process. "
            "Ignored if --output is also given."
        ),
    )
    args = parser.parse_args()

    conditions = list(dict.fromkeys(args.condition))  # de-duplicate, keep order
    runs = list(dict.fromkeys(args.run)) if args.run else [None]
    if args.output and (len(conditions) > 1 or len(runs) > 1):
        parser.error("--output can only be used with a single --condition and a single --run "
                     "(each needs its own results file)")

    print(f"Using model: {args.model}   condition(s): {', '.join(conditions)}")

    participant_filter = args.p.strip().lower() if args.p else None
    session_filter = int(args.session) if args.session else None

    # Collect and select samples for every requested condition first, so the (slow) model
    # load happens once and only if there is something to run.
    plan = []
    for condition in conditions:
        samples = collect_samples(participant_filter, session_filter, condition)
        if not samples:
            print(f"No matching samples found for experiment2b condition {condition!r}.", file=sys.stderr)
            continue

        if args.exp:
            selected = [s for s in samples if s["experiment_id"] == args.exp]
            if not selected:
                print(f"No sample found with experiment_id={args.exp!r} in condition {condition!r}.", file=sys.stderr)
                continue
        else:
            selected = samples if args.all else samples[: args.limit]
        print(f"[{condition}] {len(samples)} sample(s) matched; running {len(selected)} of them.")
        plan.append((condition, selected))

    if not plan:
        return

    client = None
    qwen_model = qwen_processor = None
    if not args.dry_run:
        if args.model in ("qwen", "qwen32b"):
            print_gpu_info()
            qwen_model, qwen_processor = get_qwen_model_and_processor(QWEN_MODEL_NAMES[args.model])
        else:
            client = get_client()

    user_output = args.output
    for condition, selected in plan:
        args.condition = condition  # resolve_output() / run_batch() read the current condition from args
        for run in runs:
            args.output = user_output
            args.output = resolve_output(args, run)
            if len(runs) > 1 or len(plan) > 1:
                prefix = f"{condition} | " if len(plan) > 1 else ""
                print(f"\n===== {prefix}run {run} -> {args.output} =====")
            run_batch(args, selected, client, qwen_model, qwen_processor)




if __name__ == "__main__":
    main()
