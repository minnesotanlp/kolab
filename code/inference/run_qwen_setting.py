"""
Run (video, prompt) pairs through a locally-hosted video-LLM and print/save
the model's predicted label, for experiments 1a-1d (--setting).

Unlike run_gemini_setting.py / run_kimi_setting.py / run_gpt_setting.py, this
does not call a remote API -- the model is loaded once from Hugging Face
(weights cached locally after the first run) and run on this machine's
GPU(s), reading the video file directly from disk.

By default this uses Qwen3-VL-30B-A3B-Instruct (MoE). Pass --32b to use the
dense Qwen3-VL-32B-Instruct instead. Pass --use_video_chat3 to use
MCG-NJU/VideoChat3-4B instead -- a much smaller (4B), fully-open model that
reportedly beats Qwen3-VL-4B on most video benchmarks via a different
architecture (inflated 3D ViT + aggressive temporal token compression).
NOTE: VideoChat3 is a newer, custom-architecture model (loaded with
trust_remote_code=True); its exact chat-message format is not as
battle-tested in transformers as Qwen's, so call_video_chat3() below is the
place to adjust if its actual API differs from what's implemented here --
this hasn't been run end-to-end, only verified against public documentation.
Results from the two models are written to separate CSVs (qwen_ vs
videochat3_ prefixed) so they're never mixed together.

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

By default this only runs ONE (video, prompt) pair so you can confirm the
pipeline works before running everything (model loading + first inference is
slow). Pass --limit N to run more, or --all to run every pair found. Use
--dry-run to see what would be sent (including video file size) without
loading the model at all.

Results (--output, default experiment_results/qwen_experiment<ID>_results.csv)
are appended as CSV rows, one at a time as each completes: participant_id,
experiment_id, real_answer, predicted_answer, video_clip_length, step_jump.
  - video_clip_length: the clip's duration in seconds (end_timestamp minus
    begin_timestamp from the labels_experiment<N> CSV row).
  - step_jump: how many workflow steps apart the AI action and the user's
    response to it are (their "Step N" segment numbers), e.g. 0 if the AI
    action and the user's response both happen within the same step, 1 if
    the AI action was in step 1 and the user's response came in step 2.
Rows already present (by experiment_id) are skipped on a re-run, so an
interrupted --all run -- or one you extend later with new participants/videos
-- can just be re-invoked and it will only add rows for what's missing.

Usage:
    python run_qwen_setting.py --setting 1a --dry-run                     # preview the 1 default sample
    python run_qwen_setting.py --setting 1a                                # actually run Qwen3-VL once
    python run_qwen_setting.py --setting 1b --p p1 --session 1 --limit 1   # a specific, deterministic pick
    python run_qwen_setting.py --setting 1a 1b --all --run 1 2 3            # experiments 1a AND 1b, runs 1-3, model loaded once
    python run_qwen_setting.py --setting 1a 1b --condition video-only graph-only chat-only --all   # + input modes, model loaded once
    python run_qwen_setting.py --setting 1a --exp exp1_p1_s1               # run one specific experiment_id
    python run_qwen_setting.py --setting 1a --all                          # run everything, save to the default CSV
    python run_qwen_setting.py --setting 1a --use_video_chat3               # same, but with VideoChat3-4B
    python run_qwen_setting.py --setting 1a --32b                          # same, but with Qwen3-VL-32B-Instruct (dense)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import csv
import os

# Must be set before torch initializes its CUDA allocator (first `import torch`,
# even a lazy/transitive one via transformers below) to take effect. Reduces
# the "OOM despite enough total free memory" failure mode from allocator
# fragmentation over a long --all run of variable-length video clips.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
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

QWEN_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # MoE model; device_map="auto" shards it across available GPUs
QWEN_32B_MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"  # dense model
VIDEOCHAT3_MODEL_NAME = "MCG-NJU/VideoChat3-4B"
MAX_NEW_TOKENS = 128
# qwen_vl_utils samples at `fps` up to its own default max_frames=768 -- fine for
# short clips, but at fps=1.0 a long step (e.g. a 28-minute clip) hits that cap
# and floods the forward pass with hundreds of video-token frames, which is what
# actually OOMs generate() (max_memory only bounds where WEIGHTS get placed at
# load time, not this per-call activation/KV-cache cost). Capping max_frames
# bounds token count regardless of clip length.
DEFAULT_MAX_VIDEO_FRAMES = 32

VALID_LABELS = ("ADOPT", "MODIFY", "REJECT/BYPASS")


def truncated_videos_dir(setting):
    return ROOT_DIR / "extracted_timestamps" / "truncated_videos" / f"experiment{setting}"


def prompts_dir(setting, condition="video-only"):
    return condition_prompts_dir(setting, condition)


def default_output(setting, use_video_chat3, use_32b=False, condition="video-only"):
    slug = "videochat3" if use_video_chat3 else ("qwen32b" if use_32b else "qwen")
    return condition_results_path(setting, condition, slug)


def _max_memory_map(headroom_gib=6):
    """Cap each GPU's model-weight budget a few GiB below its physical capacity
    (device_map="auto" otherwise packs weights right up to the edge), so
    there's still room left for a long video clip's activation/KV-cache memory
    without OOMing -- see the CUDACachingAllocator OOM this was added for: a
    dense 32B model in bf16 needs ~64GB, so it's already sharded across
    multiple GPUs, and the default packing left very little free on one card,
    not enough for some clips' visual-token activations."""
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        i: f"{max(int(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)) - headroom_gib, 1)}GiB"
        for i in range(torch.cuda.device_count())
    }


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


def get_model_and_processor(use_video_chat3, use_32b=False):
    # Lazy import so --dry-run works even if torch/transformers aren't installed yet.
    max_memory = _max_memory_map()

    if use_video_chat3:
        from transformers import AutoModelForCausalLM, AutoProcessor

        model = AutoModelForCausalLM.from_pretrained(
            VIDEOCHAT3_MODEL_NAME, torch_dtype="auto", device_map="auto", trust_remote_code=True,
            max_memory=max_memory,
        )
        processor = AutoProcessor.from_pretrained(VIDEOCHAT3_MODEL_NAME, trust_remote_code=True)
        return model, processor

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_name = QWEN_32B_MODEL_NAME if use_32b else QWEN_MODEL_NAME
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", max_memory=max_memory
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


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


def call_qwen(model, processor, prompt_text, video_path, max_frames=DEFAULT_MAX_VIDEO_FRAMES):
    from qwen_vl_utils import process_vision_info

    content = [{"type": "text", "text": prompt_text}]
    if video_path is not None:  # None for the graph-only / chat-only / graph-chat conditions: text only
        video_uri = f"file://{os.path.abspath(video_path)}"
        content.insert(0, {"type": "video", "video": video_uri, "fps": 1.0, "max_frames": max_frames})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return answer


def call_video_chat3(model, processor, prompt_text, video_path):
    """
    NOT independently verified end-to-end -- based on the "video" content-part
    chat-template convention MCG-NJU/VideoChat3-4B's model card documents,
    which mirrors Qwen-VL's, but without a qwen_vl_utils-equivalent helper
    (VideoChat3 isn't a Qwen model). If this errors on a real run, check the
    actual demo script (demo_vc3_proactive.py in the VideoChat3 repo) for the
    exact expected message/processor call shape and adjust here.
    """
    video_uri = f"file://{os.path.abspath(video_path)}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_uri},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return answer


def run_setting(setting, selected, output, args, model, processor, model_label, condition="video-only"):
    """Run `selected` samples for one setting with the already-loaded model,
    appending to `output` (rows already in it are skipped)."""
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
        for sample in tqdm(selected, desc=f"Running {model_label} {condition_folder(setting, condition)} experiments", unit="clip"):
            if sample["experiment_id"] in already_done:
                tqdm.write(f"Skipping (already in {output}): {sample['prompt_path'].name}")
                skipped += 1
                continue

            prompt_text = sample["prompt_path"].read_text()
            video_path = sample["video_path"]  # None for the graph-only / chat-only / graph-chat conditions
            size_bytes = video_path.stat().st_size if video_path is not None else 0

            tqdm.write("=" * 70)
            tqdm.write(f"Setting: {setting}  Condition: {condition}  Experiment: {sample['experiment_id']}")
            tqdm.write(f"Participant: {sample['participant_id']}  Session: {sample['session']}")
            if video_path is not None:
                tqdm.write(f"Video:  {video_path.name}  ({size_bytes / 1_000_000:.1f} MB)")
            else:
                tqdm.write("Video:  (none -- text-only condition, prompt text only)")
            tqdm.write(f"Prompt: {sample['prompt_path'].name}")
            tqdm.write(f"Ground-truth label (not sent to the model): {sample['ground_truth_label']}")

            if args.dry_run:
                continue

            try:
                if args.use_video_chat3:
                    if video_path is None:
                        raise ValueError("--use_video_chat3 needs a video; use a video-* condition")
                    answer = call_video_chat3(model, processor, prompt_text, sample["video_path"])
                else:
                    answer = call_qwen(model, processor, prompt_text, sample["video_path"], args.max_frames)
            except Exception as exc:
                tqdm.write(f"ERROR running {model_label} for {sample['experiment_id']}: {exc}")
                tqdm.write("Skipping this sample and continuing with the rest.")
                failed += 1
                continue
            finally:
                # Release cached-but-unallocated CUDA memory after every clip (not
                # just on failure) so fragmentation doesn't accumulate over a long
                # --all run of variable-length videos and OOM a later clip.
                import torch
                torch.cuda.empty_cache()

            tqdm.write(f"{model_label} answer: {answer}")

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
        help="Which setting(s) to run, e.g. --setting 1a or --setting 1a 1b (the model is loaded once and reused)",
    )
    parser.add_argument(
        "--condition", nargs="+", default=["video-only"], choices=CONDITION_CHOICES,
        help=(
            "Input mode(s), e.g. --condition video-only graph-only chat-only (default: video-only, the plain "
            "experiment<ID> run). Prompts come from prompts/experiment<ID>[_<mode>]/ (prompt_maker_setting.py) and "
            "results go to experiment_results/experiment<ID>[_<mode>]/. graph-only / chat-only / graph-chat send "
            "no video. Several modes (x several settings x several --run numbers) run in ONE process with "
            "the model loaded once."
        ),
    )
    parser.add_argument("--p", required=False, default=None, help="Participant id, e.g. p1")
    parser.add_argument("--session", required=False, default=None, choices=["1", "2"], help="Session number: 1 or 2")
    parser.add_argument(
        "--exp", required=False, default=None,
        help="Run only this one experiment_id (e.g. exp1_p1_s1), from the labels_experiment<N> CSV column",
    )
    parser.add_argument(
        "--limit", type=int, default=1,
        help="Max number of samples to actually run through the model, per setting (default: 1)",
    )
    parser.add_argument("--all", action="store_true", help="Ignore --limit and run every matching sample")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be sent (paths, sizes) without loading the model"
    )
    parser.add_argument(
        "--use_video_chat3", action="store_true",
        help=f"Use {VIDEOCHAT3_MODEL_NAME} instead of the default {QWEN_MODEL_NAME}",
    )
    parser.add_argument(
        "--32b", dest="use_32b", action="store_true",
        help=f"Use {QWEN_32B_MODEL_NAME} (dense) instead of the default {QWEN_MODEL_NAME} (MoE)",
    )
    parser.add_argument(
        "--max_frames", type=int, default=DEFAULT_MAX_VIDEO_FRAMES,
        help=(
            f"Cap on sampled video frames per clip (default: {DEFAULT_MAX_VIDEO_FRAMES}). "
            "Long clips at fps=1.0 can otherwise reach qwen_vl_utils' own default cap of "
            "768 frames, which is what actually drives OOM during generate() (not model-weight "
            "placement) -- lower this further if OOMs persist, especially with --32b."
        ),
    )
    parser.add_argument(
        "--output", required=False, default=None,
        help=(
            "CSV file to append results to (default: experiment_results/experiment<ID>/<qwen|qwen32b|videochat3>"
            "_experiment<ID>_results.csv, or see --run). Only allowed with a single --setting."
        ),
    )
    parser.add_argument(
        "--run", type=int, nargs="+", default=None,
        help=(
            "Run number(s) for this batch, e.g. --run 4 writes to "
            "experiment_results/experiment<ID>/<qwen|qwen32b|videochat3>_experiment<ID>_results_run4.csv, "
            "one file per setting. Several numbers (--run 1 2 3) repeat the whole batch once per "
            "number with the model loaded only once. Ignored if --output is also given."
        ),
    )
    args = parser.parse_args()

    if args.use_video_chat3 and args.use_32b:
        parser.error("--use_video_chat3 and --32b are mutually exclusive")

    settings = list(dict.fromkeys(args.setting))  # de-duplicate, keep order
    conditions = list(dict.fromkeys(args.condition))
    if args.output and (len(settings) > 1 or len(conditions) > 1):
        parser.error("--output can only be used with a single --setting and a single --condition "
                     "(each needs its own results file)")
    runs = list(dict.fromkeys(args.run)) if args.run else [None]
    if args.output and len(runs) > 1:
        parser.error("--output can only be used with a single --run (each run needs its own results file)")

    slug = "videochat3" if args.use_video_chat3 else ("qwen32b" if args.use_32b else "qwen")

    def output_for(setting, condition, run):
        if args.output:
            return args.output
        return str(condition_results_path(setting, condition, slug, run))

    model_label = "VideoChat3" if args.use_video_chat3 else ("Qwen3-VL-32B" if args.use_32b else "Qwen3-VL")

    participant_filter = args.p.strip().lower() if args.p else None
    session_filter = int(args.session) if args.session else None

    # Collect and select samples for every requested setting first, so the
    # (slow) model load happens once and only if there's something to run.
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

    if not args.dry_run:
        print_gpu_info()
    model, processor = (None, None) if args.dry_run else get_model_and_processor(args.use_video_chat3, args.use_32b)

    for run in runs:
        if run is not None:
            print(f"=== run {run} ===")
        for setting, condition, selected in plan:
            run_setting(setting, selected, output_for(setting, condition, run), args, model, processor,
                        model_label, condition)


if __name__ == "__main__":
    main()
