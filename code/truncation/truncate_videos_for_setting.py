"""
Truncate source videos into per-experiment clips.

Two input modes are supported via --setting:

1. experiment1a / experiment1b (Experiment 1): reads
   every CSV in "../extracted_timestamps/labels_<setting>/" (as produced by
   experiment1a/1b_extract_user_response_labels.py). Each CSV's filename
   carries a "session<N>_p<M>" id, and each row has begin_timestamp /
   end_timestamp columns for a single clip.

2. experiment2a (Experiment 2a, collaboration-rating prediction): reads the
   single combined CSV at
   "../extracted_timestamps/experiment2a/basic_experiment_setup.csv". Each
   row is one workflow step, with an mp4_file_name column and a
   timestamp_ranges column (semicolon-separated "begin-end" ranges -- more
   than one when the step's footage is disjoint, e.g. the participant
   revisited it later). All ranges for a row are truncated and concatenated
   into a single clip per step. This mode also rewrites
   basic_experiment_setup.csv in place, adding (or updating, if already
   present) a "truncated_video_filename" column with the resulting clip's
   filename.

In both modes, for each row this script:
1. Looks up the source video at
   "../session<N>_videos/real_videos/session<N>_p<M>.mp4".
2. Writes the output clip to
   "../extracted_timestamps/truncated_videos/<setting>/".
3. Skips truncation if a clip with that exact output name already exists in
   that folder, so re-running is idempotent.

Trimming uses `ffmpeg -ss <begin> -i <source> -t <duration> ...` (fast
input seek, then accurate re-encode of just the clip's duration) so cuts
land on the exact requested timestamps without re-encoding the whole
source video. Multi-range (experiment2a) clips are joined with ffmpeg's
concat demuxer (`-c copy`, no re-encode for the join itself).

Usage:
    python truncate_videos_for_setting.py --setting experiment1a
    python truncate_videos_for_setting.py --setting experiment2a
    python truncate_videos_for_setting.py --setting experiment2b   # cuts any clip missing for a 2b-only step into the shared experiment2a clips folder; leaves the 2b CSV unchanged
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder
from config import EXTRACTED_DIR, ROOT_DIR

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = ROOT_DIR
EXTRACTED_TIMESTAMPS_DIR = EXTRACTED_DIR
TRUNCATED_VIDEOS_DIR = EXTRACTED_TIMESTAMPS_DIR / "truncated_videos"
EXPERIMENT2A_CSV = EXTRACTED_TIMESTAMPS_DIR / "experiment2a" / "basic_experiment_setup.csv"
EXPERIMENT2B_CSV = EXTRACTED_TIMESTAMPS_DIR / "experiment2b" / "basic_experiment_setup.csv"
EXPERIMENT2A_TRUNCATED_COLUMN = "truncated_video_filename"

SUPPORTED_SETTINGS = ["experiment1a", "experiment1b", "experiment2a", "experiment2b"]

SESSION_P_RE = re.compile(r"session(\d+)_p(\d+)")


def timestamp_to_seconds(ts):
    parts = [int(part) for part in ts.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def parse_timestamp_ranges(ranges_str):
    """Split an experiment2a timestamp_ranges cell (semicolon-separated
    "begin-end" pieces, dash or en-dash) into a list of (begin, end) strings."""
    ranges = []
    for part in ranges_str.split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = re.split(r"[-–]", part, maxsplit=1)
        if len(pieces) != 2 or not pieces[0].strip() or not pieces[1].strip():
            print(f"Warning: could not parse timestamp range {part!r}; skipping.", file=sys.stderr)
            continue
        ranges.append((pieces[0].strip(), pieces[1].strip()))
    return ranges


def safe_timestamp(ts):
    return ts.strip().replace(":", "-")


def find_source_video(session, participant):
    """Usually "session<S>_p<M>.mp4"; at least one (session1_p16) is instead
    named "clean_session<S>_p<M>.mp4", so fall back to that before giving up."""
    real_videos_dir = PROJECT_ROOT / f"session{session}_videos" / "real_videos"
    for name in (f"session{session}_p{participant}.mp4", f"clean_session{session}_p{participant}.mp4"):
        video_path = real_videos_dir / name
        if video_path.exists():
            return video_path
    return None


# ---- clip checks: an existing clip is only reused when it is as long as its window, and a window
# that does not fit inside the source video is reported instead of being cut into an empty / short clip.
_SOURCE_DURATIONS = {}


def media_duration(path):
    """Duration in seconds from ffprobe, or None if the file is missing / unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        return float(out.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def source_duration(source_video):
    if source_video not in _SOURCE_DURATIONS:
        _SOURCE_DURATIONS[source_video] = media_duration(source_video)
    return _SOURCE_DURATIONS[source_video]


def window_fits_video(source_video, ranges_seconds, name):
    """False (with a warning) if any (begin, end) in seconds is not inside the source video:
    a start at / after its end gives an empty clip, an end past it a silently shorter one."""
    total = source_duration(source_video)
    if total is None:
        return True  # cannot tell; do not block the cut
    for begin, end in ranges_seconds:
        if begin >= total - 1 or end > total + 2:
            print(
                f"Warning: NEEDS A HUMAN: {name}: window {begin}-{end} s does not fit inside {source_video.name} "
                f"({total:.0f} s long); not cut, an existing clip is left as it is.",
                file=sys.stderr,
            )
            return False
    return True


def existing_clip_ok(output_path, expected_seconds):
    """True if the clip on disk is readable and as long as its window (3 s or 1 % tolerance)."""
    actual = media_duration(output_path)
    if actual is None or actual < 1:
        print(f"Warning: existing clip {output_path.name} is empty/unreadable; recutting.", file=sys.stderr)
        return False
    if abs(actual - expected_seconds) > max(3.0, 0.01 * expected_seconds):
        print(f"Warning: existing clip {output_path.name} is {actual:.0f} s but its window is "
              f"{expected_seconds:.0f} s (cut from an older window?); recutting.", file=sys.stderr)
        return False
    return True


def truncate_clip(source_video, begin_timestamp, end_timestamp, output_path, fps=None):
    begin_seconds = timestamp_to_seconds(begin_timestamp)
    end_seconds = timestamp_to_seconds(end_timestamp)
    duration = end_seconds - begin_seconds
    if duration <= 0:
        print(
            f"Warning: skipping {output_path.name!r}: end_timestamp "
            f"({end_timestamp}) is not after begin_timestamp ({begin_timestamp}).",
            file=sys.stderr,
        )
        return False

    part_path = output_path.with_name(output_path.name + ".part")
    part_path.unlink(missing_ok=True)  # left over from an interrupted run
    cmd = [
        "ffmpeg",
        "-y",
        # Pass computed seconds, not the raw timestamp string: ffmpeg's -ss parser
        # rejects a bare "MM:SS" once MM >= 60 (e.g. "72:00" -> "Invalid duration
        # for option ss"), which some graph_clean time_range strings use for
        # anything past the first hour instead of "HH:MM:SS". Plain seconds are
        # accepted unambiguously either way.
        "-ss", str(begin_seconds),
        "-i", str(source_video),
        "-t", str(duration),
        *(["-vf", f"fps={fps}"] if fps else []),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        # Written under a temporary name and renamed when finished, so a clip with its final
        # name is always complete (a model run or the "already exists" check never sees half a file).
        "-f", "mp4",
        str(part_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: ffmpeg failed for {output_path.name!r}:\n{result.stderr}", file=sys.stderr)
        part_path.unlink(missing_ok=True)
        return False
    os.replace(part_path, output_path)
    return True


def truncate_and_concat_clip(source_video, ranges, output_path):
    """Truncate each (begin, end) range and join them into a single output
    clip, in order. Used for experiment2a rows whose footage is disjoint
    (the step was revisited later in the session)."""
    if len(ranges) == 1:
        begin, end = ranges[0]
        return truncate_clip(source_video, begin, end, output_path)

    part_path = output_path.with_name(output_path.name + ".part")
    part_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        segment_paths = []
        for i, (begin, end) in enumerate(ranges):
            segment_path = tmp_dir_path / f"segment_{i}.mp4"
            if not truncate_clip(source_video, begin, end, segment_path):
                return False
            segment_paths.append(segment_path)

        concat_list_path = tmp_dir_path / "concat_list.txt"
        with open(concat_list_path, "w") as f:
            for segment_path in segment_paths:
                f.write(f"file '{segment_path.resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",
            "-f", "mp4",
            str(part_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: ffmpeg concat failed for {output_path.name!r}:\n{result.stderr}", file=sys.stderr)
            part_path.unlink(missing_ok=True)
            return False
        os.replace(part_path, output_path)
    return True


def process_experiment2a_csv(csv_path, output_dir, setting="experiment2a", write_column=True):
    """Truncate (and concatenate, for disjoint footage) a clip per row of
    basic_experiment_setup.csv, then (write_column=True, the default) rewrite that
    CSV in place with a truncated_video_filename column recording each row's
    output clip. write_column=False leaves the CSV untouched (experiment2b: its
    runner finds clips by experiment_id, and it shares 2a's clips)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if write_column and EXPERIMENT2A_TRUNCATED_COLUMN not in fieldnames:
        fieldnames.append(EXPERIMENT2A_TRUNCATED_COLUMN)

    created, skipped, failed = 0, 0, 0
    for row in tqdm(rows, total=len(rows)):
        experiment_id = row.get("experiment_id", "").strip()
        mp4_file_name = row.get("mp4_file_name", "").strip()
        timestamp_ranges = row.get("timestamp_ranges", "").strip()
        row.setdefault(EXPERIMENT2A_TRUNCATED_COLUMN, "")

        if not experiment_id or not mp4_file_name or not timestamp_ranges:
            print(f"Warning: row missing experiment_id/mp4_file_name/timestamp_ranges; skipping row: {row}",
                  file=sys.stderr)
            failed += 1
            continue

        match = SESSION_P_RE.search(mp4_file_name)
        if not match:
            print(f"Warning: could not find 'session<N>_p<M>' in mp4_file_name {mp4_file_name!r}; "
                  f"skipping {experiment_id!r}.", file=sys.stderr)
            failed += 1
            continue
        session, participant = match.group(1), match.group(2)

        source_video = find_source_video(session, participant)
        if source_video is None:
            print(f"Warning: source video for session{session}_p{participant} not found "
                  f"under 'session{session}_videos/real_videos/'; skipping {experiment_id!r}.", file=sys.stderr)
            failed += 1
            continue

        ranges = parse_timestamp_ranges(timestamp_ranges)
        if not ranges:
            print(f"Warning: no valid timestamp ranges for {experiment_id!r}; skipping.", file=sys.stderr)
            failed += 1
            continue

        output_name = f"{experiment_id}.mp4"
        output_path = output_dir / output_name

        windows = [(timestamp_to_seconds(b), timestamp_to_seconds(e)) for b, e in ranges]
        if not window_fits_video(source_video, windows, output_name):
            failed += 1
            continue
        if output_path.exists():
            if existing_clip_ok(output_path, sum(e - b for b, e in windows)):
                print(f"[{setting}] Already exists, skipping: {output_path.name}")
                row[EXPERIMENT2A_TRUNCATED_COLUMN] = output_name
                skipped += 1
                continue
            output_path.unlink()

        print(f"[{setting}] Truncating {source_video.name} -> {output_name} ({len(ranges)} range(s)) ...")
        if truncate_and_concat_clip(source_video, ranges, output_path):
            row[EXPERIMENT2A_TRUNCATED_COLUMN] = output_name
            created += 1
        else:
            failed += 1

    if write_column:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return created, skipped, failed


def warn_if_ranges_differ_from_2a(csv_path_2b):
    """experiment2b clips are the same files as experiment2a's (one <experiment_id>.mp4
    per step, existing clips are skipped, never recut). That is only right if a step
    has the same timestamp_ranges in both CSVs, so say so when one differs."""
    def ranges(path):
        with open(path, newline="") as f:
            return {r["experiment_id"].strip(): r.get("timestamp_ranges", "").strip() for r in csv.DictReader(f)}

    if not EXPERIMENT2A_CSV.is_file():
        return
    ranges_2a = ranges(EXPERIMENT2A_CSV)
    for experiment_id, timestamp_ranges in ranges(csv_path_2b).items():
        if experiment_id in ranges_2a and ranges_2a[experiment_id] != timestamp_ranges:
            print(
                f"Warning: {experiment_id!r} has timestamp_ranges {timestamp_ranges!r} in experiment2b but "
                f"{ranges_2a[experiment_id]!r} in experiment2a; the shared clip is NOT recut, so 2b would "
                f"use 2a's footage for it.",
                file=sys.stderr,
            )


def process_csv(csv_path, output_dir, setting):
    match = SESSION_P_RE.search(csv_path.stem)
    if not match:
        print(
            f"Warning: could not find 'session<N>_p<M>' in {csv_path.name!r}; skipping file.",
            file=sys.stderr,
        )
        return 0, 0

    session, participant = match.group(1), match.group(2)
    source_video = find_source_video(session, participant)
    if source_video is None:
        print(
            f"Warning: source video for session{session}_p{participant} not found "
            f"under 'session{session}_videos/real_videos/'; skipping {csv_path.name!r}.",
            file=sys.stderr,
        )
        return 0, 0

    created, skipped = 0, 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            begin_timestamp = row.get("begin_timestamp", "").strip()
            end_timestamp = row.get("end_timestamp", "").strip()
            if not begin_timestamp or not end_timestamp:
                print(
                    f"Warning: row with missing timestamp in {csv_path.name!r}; skipping row.",
                    file=sys.stderr,
                )
                continue

            output_name = (
                f"session{session}_p{participant}_"
                f"{safe_timestamp(begin_timestamp)}_{safe_timestamp(end_timestamp)}.mp4"
            )
            output_path = output_dir / output_name

            window = (timestamp_to_seconds(begin_timestamp), timestamp_to_seconds(end_timestamp))
            if not window_fits_video(source_video, [window], output_name):
                continue
            if output_path.exists():
                if existing_clip_ok(output_path, window[1] - window[0]):
                    print(f"[{setting}] Already exists, skipping: {output_path.name}")
                    skipped += 1
                    continue
                output_path.unlink()

            print(f"[{setting}] Truncating {source_video.name} -> {output_name} ...")
            if truncate_clip(source_video, begin_timestamp, end_timestamp, output_path):
                if (media_duration(output_path) or 0) < 1:
                    print(f"Warning: {output_name} came out empty; removed.", file=sys.stderr)
                    output_path.unlink(missing_ok=True)
                else:
                    created += 1

    return created, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setting",
        required=True,
        choices=SUPPORTED_SETTINGS,
        help="experiment1a/experiment1b (CSVs), experiment2a (Experiment 2a) or "
             "experiment2b (Experiment 2b; cuts into the shared experiment2a clips folder)",
    )
    args = parser.parse_args()

    # experiment2b has no clips folder of its own: it shares experiment2a's (same source
    # video and timestamp ranges per experiment_id), so existing clips are never recut.
    output_dir = TRUNCATED_VIDEOS_DIR / ("experiment2a" if args.setting == "experiment2b" else args.setting)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.setting == "experiment2b":
        if not EXPERIMENT2B_CSV.is_file():
            raise SystemExit(f"No such file: {EXPERIMENT2B_CSV}")

        warn_if_ranges_differ_from_2a(EXPERIMENT2B_CSV)
        created, skipped, failed = process_experiment2a_csv(
            EXPERIMENT2B_CSV, output_dir, setting="experiment2b", write_column=False
        )
        print(
            f"[experiment2b] Done. Created {created} clip(s), skipped {skipped} already-existing clip(s), "
            f"{failed} row(s) failed/skipped. (Clips are in {output_dir}, shared with experiment2a.)"
        )
        return

    if args.setting == "experiment2a":
        if not EXPERIMENT2A_CSV.is_file():
            raise SystemExit(f"No such file: {EXPERIMENT2A_CSV}")

        created, skipped, failed = process_experiment2a_csv(EXPERIMENT2A_CSV, output_dir)
        print(
            f"[experiment2a] Done. Created {created} clip(s), skipped {skipped} already-existing clip(s), "
            f"{failed} row(s) failed/skipped."
        )
        return

    csv_dir = EXTRACTED_TIMESTAMPS_DIR / f"labels_{args.setting}"
    if not csv_dir.is_dir():
        raise SystemExit(f"No such directory: {csv_dir}")

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}", file=sys.stderr)
        return

    total_created, total_skipped = 0, 0
    for csv_path in tqdm(csv_files, total=len(csv_files)):
        created, skipped = process_csv(csv_path, output_dir, args.setting)
        total_created += created
        total_skipped += skipped

    print(f"[{args.setting}] Done. Created {total_created} clip(s), skipped {total_skipped} already-existing clip(s).")


if __name__ == "__main__":
    main()

