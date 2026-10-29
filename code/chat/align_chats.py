"""
Stage 2 of the Experiment 2a chat ablation (see run_chat_pipeline.sh).

For every row of chat_anchors.csv whose video_first_mmss is filled, writes
extracted_timestamps/session<S>_chats_aligned/<p>_session<S>_chats.csv: the
session's actual chat messages (Human and Claude rows with non-empty "chat" text;
tool-use/workflow rows -- speaker Claude with an empty "chat" field and the
ai_workflow/tool_use/AI_high_level_action columns filled instead -- are dropped,
they were never part of the visible conversation), stable-sorted by time, plus
two columns on the session video's clock,

    video_ts_seconds   e.g. 1352
    video_ts           e.g. 00:22:32

with  video_ts = wall-clock timestamp + offset  and
      offset   = video_first_mmss - chat_first_ts   (the first human message).

Sanity check: a warning is printed when any row lands before 0 or past the end of
the session video (found via ffprobe in session<S>_videos/ when possible) -- that
means the anchor is probably wrong.

Rows with a blank or malformed video_first_mmss are skipped (never a guessed offset).
Aligned files are simply rewritten on every run.

Usage:
    python align_chats.py
    python align_chats.py --p p1 --session 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import csv
import subprocess

from chat.chat_common import (
    ROLE_LABEL, ROOT_DIR, aligned_chat_path, aligned_chats_dir, chat_files, format_hms, parse_mmss,
    parse_wallclock, read_anchors, read_chat_rows, warn,
)


def find_session_video(participant_id, session_id):
    folder = ROOT_DIR / f"{session_id}_videos"
    for name in (f"{session_id}_{participant_id}.mp4", f"{participant_id}_video_{session_id}.mp4"):
        if (folder / name).is_file():
            return folder / name
    return None


def video_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", default=None, help="Only this participant id, e.g. p1")
    parser.add_argument("--session", default=None, choices=["1", "2"], help="Only this session")
    args = parser.parse_args()

    written = 0
    for anchor in read_anchors():
        participant_id, session_id = anchor["participant_id"], anchor["session_id"]
        if args.p and participant_id != args.p.strip().lower():
            continue
        if args.session and session_id != f"session{args.session}":
            continue
        mmss = (anchor.get("video_first_mmss") or "").strip()
        if not mmss:
            continue
        try:
            video_first = parse_mmss(mmss)
        except ValueError as exc:
            warn(f"{participant_id} {session_id}: {exc}; skipped.")
            continue
        if not chat_files(participant_id, session_id):
            warn(f"{participant_id} {session_id}: no chat file; skipped.")
            continue

        offset = video_first - parse_wallclock(anchor["chat_first_ts"])
        rows = [
            row for row in read_chat_rows(participant_id, session_id)
            if row["speaker"].strip().lower() in ROLE_LABEL and row["chat"].strip()
        ]
        if not rows:
            warn(f"{participant_id} {session_id}: no chat-text rows (only tool-use/workflow rows); skipped.")
            continue
        for row in rows:
            seconds = parse_wallclock(row["timestamp"]) + offset
            row["video_ts_seconds"] = round(seconds, 1)
            row["video_ts"] = format_hms(seconds) if seconds >= 0 else f"-{format_hms(-seconds)}"

        first, last = rows[0]["video_ts_seconds"], rows[-1]["video_ts_seconds"]
        duration = None
        video = find_session_video(participant_id, session_id)
        if video is not None:
            duration = video_duration(video)
        if first < 0 or (duration is not None and last > duration):
            warn(
                f"{participant_id} {session_id}: aligned chat spans {first:.0f}s..{last:.0f}s but the video is "
                f"{'%.0fs' % duration if duration else 'of unknown length'} -- check the anchor {mmss!r}."
            )
        elif duration is None:
            print(f"Note: no session video found to length-check {participant_id} {session_id}.")

        aligned_chats_dir(session_id).mkdir(parents=True, exist_ok=True)
        out_path = aligned_chat_path(participant_id, session_id)
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{participant_id} {session_id}: {len(rows)} row(s), video {format_hms(first)}..{format_hms(last)} -> {out_path.name}")
        written += 1

    print(f"Aligned {written} session(s).")


if __name__ == "__main__":
    main()
