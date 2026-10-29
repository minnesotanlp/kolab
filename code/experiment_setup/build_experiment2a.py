"""
Builds extracted_timestamps/experiment2a/basic_experiment_setup.csv for Experiment 2a
(collaboration rating prediction). Scope: every AI-used step with a present collab
rating, for any participant who has a behavior graph in EITHER annotated_graphs (preferred)
OR graph_clean (fallback, used when the participant has no annotated_graphs file).
P19 session2 and P39 session1 are excluded for now per the user - their remaining
gaps are still open.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder
from config import RATING_AUDIT_DIR, ROOT_DIR
import os, re, csv, json, glob, sys, argparse

BASE = str(ROOT_DIR)
# all_steps_validity_vs_ai_usage.csv (written by rating_audit/valid_steps_summary.py) lives in <data>/rating_audit/
AUDIT_DIR = str(RATING_AUDIT_DIR)
from experiment_setup.rating_audit.csv_merge import merge_keep_old_only
from experiment_setup.rating_audit.audit_ratings import (get_final_participants, find_file, extract_paragraphs, parse_steps,
                           STEP_RE, HAC_RE, match_cr_trigger, classify_value, scan_forward)

ANNOTATED_GRAPHS_DIR = os.path.join(BASE, "new_annotated_graphs")  # the only graph source
OUT_DIR = os.path.join(BASE, "extracted_timestamps", "experiment2a")

EXCLUDED_SESSIONS = {("P19", "2"), ("P39", "1")}  # deferred per user, will use later

CHECKLIST_ITEMS = [
    ("collab_understood", "understood the task"),
    ("collab_adapted", "adapted when given feedback"),
    ("collab_reduced_effort", "reduced effort"),
    ("collab_proactive", "proactively suggested"),
    ("collab_accurate", "produced accurate"),
    ("collab_smooth", "felt smooth"),
    ("collab_guide_override", "guide, redirect, and override"),
]


def clean_rating(raw):
    if raw is None:
        return ''
    m = re.search(r'(\d(?:\.\d)?)', str(raw))
    return m.group(1) if m else ''


def extract_checklist(paras, start_idx, end_idx):
    """Return dict of the 7 canonical collaboration-checklist item ratings within paras[start:end].
    The rating is usually on the same line as the statement, but some participants put it on
    the very next paragraph instead (a lone digit)."""
    out = {key: '' for key, _ in CHECKLIST_ITEMS}
    for j in range(start_idx, min(end_idx, len(paras))):
        t = paras[j]
        tl = t.lower()
        for key, needle in CHECKLIST_ITEMS:
            if needle in tl and out[key] == '':
                stripped = re.sub(r'1[-–]5', '', t)
                m = re.search(r'([0-5](?:\.\d)?)', stripped)
                if m:
                    out[key] = m.group(1)
                elif j + 1 < end_idx:
                    nxt = paras[j + 1].strip()
                    m2 = re.match(r'^[\[=]?\s*([0-5](?:\.\d)?)\s*\]?\s*$', nxt)
                    if m2:
                        out[key] = m2.group(1)
    return out


LABEL_LINE_RE = re.compile(
    r'^(collaboration satisfaction|human-agent collaboration|\[|did the a[in]|why\b)', re.IGNORECASE)


def extract_completion_and_reason(paras, start_idx, end_idx):
    completion, reason = '', ''
    answer_consumed_next_line = False
    task_idx = None
    for j in range(start_idx, min(end_idx, len(paras))):
        t = paras[j].strip()
        if not t:
            continue
        if task_idx is None and 'successfully complete the task' in t.lower():
            task_idx = j
            tail = re.split(r'successfully complete the task\??', t, flags=re.IGNORECASE)[-1]
            tail = tail.strip(' :?/')
            if not re.fullmatch(r'yes\s*/\s*no\.?', tail, re.IGNORECASE) and tail != '':
                m = re.search(r'\b(yes|no)\b', tail, re.IGNORECASE)
                completion = m.group(1).capitalize() if m else tail
            if completion == '' and j + 1 < end_idx:
                nxt = paras[j + 1].strip()
                if nxt and len(nxt) < 20 and not LABEL_LINE_RE.match(nxt):
                    m = re.match(r'^(yes|no)\b', nxt, re.IGNORECASE)
                    completion = m.group(1).capitalize() if m else nxt
                    answer_consumed_next_line = True
            continue
        if task_idx is not None and reason == '' and j > task_idx:
            search_start = task_idx + (2 if answer_consumed_next_line else 1)
            if j < search_start:
                continue
            if re.match(r'^why\??\s*:?', t, re.IGNORECASE):
                tail = re.sub(r'^why\??\s*:?\s*', '', t, flags=re.IGNORECASE).strip()
                if tail:
                    reason = tail
                elif j + 1 < end_idx and not LABEL_LINE_RE.match(paras[j + 1].strip()):
                    reason = paras[j + 1].strip()
                break
            elif j == search_start and not LABEL_LINE_RE.match(t):
                reason = t
                break
            elif LABEL_LINE_RE.match(t):
                break
    return completion, reason


def find_step_block(paras, step_num):
    """Re-locate a step's paragraph block by number, same header logic as parse_steps."""
    headers = []
    last_num = 0
    for i, t in enumerate(paras):
        s = t.strip()
        m = STEP_RE.match(s)
        if m:
            num = int(m.group(1))
            if num > last_num:
                headers.append((i, num))
                last_num = num
    headers.append((len(paras), None))
    for idx in range(len(headers) - 1):
        if headers[idx][1] == step_num:
            return headers[idx][0], headers[idx + 1][0]
    return None, None


def find_graph(pid, session):
    """The participant/session graph in new_annotated_graphs/, or None."""
    num = pid[1:]
    for d in (ANNOTATED_GRAPHS_DIR,):
        pats = [f"session{session}_p{num}_*.json", f"session{session}_P{num}_*.json"]
        matches = []
        for p in pats:
            matches.extend(glob.glob(os.path.join(d, p)))
        if matches:
            return sorted(matches)[0]
    return None


def seconds_to_timestamp(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def time_ranges_for_step(graph_path, step_num):
    """Prefer start_seconds/end_seconds (numeric, machine-derived) over the
    free-text time_range string: at least one graph (P29 session2 step 3) has
    a time_range string that doesn't match its own start/end seconds (a
    copy-paste typo -- "16:46-12:04" reusing the previous step's end time as
    this step's start), while start_seconds/end_seconds were correct. Numeric
    fields also sidestep time_range strings using ambiguous "MM:SS" for
    anything past the first hour (e.g. "72:00" instead of "01:12:00"), which
    ffmpeg's -ss parser rejects outright once MM >= 60."""
    d = json.load(open(graph_path))
    ranges = []
    for s in d.get('steps', []):
        badge = s.get('badge', '') or ''
        if 'custom' in badge.lower():
            continue
        nums = [int(n) for n in re.findall(r'\d+', badge)]
        if step_num in nums:
            start_s, end_s = s.get('start_seconds'), s.get('end_seconds')
            if start_s is not None and end_s is not None:
                ranges.append(f"{seconds_to_timestamp(start_s)}-{seconds_to_timestamp(end_s)}")
            else:
                tr = s.get('time_range', '')
                if tr:
                    ranges.append(tr)
    return ranges


def mp4_name(pid, session):
    num = pid[1:]
    return f"session{session}_p{num}.mp4"


def load_target_steps(dummy=False):
    """Every step with graph-confirmed AI usage and a present collab rating,
    regardless of whether that graph lives in graph_clean or annotated_graphs
    (all_steps_validity_vs_ai_usage.csv's ai_used already reflects both, via
    valid_steps_summary.py's own graph_clean-then-annotated_graphs fallback).

    With dummy=True, selects the opposite: steps the graph confirms had NO AI
    usage at all, but which still have a real (non-fabricated) collaboration
    rating on file -- a bias-check set for "does the model's rating track the
    video, or does it just anchor on the prompt/graph template assuming AI was
    involved regardless of what's shown?" Requires valid=='True' too, since a
    couple of no-AI steps have a missing/invalid graph reference."""
    path = os.path.join(AUDIT_DIR, "all_steps_validity_vs_ai_usage.csv")
    targets = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r['participant'], r['session']) in EXCLUDED_SESSIONS:
                continue
            if dummy:
                if r['ai_used'] == 'no' and r['cr_status'] == 'found' and r['valid'] == 'True':
                    targets.append((r['participant'], r['session'], int(r['step_num'])))
            else:
                if r['ai_used'] == 'yes' and r['cr_status'] == 'found':
                    targets.append((r['participant'], r['session'], int(r['step_num'])))
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dummy", action="store_true",
        help="Build small_dummy_study.csv instead: steps with graph-confirmed NO AI usage, "
             "for testing whether models are biased toward assuming AI involvement from the "
             "prompt/graph template alone.",
    )
    args = parser.parse_args()

    targets = load_target_steps(dummy=args.dummy)
    doc_cache = {}
    rows = []

    for pid, session, step_num in targets:
        key = (pid, session)
        if key not in doc_cache:
            files = find_file(pid, session)
            paras = extract_paragraphs(files[0]) if files else None
            doc_cache[key] = paras
        paras = doc_cache[key]
        if paras is None:
            print(f"WARNING: no doc for {pid} s{session}, skipping step {step_num}")
            continue

        start_i, end_i = find_step_block(paras, step_num)
        if start_i is None:
            print(f"WARNING: could not relocate step {step_num} for {pid} s{session}")
            continue

        # re-derive main collaboration rating and its paragraph index within this block,
        # using the same trigger-matching logic as audit_ratings.py (handles the many
        # phrasing variants and rejects false-positive lines that merely mention the phrase)
        hac_idx = None
        cr_idx = None
        cr_raw = None
        for j in range(start_i, end_i):
            t = paras[j]
            if hac_idx is None and HAC_RE.search(t):
                hac_idx = j
            if cr_idx is None:
                r = match_cr_trigger(t)
                if r is not None:
                    cr_idx = j
                    cr_raw = r
        status, val = classify_value(cr_raw) if cr_idx is not None else ('missing', None)
        if status != 'found':
            status, val = scan_forward(paras, cr_idx + 1, end_i) if cr_idx is not None else ('missing', None)
        main_rating = clean_rating(val) if status == 'found' else ''

        checklist_end = cr_idx if cr_idx is not None else end_i
        checklist = extract_checklist(paras, hac_idx or start_i, checklist_end)
        completion, reason = extract_completion_and_reason(paras, hac_idx or start_i, checklist_end)

        graph_path = find_graph(pid, session)
        ranges = time_ranges_for_step(graph_path, step_num) if graph_path else []

        num = pid[1:]
        exp_id = f"p{num}_session{session}_step{step_num}_basic_setup"
        row = {
            'experiment_id': exp_id,
            'participant_id': f"p{num}",
            'session_id': f"session{session}",
            'step_id': f"step{step_num}",
            'mp4_file_name': mp4_name(pid, session),
            'timestamp_ranges': ';'.join(ranges),
            'main_collaboration_rating': main_rating,
            'agent_completion': completion,
            'reason': reason,
        }
        row.update(checklist)
        rows.append(row)

    fieldnames = ['experiment_id', 'participant_id', 'session_id', 'step_id', 'mp4_file_name',
                  'timestamp_ranges', 'main_collaboration_rating'] + [k for k, _ in CHECKLIST_ITEMS] + \
                 ['agent_completion', 'reason']

    out_path = os.path.join(OUT_DIR, "small_dummy_study.csv" if args.dummy else "basic_experiment_setup.csv")
    if not args.dummy:
        rows, fieldnames = merge_keep_old_only(rows, out_path, fieldnames, ANNOTATED_GRAPHS_DIR)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_path}")

    # quick data-quality check
    missing_checklist = sum(1 for r in rows if any(r[k] == '' for k, _ in CHECKLIST_ITEMS))
    missing_completion = sum(1 for r in rows if r['agent_completion'] == '')
    missing_reason = sum(1 for r in rows if r['reason'] == '')
    missing_timestamps = sum(1 for r in rows if r['timestamp_ranges'] == '')
    print(f"Rows with >=1 blank checklist item: {missing_checklist}")
    print(f"Rows with blank agent_completion: {missing_completion}")
    print(f"Rows with blank reason: {missing_reason}")
    print(f"Rows with no timestamp ranges: {missing_timestamps}")


if __name__ == '__main__':
    main()

