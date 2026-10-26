"""
Builds extracted_timestamps/experiment2b/basic_experiment_setup.csv for Experiment 2b
(output-quality rating prediction). Scope: every AI-used step with a present output-
quality rating, for any participant who has a behavior graph in EITHER annotated_graphs
(preferred) OR graph_clean (fallback, used when the participant has no annotated_graphs
file).

Unlike build_experiment2a.py, this does NOT hardcode an EXCLUDED_SESSIONS set --
that exclusion (P19 session2, P39 session1) was specific to two unresolved
*collaboration*-rating gaps and may not apply to output-quality data at all. The
oq_status == 'found' row-level filter below naturally includes/excludes whatever
is appropriate for output quality.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder
from config import RATING_AUDIT_DIR, ROOT_DIR
import os, re, csv, json, glob, sys

BASE = str(ROOT_DIR)
# all_steps_validity_vs_ai_usage.csv (written by rating_audit/valid_steps_summary.py) lives in <data>/rating_audit/
AUDIT_DIR = str(RATING_AUDIT_DIR)
from experiment_setup.rating_audit.csv_merge import merge_keep_old_only
from experiment_setup.rating_audit.audit_ratings import (get_final_participants, find_file, extract_paragraphs, parse_steps,
                           STEP_RE, HAC_RE, match_oq_trigger, classify_value, scan_forward)

ANNOTATED_GRAPHS_DIR = os.path.join(BASE, "new_annotated_graphs")  # the only graph source
OUT_DIR = os.path.join(BASE, "extracted_timestamps", "experiment2b")


def clean_rating(raw):
    if raw is None:
        return ''
    m = re.search(r'(\d(?:\.\d)?)', str(raw))
    return m.group(1) if m else ''


# Boundary lines that mark the end of the output-quality "Reason for your rating:"
# field -- extended from build_experiment2a.py's LABEL_LINE_RE (used there for the
# separate, later collaboration/task-completion "Why?" field) to also treat
# "which ones of the rubrics above..." as a stop/boundary, since that's the
# template line that immediately follows the output-quality reason field.
OQ_BOUNDARY_RE = re.compile(
    r'^(which ones of the rubrics|collaboration satisfaction|human-agent collaboration|\[|did the a[in]|why\b)',
    re.IGNORECASE)
OQ_REASON_LABEL_RE = re.compile(r'^reason for your rating\??\s*:?', re.IGNORECASE)


def extract_output_quality_reason(paras, oq_trigger_idx, end_idx):
    """Extract the output-quality 'Reason for your rating:' text -- a DIFFERENT
    field from the later Human-Agent Collaboration section's 'Why?' reason
    (that one is extracted by build_experiment2a.py's extract_completion_and_reason;
    do not reuse it here, it targets the wrong field).

    Starts searching just after the paragraph where the output-quality rating
    trigger was found (oq_trigger_idx). Looks for a paragraph matching
    '^reason for your rating\\??\\s*:?' (case-insensitive):
      - if that line has trailing text after the label, returns it;
      - else falls through to the next non-empty paragraph, UNLESS that
        paragraph is itself a template label/boundary line (OQ_BOUNDARY_RE),
        in which case the reason is treated as blank.
    Stops searching (returns '') once a boundary line is hit before any
    "Reason for your rating:" label is found, or once paras run out."""
    if oq_trigger_idx is None:
        return ''
    for j in range(oq_trigger_idx + 1, min(end_idx, len(paras))):
        t = paras[j].strip()
        if not t:
            continue
        if OQ_BOUNDARY_RE.match(t):
            break
        if OQ_REASON_LABEL_RE.match(t):
            tail = OQ_REASON_LABEL_RE.sub('', t).strip()
            if tail:
                return tail
            for k in range(j + 1, min(end_idx, len(paras))):
                nxt = paras[k].strip()
                if not nxt:
                    continue
                if OQ_BOUNDARY_RE.match(nxt):
                    return ''
                return nxt
            return ''
    return ''


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


def load_target_steps():
    """Every step with graph-confirmed AI usage and a present output-quality
    rating, regardless of whether that graph lives in graph_clean or
    annotated_graphs (all_steps_validity_vs_ai_usage.csv's ai_used already
    reflects both, via valid_steps_summary.py's own graph_clean-then-
    annotated_graphs fallback)."""
    path = os.path.join(AUDIT_DIR, "all_steps_validity_vs_ai_usage.csv")
    targets = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['ai_used'] == 'yes' and r['oq_status'] == 'found':
                targets.append((r['participant'], r['session'], int(r['step_num'])))
    return targets


def main():
    targets = load_target_steps()
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

        # re-derive output-quality rating and its paragraph index within this block,
        # using the same trigger-matching logic as audit_ratings.py (handles the many
        # phrasing variants and rejects false-positive lines that merely mention the phrase)
        hac_idx = None
        oq_idx = None
        oq_raw = None
        for j in range(start_i, end_i):
            t = paras[j]
            if hac_idx is None and HAC_RE.search(t):
                hac_idx = j
            if oq_idx is None:
                r = match_oq_trigger(t)
                if r is not None:
                    oq_idx = j
                    oq_raw = r
        status, val = classify_value(oq_raw) if oq_idx is not None else ('missing', None)
        if status != 'found' and oq_idx is not None:
            scan_end = hac_idx if (hac_idx is not None and hac_idx > oq_idx) else end_i
            status, val = scan_forward(paras, oq_idx + 1, scan_end)
        output_quality_rating = clean_rating(val) if status == 'found' else ''

        reason = extract_output_quality_reason(paras, oq_idx, end_i)

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
            'output_quality_rating': output_quality_rating,
            'reason': reason,
        }
        rows.append(row)

    fieldnames = ['experiment_id', 'participant_id', 'session_id', 'step_id', 'mp4_file_name',
                  'timestamp_ranges', 'output_quality_rating', 'reason']

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "basic_experiment_setup.csv")
    rows, fieldnames = merge_keep_old_only(rows, out_path, fieldnames, ANNOTATED_GRAPHS_DIR)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_path}")

    # quick data-quality check
    missing_rating = sum(1 for r in rows if r['output_quality_rating'] == '')
    missing_reason = sum(1 for r in rows if r['reason'] == '')
    missing_timestamps = sum(1 for r in rows if r['timestamp_ranges'] == '')
    print(f"Rows with blank output_quality_rating: {missing_rating}")
    print(f"Rows with blank reason: {missing_reason}")
    print(f"Rows with no timestamp ranges: {missing_timestamps}")


if __name__ == '__main__':
    main()

