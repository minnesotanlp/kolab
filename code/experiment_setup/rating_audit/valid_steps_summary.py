"""
Computes how many workflow-doc steps currently have complete ratings
(output_quality + collab_rating both present), excluding steps where the
graph shows no AI-actor node at all (i.e. the participant didn't use Claude
for that step). AI usage is read from node['actor_type'] in the behavior
graphs in new_annotated_graphs/ (the only graph source). A session with no graph there keeps the
ai_used / graph_file already in all_steps_validity_vs_ai_usage.csv instead of becoming
'unknown_no_graph', so its steps stay in the 2a/2b setups.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the code/ folder
from config import RATING_AUDIT_DIR, ROOT_DIR
import os, re, csv, json, glob, sys

from experiment_setup.rating_audit.audit_ratings import get_final_participants, find_file, extract_paragraphs, parse_steps

BASE = str(ROOT_DIR)
GRAPH_DIRS = [os.path.join(BASE, "new_annotated_graphs")]
OUT_DIR = str(RATING_AUDIT_DIR)  # generated CSVs go to <data>/rating_audit/


def find_graph(pid, session):
    """The participant/session graph in new_annotated_graphs/, or None."""
    num = pid[1:]
    for d in GRAPH_DIRS:
        if not os.path.isdir(d):
            continue
        pats = [f"session{session}_p{num}_*.json", f"session{session}_P{num}_*.json"]
        matches = []
        for p in pats:
            matches.extend(glob.glob(os.path.join(d, p)))
        if matches:
            return sorted(matches)[0]
    return None


def ai_usage_by_step(path):
    """Return {step_num: bool_ai_used} aggregated across all graph segments for that step_num."""
    d = json.load(open(path))
    result = {}
    for s in d.get('steps', []):
        badge = s.get('badge', '') or ''
        if 'custom' in badge.lower():
            continue
        nums = set(int(n) for n in re.findall(r'\d+', badge))
        has_ai = any(n.get('actor_type') == 'ai' for n in s.get('nodes', []))
        for num in nums:
            result[num] = result.get(num, False) or has_ai
    return result


def read_previous_usage(path):
    """{(participant, session, step_num): (ai_used, graph_file)} from the last run's CSV."""
    if not os.path.isfile(path):
        return {}
    with open(path, newline='', encoding='utf-8') as f:
        return {(r['participant'], r['session'], r['step_num']): (r['ai_used'], r['graph_file'])
                for r in csv.DictReader(f)}


def main():
    participants = get_final_participants()
    previous = read_previous_usage(os.path.join(OUT_DIR, "all_steps_validity_vs_ai_usage.csv"))
    rows = []
    for pid in participants:
        for session in ['1', '2']:
            files = find_file(pid, session)
            if not files:
                continue
            path = files[0]
            try:
                paras = extract_paragraphs(path)
                steps = parse_steps(paras)
            except Exception as e:
                print(f"ERROR parsing {pid} s{session}: {e}")
                continue

            graph_path = find_graph(pid, session)
            ai_map = ai_usage_by_step(graph_path) if graph_path else {}

            for st in steps:
                step_num = int(st['step_num']) if st['step_num'] else None
                oq_ok = st['oq_status'] == 'found'
                cr_ok = st['cr_status'] == 'found'
                valid = oq_ok and cr_ok
                previous_usage = previous.get((pid, session, str(step_num)))
                if graph_path is None and previous_usage:
                    ai_used = previous_usage[0]  # session has no graph here any more: keep what was known
                elif graph_path is None:
                    ai_used = 'unknown_no_graph'
                elif step_num in ai_map:
                    ai_used = 'yes' if ai_map[step_num] else 'no'
                else:
                    ai_used = 'unknown_step_not_in_graph'

                rows.append({
                    'participant': pid, 'session': session, 'step_num': step_num,
                    'step_name': st['step_name'], 'oq_status': st['oq_status'],
                    'cr_status': st['cr_status'], 'valid': valid,
                    'graph_file': os.path.basename(graph_path) if graph_path else (previous_usage[1] if previous_usage else ''),
                    'ai_used': ai_used,
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "all_steps_validity_vs_ai_usage.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['participant', 'session', 'step_num', 'step_name',
                                          'oq_status', 'cr_status', 'valid', 'graph_file', 'ai_used'])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}\n")

    total = len(rows)
    no_ai = [r for r in rows if r['ai_used'] == 'no']
    yes_ai = [r for r in rows if r['ai_used'] == 'yes']
    unknown = [r for r in rows if r['ai_used'].startswith('unknown')]

    print(f"Total steps across all {len(participants)} final participants (both sessions): {total}")
    print(f"  Steps with graph-confirmed NO AI usage (excluded): {len(no_ai)}")
    print(f"  Steps with graph-confirmed AI usage: {len(yes_ai)}")
    print(f"  Steps with unknown AI usage (no graph / step not in graph): {len(unknown)}")

    def valid_count(subset):
        return sum(1 for r in subset if r['valid'])

    print()
    print(f"Among AI-used steps: {valid_count(yes_ai)}/{len(yes_ai)} valid (both ratings present)")
    print(f"Among unknown-usage steps: {valid_count(unknown)}/{len(unknown)} valid (both ratings present)")
    print(f"Among no-AI-usage steps (excluded from denominator): {valid_count(no_ai)}/{len(no_ai)} valid anyway")

    denom = yes_ai + unknown
    print()
    print(f"==> Valid steps (excluding confirmed no-AI steps): {valid_count(denom)}/{len(denom)}")

    print()
    print("--- No-AI-usage steps detail (excluded) ---")
    for r in no_ai:
        print(f"  {r['participant']} s{r['session']} Step {r['step_num']}: {r['step_name']} "
              f"(oq={r['oq_status']}, cr={r['cr_status']}, valid={r['valid']})")


if __name__ == '__main__':
    main()

