"""
Audits workflow_docs/*.docx (for the final ICLR participant list) for missing
Output Quality / Collaboration ratings per step, and checks availability of
Claude chat-log exports per participant/session. Read-only: does not modify
or process any chat log content, only checks file existence.

Outputs (written next to this script):
  missing_workflow_docs.csv  - participant/session with no workflow doc at all
  missing_ratings.csv        - steps where output_quality or collab_rating is blank/N-A
  format_anomalies.csv       - steps where the rating question itself wasn't found (manual check needed)
  missing_chat_logs.csv      - participant/session with no session*_claude_chats file
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the code/ folder
from config import RATING_AUDIT_DIR, ROOT_DIR
import zipfile, re, csv, os, glob

BASE = str(ROOT_DIR)
WFDIR = os.path.join(BASE, "workflow_docs")
CSV_PATH = os.path.join(BASE, "final_participants_for_iclr.csv")
OUT_DIR = str(RATING_AUDIT_DIR)  # generated CSVs go to <data>/rating_audit/

W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
STEP_RE = re.compile(r'^Step\s*(\d+)\s*(\[[^\]]*\])?\s*:?\s*(.*)$', re.IGNORECASE)
HAC_RE = re.compile(r'human-agent collaboration', re.IGNORECASE)
NA_VALUES = {'n/a', 'na', 'not assessed', 'not applicable', 'not rated', 'n / a'}


def get_final_participants():
    ids = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            pid = row[0].strip()
            if re.match(r'^P\d+$', pid):
                ids.append(pid)
    return ids


def extract_paragraphs(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET_fromstring(xml)
    paras = []
    for p in root.iter(W_NS + 'p'):
        texts = p.findall('.//' + W_NS + 't')
        paras.append(''.join(t.text or '' for t in texts))
    return paras


from xml.etree.ElementTree import fromstring as ET_fromstring


def find_file(pid, session):
    num = pid[1:]
    patterns = [
        f"P{num}_session{session}_workflow*.docx",
        f"p{num}_session{session}_workflow*.docx",
        f"P{num}_Session{session}_workflow*.docx",
        f"p{num}_Session{session}_workflow*.docx",
        f"P{num}_Workflow*Session{session}*.docx",
        f"p{num}_Workflow*Session{session}*.docx",
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(WFDIR, pat)))
    return sorted(set(m for m in matches if not os.path.basename(m).startswith('~$')))


def _tail_or_parenthetical(t):
    """Value normally follows the last ':'. Some participants instead wrote it as a
    bracketed number replacing the '(1 not satisfy, 5 satisfy)' clause, right before
    that colon (e.g. 'Rate the output quality from 1-5 (5): '). Fall back to that."""
    tail = t[t.rfind(':') + 1:]
    if re.search(r'\d', tail):
        return tail
    head = t[:t.rfind(':')]
    m = re.search(r'\((\d(?:\.\d)?)\)\s*$', head)
    return f"({m.group(1)})" if m else tail


def match_oq_trigger(t):
    if re.search(r'output quality', t, re.IGNORECASE) and not re.search(r'assessment rubric', t, re.IGNORECASE):
        if ':' in t:
            return _tail_or_parenthetical(t)
        m = re.search(r'output quality\D{0,15}?(\d(?:\.\d)?)\s*$', t, re.IGNORECASE)
        return m.group(1) if m else None
    return None


def match_cr_trigger(t):
    if re.search(r'overall collaboration experience', t, re.IGNORECASE):
        if ':' in t:
            return _tail_or_parenthetical(t)
        m = re.search(r'overall collaboration experience\D{0,15}?(\d(?:\.\d)?)\s*$', t, re.IGNORECASE)
        return m.group(1) if m else None
    return None


def classify_value(raw):
    if raw is None:
        return ('missing', raw)
    v = raw.strip().rstrip('.').strip()
    if v == '':
        return ('missing', raw)
    if v.lower() in NA_VALUES:
        return ('na', raw)
    if re.search(r'\d', v):
        return ('found', v)
    return ('missing', raw)


def scan_forward(paras, start_idx, stop_idx):
    for j in range(start_idx, min(stop_idx, len(paras))):
        t = paras[j].strip()
        if not t:
            continue
        m = re.match(r'^\[?\s*(\d(?:\.\d)?)\s*\]?', t)
        if m:
            return ('found', m.group(1))
        m2 = re.search(r'\brating\b[^:\n]*:\s*\[?\s*(\d(?:\.\d)?)', t, re.IGNORECASE)
        if m2:
            return ('found', m2.group(1))
        m3 = re.search(r'\bcollaboration\b[^:\n]*:\s*\[?\s*(\d(?:\.\d)?)', t, re.IGNORECASE)
        if m3:
            return ('found', m3.group(1))
        m4 = re.search(r'\[\s*(\d(?:\.\d)?)\s*/\s*5\s*\]', t)
        if m4:
            return ('found', m4.group(1))
        if t.lower() in NA_VALUES or re.fullmatch(r'n\s*/\s*a\.?', t, re.IGNORECASE):
            return ('na', t)
    return ('missing', None)


def parse_steps(paras):
    headers = []
    last_num = 0
    for i, t in enumerate(paras):
        s = t.strip()
        m = STEP_RE.match(s)
        if m:
            num = int(m.group(1))
            if num > last_num:
                headers.append((i, m.group(1), m.group(3).strip()))
                last_num = num
    headers.append((len(paras), None, None))

    results = []
    for idx in range(len(headers) - 1):
        start_i, step_num, step_name = headers[idx]
        end_i = headers[idx + 1][0]
        block = range(start_i, end_i)

        oq_trigger_idx = cr_trigger_idx = hac_idx = None
        oq_raw = cr_raw = None
        for j in block:
            t = paras[j]
            if oq_trigger_idx is None:
                r = match_oq_trigger(t)
                if r is not None:
                    oq_trigger_idx, oq_raw = j, r
            if cr_trigger_idx is None:
                r = match_cr_trigger(t)
                if r is not None:
                    cr_trigger_idx, cr_raw = j, r
            if hac_idx is None and HAC_RE.search(t):
                hac_idx = j

        if oq_trigger_idx is not None:
            status, val = classify_value(oq_raw)
            if status != 'found':
                scan_end = hac_idx if (hac_idx is not None and hac_idx > oq_trigger_idx) else end_i
                status, val = scan_forward(paras, oq_trigger_idx + 1, scan_end)
        else:
            status, val = 'no_trigger', None
        oq_status, oq_val = status, val

        if cr_trigger_idx is not None:
            status, val = classify_value(cr_raw)
            if status != 'found':
                status, val = scan_forward(paras, cr_trigger_idx + 1, end_i)
        else:
            status, val = 'no_trigger', None
        cr_status, cr_val = status, val

        results.append({
            'step_num': step_num, 'step_name': step_name,
            'oq_status': oq_status, 'oq_val': oq_val,
            'cr_status': cr_status, 'cr_val': cr_val,
        })
    return results


def has_chat_log(pid, session):
    num = pid[1:].lower()
    d = os.path.join(BASE, "extracted_timestamps", f"session{session}_chats")
    if not os.path.isdir(d):
        return False
    for fn in os.listdir(d):
        if fn.lower().startswith(f"p{num}_session{session}"):
            return True
    return False


def main():
    participants = get_final_participants()
    missing_files, flagged, no_trigger = [], [], []

    for pid in participants:
        for session in ['1', '2']:
            files = find_file(pid, session)
            if not files:
                missing_files.append({'participant': pid, 'session': session})
                continue
            for path in files:
                fname = os.path.basename(path)
                try:
                    paras = extract_paragraphs(path)
                except Exception as e:
                    flagged.append({'participant': pid, 'session': session, 'file': fname,
                                     'step': 'N/A', 'issue': f'ERROR reading file: {e}'})
                    continue
                steps = parse_steps(paras)
                if not steps:
                    flagged.append({'participant': pid, 'session': session, 'file': fname,
                                     'step': 'N/A', 'issue': 'NO STEPS PARSED'})
                    continue
                for st in steps:
                    label = f"Step {st['step_num']}: {st['step_name']}"
                    for key, name in [('oq', 'output_quality'), ('cr', 'collab_rating')]:
                        status = st[f'{key}_status']
                        if status == 'missing':
                            flagged.append({'participant': pid, 'session': session, 'file': fname,
                                             'step': label, 'field': name, 'issue': 'missing'})
                        elif status == 'na':
                            flagged.append({'participant': pid, 'session': session, 'file': fname,
                                             'step': label, 'field': name, 'issue': 'marked_na'})
                        elif status == 'no_trigger':
                            no_trigger.append({'participant': pid, 'session': session, 'file': fname,
                                                'step': label, 'field': name})

    missing_chats = []
    for pid in participants:
        for session in ['1', '2']:
            if not has_chat_log(pid, session):
                missing_chats.append({'participant': pid, 'session': session})

    def write_csv(name, rows, fieldnames):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, name)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {len(rows)} rows -> {path}")

    write_csv('missing_workflow_docs.csv', missing_files, ['participant', 'session'])
    write_csv('missing_ratings.csv', flagged, ['participant', 'session', 'file', 'step', 'field', 'issue'])
    write_csv('format_anomalies.csv', no_trigger, ['participant', 'session', 'file', 'step', 'field'])
    write_csv('missing_chat_logs.csv', missing_chats, ['participant', 'session'])

    print(f"\nTotal final participants: {len(participants)}")


if __name__ == '__main__':
    main()

