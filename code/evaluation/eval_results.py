"""
Compute evaluation metrics for experiment1a/experiment1b/experiment2a/experiment2b results
CSVs (the output of run_gemini_setting.py / run_qwen_setting.py / run_gpt_setting.py /
run_kimi_setting.py / run_experiment2a.py / run_experiment2b.py).

Auto-discovers result files instead of requiring you to type paths:

    python eval_results.py --setting 1a --model gemini
    python eval_results.py --setting 1b --model qwen
    python eval_results.py --setting experiment2a --model flash
    python eval_results.py --setting experiment2b --model flash

This looks in experiment_results/experiment<ID>/ (or experiment_results/experiment2a/
for --setting experiment2a, experiment_results/experiment2b/ for --setting
experiment2b) for every file matching "<model>_experiment<ID>_results*.csv" (or
"<model>_experiment2a_results*.csv" / "<model>_experiment2b_results*.csv") --
that glob catches both the bare *_results.csv (no --run given) and every
*_results_run<R>.csv. If exactly one file matches, it prints that single run's
report. If more than one matches (e.g. run1/run2/run3), it prints each run's
own report plus a mean +/- std summary across all of them -- no need to know
in advance how many runs exist or list them by hand.

Experiment 1a/1b also each have their own graph/chat ablation conditions
(prompt_maker_setting.py's SETTING_CONDITIONS: video_graph, graph_only, video_chat,
chat_only, graph_chat, video_graph_chat). Pass the condition's own folder name as
--setting (it already includes the "experiment" prefix, unlike the bare "1a"/"1b"):

    python eval_results.py --setting experiment1a_graph_only --model qwen
    python eval_results.py --setting experiment1b_chat_only --model gemini

These are still ADOPT/MODIFY/REJECT-BYPASS classification, evaluated exactly like
plain "1a"/"1b" -- only the results folder differs (experiment_results/experiment1a_graph_only/
etc., matching what a runner script saves there once it supports these conditions).

--model accepts: gemini, flash, gpt, qwen, videochat3, kimi (either "flash" or
"gemini_flash" works for any --setting -- run_gemini_setting.py itself saves
gemini-3.8-flash results under the "gemini_flash" prefix for experiment1a/3, while
run_experiment2a.py saves the same model under plain "flash"; this script
normalizes between the two so you don't have to remember which).

Pass --input to instead evaluate one or more specific files directly (bypasses
auto-discovery, e.g. for a custom --output path you gave a runner script).

Two different metric sets, depending on --setting:

experiment1a / experiment1b (ADOPT / MODIFY / REJECT-BYPASS classification):
  accuracy, per-class and macro/weighted precision/recall/F1, confusion matrix.
  Labels are normalized before scoring: lowercased, and "reject" is treated the
  same as "reject/bypass" (the model is asked for "REJECT/BYPASS" but the
  ground-truth CSVs in labels_experiment1a use the shorter "reject").

experiment2a / experiment2b, and every one of their graph/video/chat conditions
(experiment2a_video_graph, experiment2a_graph_only, experiment2a_video_chat,
experiment2a_chat_only, experiment2a_graph_chat, experiment2a_video_graph_chat, the
_ts graph variants, and the same for experiment2b) -- see ORDINAL_SETTINGS -- share
one metric set: this is a 1-5 rating prediction regardless of which sources (video,
graph text, chat text) the model was given, so the --setting value only changes
which results folder is read, never which metrics are computed.

experiment2a (1-5 collaboration-rating prediction): plain classification
metrics (accuracy/F1 treating 1,2,3,4,5 as unrelated categories) are a poor
fit for an ordinal 1-5 scale -- predicting 4 when the truth is 5 is a much
smaller error than predicting 1, but accuracy/F1 score both as equally wrong.
So this instead reports:
  - MAE / RMSE: average (squared) distance between predicted and true rating
    -- directly rewards "closer" answers.
  - Exact-match and within-1 accuracy (|predicted - true| <= 1).
  - Pearson r and Spearman rho: does the model's ranking track the human's,
    even if the absolute scale is shifted.
  - Quadratic Weighted Kappa (QWK): the standard ordinal-agreement metric for
    Likert-style ratings (e.g. essay/exam scoring) -- agreement above chance,
    weighted so a 2-point miss counts far more than a 1-point miss. Ratings
    are rounded to the nearest whole number for this metric only, since QWK
    needs discrete categories (some ground-truth ratings in this dataset are
    e.g. 4.5, averaged from a checklist).
Predicted values are parsed permissively (first number found in the model's
response), since the prompt asks for exactly one integer but a model may
occasionally add stray text.

Incremental mode (--update-reports): scans every experiment_results/<setting>/
<model>_<setting>_results*.csv, compares each file's content hash against
report/.eval_state.json, and for every (setting, model) group where a file is new,
appended to or rewritten, saves a fresh report/<model>_<setting>_report_<timestamp>.txt
(all of that group's runs, so the mean +/- std stays consistent). Unchanged groups are
skipped. Use --dry-run to only list what changed; --force to regenerate everything.

Usage:
    python eval_results.py --update-reports
    python eval_results.py --setting 1a --model gemini
    python eval_results.py --setting 1a --model gemini_flash
    python eval_results.py --setting 1b --model qwen
    python eval_results.py --setting experiment2a --model flash
    python eval_results.py --setting 1a --input ../experiment_results/experiment1a/gemini_experiment1a_results_run1.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import contextlib
import csv
import hashlib
import io
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from experiments import CLASSIFICATION_EXPERIMENTS
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, cohen_kappa_score

from config import ROOT_DIR
RESULTS_ROOT = ROOT_DIR / "experiment_results"
REPORT_DIR = ROOT_DIR / "report"
STATE_FILE = REPORT_DIR / ".eval_state.json"

CLASSIFICATION_FIELDNAMES = [
    "participant_id", "experiment_id", "real_answer", "predicted_answer",
    "video_clip_length", "step_jump",
]
ORDINAL_FIELDNAMES = [
    "participant_id", "experiment_id", "real_answer", "predicted_answer",
    "video_clip_length", "num_timestamp_ranges",
]

YESNO_FIELDNAMES = [
    "participant_id", "experiment_id", "real_answer", "predicted_answer",
    "video_clip_length", "num_timestamp_ranges", "cumulative_step_count", "total_step_count",
]
# experiment2c predicts Yes/No (goal achieved), so it uses the classification metrics with
# YESNO_FIELDNAMES; its folder/file names are the bare setting name like the ordinal settings.
YESNO_SETTINGS = ("experiment2c",)

RUN_NUMBER_RE = re.compile(r"_run(\d+)\.csv$")

# experiment2a/2b: (band name, label, rounded true ratings in the band)
RATING_BANDS = (
    ("low", "1-2", (1, 2)),
    ("moderate", "3", (3,)),
    ("mod-high", "4", (4,)),
    ("high", "5", (5,)),
)

# experiment2a/experiment2b (and every experiment2a graph condition) predict a 1-5
# rating, so they share the ordinal metrics. For these, the --setting value is also the
# folder under experiment_results/ and the middle of the result file names
# (<model>_<setting>_results[_run<R>].csv).
ORDINAL_SETTINGS = (
    "experiment2a", "experiment2b",
    "experiment2a_graph",  # the original 126-step video+graph run (run_experiment2a.py --with-graph)
    "experiment2a_video_graph", "experiment2a_video_graph_ts",
    "experiment2a_graph_only", "experiment2a_graph_only_ts",
    # the chat ablation (run_chat_pipeline.sh / run_experiment2a.py --condition video-graph-chat|...)
    "experiment2a_video_graph_chat", "experiment2a_video_chat",
    "experiment2a_graph_chat", "experiment2a_chat_only",
    # the same conditions for experiment2b (run_experiment2b.py --condition ...)
    "experiment2b_video_graph", "experiment2b_video_graph_ts",
    "experiment2b_graph_only", "experiment2b_graph_only_ts",
    "experiment2b_video_graph_chat", "experiment2b_video_chat",
    "experiment2b_graph_chat", "experiment2b_chat_only",
)
# run_experiment2a.py --condition spelling -> --setting spelling (2a only; use the full
# experiment2b_<...> folder names for 2b's graph conditions)
CONDITION_ALIASES = {
    "video-only": "experiment2a",
    "video-graph": "experiment2a_video_graph",
    "video-graph-ts": "experiment2a_video_graph_ts",
    "graph-only": "experiment2a_graph_only",
    "graph-only-ts": "experiment2a_graph_only_ts",
    "video-graph-chat": "experiment2a_video_graph_chat",
    "video-chat": "experiment2a_video_chat",
    "graph-chat": "experiment2a_graph_chat",
    "chat-only": "experiment2a_chat_only",
}

# experiment1a/3/5's own graph/chat ablation conditions (prompt_maker_setting.py's
# SETTING_CONDITIONS). Still ADOPT/MODIFY/REJECT-BYPASS classification, same as plain
# "1a"/"1b" -- NOT in ORDINAL_SETTINGS -- just read from their own results folder, whose
# name is the --setting value itself (it already starts with "experiment", unlike the bare
# "1a"/"1b" which need "experiment" prepended -- see classification_folder()).
SETTING_GRAPH_CHAT_CONDITIONS = tuple(
    f"experiment{n}_{cond}"
    for n in CLASSIFICATION_EXPERIMENTS
    for cond in ("video_graph", "graph_only", "video_chat", "chat_only", "graph_chat", "video_graph_chat")
)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def classification_folder(setting):
    """Folder name (under experiment_results/) of a classification (experiment1a/experiment1b)
    run: bare "1a"/"1b" (video-only) need "experiment" prepended, but the new
    experiment1a_<condition>/experiment1b_<condition> keys already are the folder name."""
    return setting if setting.startswith(("experiment", *YESNO_SETTINGS)) else f"experiment{setting}"


def results_dir_for(setting):
    if setting in ORDINAL_SETTINGS:
        return ROOT_DIR / "experiment_results" / setting
    return ROOT_DIR / "experiment_results" / classification_folder(setting)


def resolve_model_slug(setting, model):
    """Normalize --model so the same name works regardless of --setting, even
    though the runner scripts don't save with a consistent slug themselves:
    run_gemini_setting.py (experiment1a/3) saves gemini-3.8-flash results under
    the "gemini_flash" prefix, while run_experiment2a.py/run_experiment2b.py
    save the same model under plain "flash". Accept either spelling for
    either setting."""
    if setting not in ORDINAL_SETTINGS and setting not in YESNO_SETTINGS and model == "flash":
        return "gemini_flash"
    if (setting in ORDINAL_SETTINGS or setting in YESNO_SETTINGS) and model == "gemini_flash":
        return "flash"
    return model


def discover_result_files(setting, model):
    """Every file matching "<model>_experiment<ID>_results*.csv" (or
    "<model>_experiment2a_results*.csv" / "<model>_experiment2b_results*.csv"),
    sorted with the bare (no --run) file first, then ascending by run number."""
    directory = results_dir_for(setting)
    suffix = setting if setting in ORDINAL_SETTINGS else classification_folder(setting)
    model = resolve_model_slug(setting, model)
    pattern = f"{model}_{suffix}_results*.csv"
    matches = sorted(directory.glob(pattern))

    def sort_key(path):
        run_match = RUN_NUMBER_RE.search(path.name)
        return (0, 0) if run_match is None else (1, int(run_match.group(1)))

    return sorted(matches, key=sort_key)


# ---------------------------------------------------------------------------
# Shared CSV reading
# ---------------------------------------------------------------------------

def read_rows(path, fieldnames):
    """Read a results CSV whether or not it has a header row."""
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f) if row]

    if not rows:
        return []
    if rows[0] == fieldnames:
        rows = rows[1:]

    # Keep only the first row per experiment_id (as progress.py does): two runs appending to the
    # same file can write an id twice, which would otherwise count that sample twice.
    out, seen, dropped = [], set(), 0
    for row in rows:
        record = dict(zip(fieldnames, row))
        eid = record.get("experiment_id")
        if eid in seen:
            dropped += 1
            continue
        seen.add(eid)
        out.append(record)
    if dropped:
        print(f"Warning: {path}: ignored {dropped} duplicate experiment_id row(s); "
              f"run clean_results.py to remove them.", file=sys.stderr)
    return out


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# experiment1a / experiment1b: ADOPT/MODIFY/REJECT-BYPASS classification metrics
# ---------------------------------------------------------------------------

def normalize_label(label):
    # Reasoning with the label inline ("...the user will likely **MODIFY** the output..."): an
    # all-caps label is the answer when it's the only distinct one in the text.
    caps = {re.sub(r"\s", "", m) for m in re.findall(r"\b(ADOPT|MODIFY|REJECT\s*/\s*BYPASS|REJECT|BYPASS)\b", label)}
    label = label.strip().lower()
    # A leading label followed by reasoning ("ADOPT\n\nReasoning: ...", "**modify** because ...") counts as that label.
    leading = re.match(r"^\W*(adopt|modify|reject\s*/\s*bypass|reject|bypass)\b", label)
    # Reasoning first, label on its own last line ("...\n\n**MODIFY**", "**Prediction:** adopt").
    trailing = re.match(r"^\W*(?:\w+\s*:\W*)?(adopt|modify|reject\s*/\s*bypass|reject|bypass)\W*$",
                        label.splitlines()[-1].strip()) if label else None
    if leading or trailing:
        label = (leading or trailing).group(1)
    elif len(caps) == 1:
        label = caps.pop().lower()
    if label.replace(" ", "") in ("reject", "bypass", "reject/bypass"):
        return "reject/bypass"
    yes_no = re.match(r"^\W*(yes|no)\b", label)  # experiment2c: "Yes." / "No, because..." -> yes / no
    if yes_no:
        return yes_no.group(1)
    return label


def compute_classification_metrics(rows, labels):
    y_true = [normalize_label(r["real_answer"]) for r in rows]
    y_pred = [normalize_label(r["predicted_answer"]) for r in rows]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n": len(rows),
        "accuracy": accuracy,
        "per_class": {
            label: {"precision": p, "recall": r, "f1": f, "support": s}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "confusion_matrix": cm,
    }


def print_classification_report(title, metrics, labels, macro_only=False):
    print(title)
    print(f"n = {metrics['n']}")

    macro = metrics["macro"]
    if macro_only:
        print(
            f"macro precision = {macro['precision']:.3f}   "
            f"macro recall = {macro['recall']:.3f}   "
            f"macro f1 = {macro['f1']:.3f}"
        )
        return

    print(f"accuracy = {metrics['accuracy']:.3f}\n")

    print(f"{'label':<15}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label in labels:
        m = metrics["per_class"][label]
        print(f"{label:<15}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10.0f}")

    weighted = metrics["weighted"]
    print(f"\n{'macro avg':<15}{macro['precision']:>10.3f}{macro['recall']:>10.3f}{macro['f1']:>10.3f}{metrics['n']:>10d}")
    print(f"{'weighted avg':<15}{weighted['precision']:>10.3f}{weighted['recall']:>10.3f}{weighted['f1']:>10.3f}{metrics['n']:>10d}")

    print("\nconfusion matrix (rows = actual, cols = predicted)")
    header = " " * 16 + "".join(f"{label:>15}" for label in labels)
    print(header)
    for label, row in zip(labels, metrics["confusion_matrix"]):
        print(f"{label:<16}" + "".join(f"{count:>15d}" for count in row))


def print_classification_average_report(all_metrics, labels, macro_only=False):
    n_runs = len(all_metrics)
    print("=" * 60)
    print(f"Average across {n_runs} run(s)\n")

    if macro_only:
        p_mean, p_std = mean_std([m["macro"]["precision"] for m in all_metrics])
        r_mean, r_std = mean_std([m["macro"]["recall"] for m in all_metrics])
        f_mean, f_std = mean_std([m["macro"]["f1"] for m in all_metrics])
        print(
            f"macro precision = {p_mean:.3f} +/- {p_std:.3f}   "
            f"macro recall = {r_mean:.3f} +/- {r_std:.3f}   "
            f"macro f1 = {f_mean:.3f} +/- {f_std:.3f}"
        )
        return

    acc_mean, acc_std = mean_std([m["accuracy"] for m in all_metrics])
    print(f"accuracy = {acc_mean:.3f} +/- {acc_std:.3f}\n")

    print(f"{'label':<15}{'precision':>16}{'recall':>16}{'f1':>16}")
    for label in labels:
        p_mean, p_std = mean_std([m["per_class"][label]["precision"] for m in all_metrics])
        r_mean, r_std = mean_std([m["per_class"][label]["recall"] for m in all_metrics])
        f_mean, f_std = mean_std([m["per_class"][label]["f1"] for m in all_metrics])
        print(
            f"{label:<15}"
            f"{p_mean:>9.3f}+/-{p_std:<5.3f}"
            f"{r_mean:>9.3f}+/-{r_std:<5.3f}"
            f"{f_mean:>9.3f}+/-{f_std:<5.3f}"
        )

    for avg_kind in ("macro", "weighted"):
        p_mean, p_std = mean_std([m[avg_kind]["precision"] for m in all_metrics])
        r_mean, r_std = mean_std([m[avg_kind]["recall"] for m in all_metrics])
        f_mean, f_std = mean_std([m[avg_kind]["f1"] for m in all_metrics])
        label = f"{avg_kind} avg"
        print(
            f"{label:<15}"
            f"{p_mean:>9.3f}+/-{p_std:<5.3f}"
            f"{r_mean:>9.3f}+/-{r_std:<5.3f}"
            f"{f_mean:>9.3f}+/-{f_std:<5.3f}"
        )


# ---------------------------------------------------------------------------
# experiment2a: ordinal (1-5 rating) metrics
# ---------------------------------------------------------------------------

def parse_rating(text):
    """Pull the first number out of a rating field. Permissive about the
    predicted_answer, since the prompt asks for exactly one integer but a
    model may occasionally wrap it in stray text."""
    if text is None:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def _fractional_ranks(values):
    """Average ("fractional") ranks, 1-based, with ties sharing the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs, ys):
    if len(xs) < 2:
        return float("nan")
    return pearson_r(_fractional_ranks(xs), _fractional_ranks(ys))


def compute_ordinal_metrics(rows):
    reals, preds = [], []
    unparseable = 0
    for r in rows:
        real = parse_rating(r["real_answer"])
        pred = parse_rating(r["predicted_answer"])
        if real is None or pred is None:
            unparseable += 1
            continue
        reals.append(real)
        preds.append(pred)

    n = len(reals)
    if n == 0:
        return {"n": 0, "unparseable": unparseable}

    errors = [p - t for p, t in zip(preds, reals)]
    abs_errors = [abs(e) for e in errors]
    mae = statistics.mean(abs_errors)
    rmse = (statistics.mean(e ** 2 for e in errors)) ** 0.5
    exact_acc = sum(1 for e in abs_errors if e == 0) / n
    within1_acc = sum(1 for e in abs_errors if e <= 1.0) / n

    # QWK needs discrete categories; round to the nearest whole rating just
    # for this metric (some ground-truth ratings are e.g. 4.5).
    rounded_true = [int(round(min(max(t, 1), 5))) for t in reals]
    rounded_pred = [int(round(min(max(p, 1), 5))) for p in preds]
    qwk = cohen_kappa_score(rounded_true, rounded_pred, weights="quadratic", labels=[1, 2, 3, 4, 5])

    # accuracy broken out by TRUE rating (1-5): reveals whether accuracy is
    # concentrated on the dominant class(es) while rare true ratings are
    # missed almost entirely -- invisible in the aggregate exact/within-1 acc.
    by_true_rating = {}
    for rating in (1, 2, 3, 4, 5):
        idx = [i for i, t in enumerate(rounded_true) if t == rating]
        if not idx:
            by_true_rating[rating] = None
            continue
        bucket_abs_errors = [abs_errors[i] for i in idx]
        by_true_rating[rating] = {
            "n": len(idx),
            "exact_acc": sum(1 for e in bucket_abs_errors if e == 0) / len(idx),
            "within1_acc": sum(1 for e in bucket_abs_errors if e <= 1.0) / len(idx),
            "mean_pred": statistics.mean(preds[i] for i in idx),
            "mae": statistics.mean(bucket_abs_errors),
        }

    # the same, but grouped into the rating bands the report viewer shows as columns
    by_band = {}
    for band, _, ratings in RATING_BANDS:
        idx = [i for i, t in enumerate(rounded_true) if t in ratings]
        if not idx:
            by_band[band] = None
            continue
        bucket_abs_errors = [abs_errors[i] for i in idx]
        by_band[band] = {
            "n": len(idx),
            "exact_acc": sum(1 for e in bucket_abs_errors if e == 0) / len(idx),
            "within1_acc": sum(1 for e in bucket_abs_errors if e <= 1.0) / len(idx),
            "mae": statistics.mean(bucket_abs_errors),
        }

    return {
        "n": n,
        "unparseable": unparseable,
        "mae": mae,
        "rmse": rmse,
        "exact_acc": exact_acc,
        "within1_acc": within1_acc,
        "pearson_r": pearson_r(reals, preds),
        "spearman_rho": spearman_rho(reals, preds),
        "qwk": qwk,
        "by_true_rating": by_true_rating,
        "by_band": by_band,
    }


def print_band_lines(all_metrics):
    """One line per rating band (read by report_viewer.py). With several runs each value
    is mean +/- std over the runs that have instances in that band."""
    for band, ratings_label, _ in RATING_BANDS:
        buckets = [m["by_band"][band] for m in all_metrics if m["by_band"].get(band)]
        prefix = f"band {band} ({ratings_label})"
        if not buckets:
            print(f"{prefix:<22} n = 0  exact acc. = --  within-1 acc. = --  MAE = --")
            continue
        if len(all_metrics) == 1:
            b = buckets[0]
            print(
                f"{prefix:<22} n = {b['n']}  exact acc. = {b['exact_acc']:.3f}  "
                f"within-1 acc. = {b['within1_acc']:.3f}  MAE = {b['mae']:.3f}"
            )
            continue
        acc_mean, acc_std = mean_std([b["exact_acc"] for b in buckets])
        w1_mean, w1_std = mean_std([b["within1_acc"] for b in buckets])
        mae_mean, mae_std = mean_std([b["mae"] for b in buckets])
        print(
            f"{prefix:<22} n = {buckets[0]['n']}  exact acc. = {acc_mean:.3f} +/- {acc_std:.3f}  "
            f"within-1 acc. = {w1_mean:.3f} +/- {w1_std:.3f}  MAE = {mae_mean:.3f} +/- {mae_std:.3f}"
        )


def print_ordinal_report(title, metrics):
    print(title)
    print(f"n = {metrics['n']}", end="")
    if metrics.get("unparseable"):
        print(f"  ({metrics['unparseable']} row(s) skipped: unparseable real/predicted rating)")
    else:
        print()
    if metrics["n"] == 0:
        print("  (no usable rows)")
        return
    print(f"MAE            = {metrics['mae']:.3f}")
    print(f"RMSE           = {metrics['rmse']:.3f}")
    print(f"exact accuracy = {metrics['exact_acc']:.3f}")
    print(f"within-1 acc.  = {metrics['within1_acc']:.3f}")
    print(f"Pearson r      = {metrics['pearson_r']:.3f}")
    print(f"Spearman rho   = {metrics['spearman_rho']:.3f}")
    print(f"quadratic weighted kappa = {metrics['qwk']:.3f}")

    print("\naccuracy by true rating (rounded)")
    print(f"{'true rating':<12}{'n':>6}{'exact acc.':>12}{'within-1 acc.':>15}{'mean pred':>12}{'MAE':>8}")
    for rating in (1, 2, 3, 4, 5):
        bucket = metrics["by_true_rating"].get(rating)
        if bucket is None:
            print(f"{rating:<12}{0:>6}{'--':>12}{'--':>15}{'--':>12}{'--':>8}")
        else:
            print(
                f"{rating:<12}{bucket['n']:>6}{bucket['exact_acc']:>12.3f}"
                f"{bucket['within1_acc']:>15.3f}{bucket['mean_pred']:>12.2f}{bucket['mae']:>8.3f}"
            )

    print("\naccuracy by rating band")
    print_band_lines([metrics])


def print_ordinal_average_report(all_metrics):
    usable = [m for m in all_metrics if m["n"] > 0]
    n_runs = len(all_metrics)
    print("=" * 60)
    print(f"Average across {n_runs} run(s)\n")
    if not usable:
        print("  (no usable rows in any run)")
        return

    for key, label in [
        ("mae", "MAE"), ("rmse", "RMSE"), ("exact_acc", "exact accuracy"),
        ("within1_acc", "within-1 acc."), ("pearson_r", "Pearson r"),
        ("spearman_rho", "Spearman rho"), ("qwk", "quadratic weighted kappa"),
    ]:
        mean, std = mean_std([m[key] for m in usable])
        print(f"{label:<26} = {mean:.3f} +/- {std:.3f}")

    print("\naccuracy by rating band")
    print_band_lines(usable)


# ---------------------------------------------------------------------------
# Incremental report generation
# ---------------------------------------------------------------------------

def known_result_folders():
    return {f"experiment{n}" for n in CLASSIFICATION_EXPERIMENTS} | {*ORDINAL_SETTINGS, *YESNO_SETTINGS, *SETTING_GRAPH_CHAT_CONDITIONS}


def file_fingerprint(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def run_number(path):
    m = RUN_NUMBER_RE.search(path.name)
    return (0, 0) if m is None else (1, int(m.group(1)))


def scan_result_groups():
    """{(setting, model): [paths]} for every results CSV under experiment_results/."""
    groups = {}
    for folder in sorted(known_result_folders()):
        directory = RESULTS_ROOT / folder
        if not directory.is_dir():
            continue
        name_re = re.compile(rf"^(.+?)_{re.escape(folder)}_results(?:_run\d+)?\.csv$")
        for path in directory.glob("*.csv"):
            match = name_re.match(path.name)
            if match:
                groups.setdefault((folder, match.group(1)), []).append(path)
    for paths in groups.values():
        paths.sort(key=run_number)
    return groups


def update_reports(macro_only=False, dry_run=False, force=False):
    try:
        state = json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written = 0
    for (setting, model), paths in scan_result_groups().items():
        fingerprints = {str(p.relative_to(ROOT_DIR)): file_fingerprint(p) for p in paths}
        key = f"{setting}/{model}"
        old = state.get(key, {})
        changed = [f for f, h in fingerprints.items() if old.get(f) != h]
        removed = [f for f in old if f not in fingerprints]
        if not (force or changed or removed):
            continue

        report_path = REPORT_DIR / f"{model}_{setting}_report_{timestamp}.txt"
        why = ", ".join([Path(f).name for f in changed]
                        + [f"removed {Path(f).name}" for f in removed]) or "forced"
        if dry_run:
            print(f"[dry-run] would write {report_path.name}  (changed: {why})")
            continue

        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                print(f"Model: {model} -- {setting} -- generated {datetime.now():%c}")
                print(f"Files: {', '.join(p.name for p in paths)}\n")
                print_reports(setting, paths, macro_only)
        except Exception as exc:  # leave state untouched so it is retried next time
            print(f"Skipped {key}: {exc}")
            continue
        REPORT_DIR.mkdir(exist_ok=True)
        report_path.write_text(buffer.getvalue())
        state[key] = fingerprints
        written += 1
        print(f"Wrote {report_path.relative_to(ROOT_DIR)}  (changed: {why})")

    if not dry_run:
        REPORT_DIR.mkdir(exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=1, sort_keys=True))
        if not written:
            print("No new or modified result files; nothing to report.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_prefix_accuracy(rows):
    """experiment2c: accuracy by how many steps of the session were shown
    (cumulative_step_count out of total_step_count)."""
    by_k = {}
    for r in rows:
        by_k.setdefault(int(r["cumulative_step_count"]), []).append(
            normalize_label(r["real_answer"]) == normalize_label(r["predicted_answer"])
        )
    full = [
        normalize_label(r["real_answer"]) == normalize_label(r["predicted_answer"])
        for r in rows if r["cumulative_step_count"] == r["total_step_count"]
    ]
    print("Accuracy by number of steps shown:")
    for k in sorted(by_k):
        hits = by_k[k]
        print(f"  first {k} step(s): {sum(hits)}/{len(hits)} = {sum(hits) / len(hits):.3f}")
    if full:
        print(f"  full session only: {sum(full)}/{len(full)} = {sum(full) / len(full):.3f}")


def print_reports(setting, paths, macro_only=False):
    is_ordinal = setting in ORDINAL_SETTINGS
    is_yesno = setting in YESNO_SETTINGS
    fieldnames = ORDINAL_FIELDNAMES if is_ordinal else YESNO_FIELDNAMES if is_yesno else CLASSIFICATION_FIELDNAMES

    all_rows = []
    for path in paths:
        rows = read_rows(path, fieldnames)
        if not rows:
            print(f"Warning: no rows found in {path}; skipping.")
            continue
        all_rows.append((path, rows))

    if not all_rows:
        print("No rows found in any input file.")
        return

    if is_ordinal:
        all_metrics = []
        for path, rows in all_rows:
            metrics = compute_ordinal_metrics(rows)
            all_metrics.append(metrics)
            print_ordinal_report(f"--- {path} ---", metrics)
            print()
        if len(all_metrics) > 1:
            print_ordinal_average_report(all_metrics)
    else:
        # Use one fixed label set (from real_answer across all runs) so every
        # run's metrics -- and the averages across runs -- line up on the same classes.
        labels = sorted({normalize_label(r["real_answer"]) for _, rows in all_rows for r in rows})
        all_metrics = []
        for path, rows in all_rows:
            metrics = compute_classification_metrics(rows, labels)
            all_metrics.append(metrics)
            print_classification_report(f"--- {path} ---", metrics, labels, macro_only=macro_only)
            if is_yesno:
                print_prefix_accuracy(rows)
            print()
        if len(all_metrics) > 1:
            print_classification_average_report(all_metrics, labels, macro_only=macro_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--setting", required=False, default=None,
        help="Which experiment's results to evaluate. For experiment2a's graph conditions use the folder "
             "name (experiment2a_video_graph, experiment2a_video_graph_ts, experiment2a_graph_only, "
             "experiment2a_graph_only_ts) or run_experiment2a.py's --condition spelling (video-graph, ...). "
             "Experiment 2b's graph conditions use the same folder names with experiment2b_ instead "
             "(experiment2b_video_graph, ...). For experiment1a/3/5's own graph/chat conditions use the "
             "prompt_maker_setting.py folder name directly (experiment1a_video_graph, experiment1b_chat_only, ...) "
             "-- these are still classification (ADOPT/MODIFY/REJECT-BYPASS), same as plain 1/3/4.",
    )
    parser.add_argument(
        "--model", required=False, default=None,
        help="Model slug to auto-discover result files for, e.g. gemini, gemini_flash, qwen, gpt, flash "
             "(must match the slug the runner script actually saved with -- see module docstring)",
    )
    parser.add_argument(
        "--input", required=False, nargs="+", default=None,
        help="Evaluate these specific file(s) directly instead of auto-discovering by --model",
    )
    parser.add_argument(
        "--macro-only", action="store_true",
        help="Classification settings only (1, 3, 5, and their experiment<ID>_... graph/chat "
             "conditions): print just macro precision/recall/f1 (and n), skipping the per-class table, "
             "accuracy, weighted avg, and confusion matrix. No effect on ordinal settings "
             "(experiment2a/experiment2b).",
    )
    parser.add_argument(
        "--update-reports", action="store_true",
        help="Scan all of experiment_results/ and write a new report/ file for every (setting, model) "
             "whose result CSVs are new or changed since the last run (tracked in report/.eval_state.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="With --update-reports: only list what would be written")
    parser.add_argument("--force", action="store_true", help="With --update-reports: regenerate every report")
    args = parser.parse_args()
    if args.update_reports:
        update_reports(args.macro_only, args.dry_run, args.force)
        return
    valid = [*CLASSIFICATION_EXPERIMENTS, *ORDINAL_SETTINGS, *YESNO_SETTINGS, *CONDITION_ALIASES, *SETTING_GRAPH_CHAT_CONDITIONS]
    if args.setting not in valid:
        parser.error("--setting is required (one of: 1, 3, 5, experiment2a, ...) unless --update-reports is given")
    args.setting = CONDITION_ALIASES.get(args.setting, args.setting)

    if args.input:
        paths = [Path(p) for p in args.input]
    else:
        if not args.model:
            raise SystemExit("Pass --model (to auto-discover result files) or --input (explicit file path(s)).")
        paths = discover_result_files(args.setting, args.model)
        if not paths:
            raise SystemExit(
                f"No result files matching '{args.model}_*results*.csv' found in {results_dir_for(args.setting)}"
            )
        print(f"Found {len(paths)} result file(s): {', '.join(p.name for p in paths)}\n")

    print_reports(args.setting, paths, args.macro_only)


if __name__ == "__main__":
    main()
