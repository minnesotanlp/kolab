# Kolab experiment code

Code that builds the Kolab experiments from the annotated workflows (labels, truncated video clips, experiment IDs and
setup CSVs, prompts), runs models on them, and evaluates the results.

## Experiments

| ID | Task | Clip / context |
|---|---|---|
| **1a** | Predict how the user responds to an AI action (adopt / modify / reject) | Source node to the end of the user's response |
| **1b** | Same task | Source node to the end of the AI action |
| **1c** | Same task, how much context helps | Ends at the AI action; starts at the last 1 / 2 / 3 steps, or the whole workflow (`1c_last1`, `1c_last2`, `1c_last3`, `1c_full`) |
| **1d** | Same task | Whole workflow, 0:00 to the target AI action |
| **2a** | Predict the collaboration rating (1-5) | One workflow step |
| **2b** | Predict the output-quality rating (1-5) | One workflow step |

Every experiment can be run with different inputs ("conditions"): `video-only`, `video-graph`, `graph-only`,
`video-chat`, `chat-only`, `graph-chat`, `video-graph-chat`.
Each experiment-condition pair has its own prompt folder and results folder, for example `experiment1a_chat_only`.

## Layout

```
code/
├── config.py                  # every path comes from here (see "Data" below)
├── experiments.py             # experiment IDs, conditions and folder naming
├── process_data_pipeline.py   # one-command prep for 1a / 1b: labels -> clips -> prompts
├── truncation/                # cut the source videos into clips
├── labels/                    # experiment 1 setup: label extraction (1a, 1b), contexts (1c), full workflow (1d)
├── experiment_setup/          # experiment IDs and setup CSVs for 2a / 2b, graph setup, rating audit
├── chat/                      # human <-> AI chat: anchors, alignment to the video clock, snippets
├── prompts/                   # prompt templates and prompt writer
├── inference/                 # run the models (Gemini, GPT, Qwen)
└── evaluation/                # metrics from the result CSVs
scripts/                       # shell wrappers that run the steps below in order
```

## Data

The code reads and writes one data folder. It is `<repo>/data` by default; set `KOLAB_DATA_DIR` to use another
location.

### Input data you need

None of the input data is part of this repository. Put these five inputs in the data folder (`S` = session 1 or 2,
`N` = participant number):

| Input | Where | Needed by |
|---|---|---|
| Annotated behavior graphs | `new_annotated_graphs/session<S>_p<N>_*.json` | everything |
| Source screen recordings | `session<S>_videos/real_videos/session<S>_p<N>.mp4` | clip cutting (all experiments with video) |
| Raw chat logs | `extracted_timestamps/session<S>_chats/p<N>_session<S>_chats.csv` | chat conditions |
| Workflow documents | `workflow_docs/P<N>_session<S>_workflow*.docx` | 2a and 2b |
| Participant list | `final_participants_for_iclr.csv` | prompts (occupation), 2a and 2b |

What the code expects in each:

- **Annotated behavior graphs.** One JSON per (participant, session): `session_id`, `participant_id`, `occupation`,
  `title`, `goal`, and `steps`. Each step has `step_name`, `step_action`, `time_range`, `start_seconds`, `end_seconds`,
  `nodes` and `edges`. Each node has `id`, `actor` (user or agent), `label`, `detail`, `time` (`["hh:mm:ss-hh:mm:ss"]`)
  and the software used. A user node can carry `user_response` entries
  (`source_node`, `target_ai`, `label` = adopt / modify / reject, `source_node_time`, `ai_node_time`,
  `user_adoption_time`, `reasoning`); these are the labels of experiments 1a to 1d. Sessions without a graph are
  skipped.
- **Source recordings.** One `.mp4` per session, named `session<S>_p<N>.mp4` (a `clean_session<S>_p<N>.mp4` is used
  if that is what exists). Timestamps in the graphs refer to this video's clock. Clips are cut from it with `ffmpeg`.
- **Raw chat logs.** One CSV per session with the columns `speaker` (Human / assistant), `timestamp` (ISO-8601, wall
  clock), `chat`, `ai_workflow`, `tool_use`, `AI_high_level_action`, `file`. `p<N>_session<S>_new_chats.csv` is
  preferred if both exist, and split files `p<N>_session<S>_chats_<i>.csv` are concatenated. `02_chat.sh` aligns the
  wall-clock times to the video using `chat_anchors.csv`, which it creates.
- **Workflow documents.** One Word file per (participant, session) with `Step <n> [Required]: ...` headings. Under
  each step, the 1-5 output-quality rating (experiment 2b) and, in the "Human-Agent Collaboration" section, the six 1-5
  collaboration ratings (experiment 2a), plus the written reasons. A step with no rating is left out.
- **Participant list.** A CSV whose first column is the participant ID (`P1`, `P2`, ...) and which has an
  `Occupation` column. The occupation fills the prompt's "a <profession> working with an AI agent" phrase, and the
  list decides which participants are included.

Experiments 1a to 1d need the graphs, recordings and participant list (plus chats for the chat conditions).
Experiments 2a and 2b additionally need the workflow documents.

> **Privacy.** The workflow documents and the participant list contain participant names and emails, and the chat logs,
> graphs and recordings contain what participants typed and showed on screen. Keep them out of the repository
> (the data folder is meant to live outside it, or be git-ignored) and check your consent terms before sharing any of it
> or of the derived prompts.

### What the code produces

| Path | What |
|---|---|
| `extracted_timestamps/labels_experiment<ID>/` | label CSVs (1a, 1b, 1c, 1d) |
| `extracted_timestamps/experiment2a/`, `experiment2b/` | setup CSVs, with the experiment IDs |
| `extracted_timestamps/truncated_videos/` | video clips |
| `extracted_timestamps/prompts/experiment<ID>[_<condition>]/` | the prompts, one `.txt` per experiment ID |
| `rating_audit/` | step validity and AI-usage table used to build 2a and 2b |
| `experiment_results/experiment<ID>[_<condition>]/` | model outputs, `<model>_<folder>_results[_run<N>].csv` |

`KOLAB_CLIP_DIR` sets where experiment 1c writes its clip folders (default `<data>/clips`); point it at a disk with
room.

## Requirements

- Python 3.8+ for everything except the Qwen runs, which need 3.9+ (the `transformers` release with Qwen3-VL needs it).
- `ffmpeg` and `ffprobe` on the PATH (clip cutting, frame extraction, clip checks).
- `pip install -r requirements.txt` for the core code: `tqdm`, `scikit-learn`, `python-docx`, `openai`.
- `pip install -r requirements-qwen.txt` for the local Qwen runs: adds `torch` (CUDA build), `torchvision`,
  `transformers>=4.57`, `accelerate` and `qwen-vl-utils`, and needs GPUs (the 30B-A3B and 32B models are sharded
  across the visible GPUs with `device_map="auto"`; weights are downloaded from Hugging Face on first use).
- `torch`, `transformers`, `qwen-vl-utils` and `openai` are imported only when a real model call is made, so
  `--dry-run` works without them. `tqdm`, `scikit-learn` and `python-docx` are imported at start-up by the scripts that
  use them.

Environment variables:

| Variable | Used by | Meaning |
|---|---|---|
| `AIGATEWAY_API_KEY` | Gemini, flash and GPT runs | key for the UMN AI gateway (`https://api.aigateway.umn.edu`, OpenAI-compatible) |
| `QWEN_BASE_URL`, `QWEN_MODEL`, `QWEN_API_KEY` | `inference/run_qwen32b_remote.py` | an OpenAI-compatible server hosting Qwen3-VL-32B (or `--base-url` etc.) |
| `KOLAB_DATA_DIR`, `KOLAB_CLIP_DIR` | all | data folder and experiment 1c clip folder (see "Data") |
| `PYTHON` | `scripts/*.sh` | interpreter to use (default `python3`) |

Models used: `gemini-3.1-pro-preview` and `gemini-3.8-flash` and GPT (`gpt-6-astra`) through the gateway, and
`Qwen/Qwen3-VL-30B-A3B-Instruct` and `Qwen/Qwen3-VL-32B-Instruct` locally. `run_qwen_setting.py --use_video_chat3` can also
load `MCG-NJU/VideoChat3-4B` (with `trust_remote_code`).

## Running the pipeline

Run the scripts in `scripts/` in order. Each accepts `--dry-run` (print the commands only) and `--help`.

```bash
export KOLAB_DATA_DIR=/path/to/data          # optional

scripts/01_prep.sh        # labels, clips and base prompts for 1a/1b; setup CSVs, clips and base prompts for 2a/2b
scripts/02_chat.sh        # chat anchors -> alignment -> per-step snippets
scripts/03_prompts.sh     # graph/chat ablation prompts, then experiments 1d and 1c
scripts/04_run.sh --models flash --exp1a video-only --dry-run    # preview a run
scripts/04_run.sh --models qwen --exp1a all --exp1b all --all --runs 1 2 3
scripts/05_eval.sh gemini qwen                                   # metrics for every experiment found
```

Details worth knowing:

- Every step skips work that already exists (clips, prompts, rows already in a results file), so an interrupted run can be
  started again.
- `04_run.sh` forwards its arguments to `inference/run_experiments.py`. Without `--all` it runs one sample per
  experiment and condition, as a smoke test. Flags are `--exp1a --exp1b --exp1c_last1 --exp1c_last2 --exp1c_last3
  --exp1c_full --exp1d --exp2a --exp2b`, each followed by the conditions to run (or `all`).
- `01_prep.sh` rewrites the 2a / 2b setup CSVs from scratch and keeps a `.bak` copy of the old ones.
- `02_chat.sh` stops with "nothing new to build" if no session needs work. The chat anchors CSV has a
  `video_first_mmss` column you can edit by hand; non-blank values are never overwritten.

Individual scripts can also be run directly from anywhere, for example
`python code/labels/build_experiment1d.py --dry-run`.

## Experiment IDs

Experiment 1 IDs look like `p6_s1_exp4` (participant, session, instance number). Experiment 2a and 2b IDs look like
`p1_session1_step2_basic_setup` (participant, session, step). The ID is the file name of the prompt and the key of each
row in the setup, label and results CSVs.
