"""
Build prediction prompts from the annotated behavior graphs, for experiment 1a,
experiment 1b, or experiment2a (--setting).

For every row in a "labels_experiment<N>/session<S>_p<N>_timestamp_labels.csv" file
(one row = one AI action the user later adopted/modified/rejected), this
script looks up:

  - the goal of the session               (annotated_graphs/*.json  "goal")
  - the participant's profession           (annotated_graphs/*.json  "occupation")
  - the AI action's own step description   (the "claude_node"/"ai_node" referenced
                                             by the CSV row's node's
                                             "user_response" entry, matched
                                             by its timestamp)
  - the user's action-step description     (the CSV row's own node "label";
                                             experiment 1a only -- experiment 1b's
                                             template doesn't show this)

and fills them into the setting's prompt template. Both settings' templates
show a "Target AI Action Timestamp" (re-zeroed to the truncated clip's own
timeline); experiment 1a's additionally shows "User Action Step", which experiment
1b's template omits, since experiment 1b's clips are already trimmed to end
exactly where the AI action ends.

Each row gets its own prompt file (a session can have many nodes/events),
named from its experiment_id (from the labels_experiment<N> CSV) as
<participant_id>_s<session>_exp<i>.txt:

    extracted_timestamps/prompts/experiment<ID>/<participant_id>_s<session>_exp<i>.txt
    (e.g. p6_s1_exp4.txt)

--setting experiment2a is a different (simpler) pipeline for Experiment 2a,
collaboration-rating prediction: one prompt per row of
extracted_timestamps/experiment2a/basic_experiment_setup.csv (one row = one
workflow step). For each row it looks up:

  - the session's goal and the step's name  (graph_clean/session<S>_p<N>_behavior_graph*.json
                                              "goal", and the step whose "badge" matches
                                              the row's step_id)
  - the participant's occupation             (final_participants_for_iclr.csv "Occupation" column)

and fills them into EXPERIMENT2A_TEMPLATE, saving to
extracted_timestamps/prompts/experiment2a/<experiment_id>.txt.

--setting experiment2b is the analogous pipeline for Experiment 2b,
output-quality-rating prediction: one prompt per row of
extracted_timestamps/experiment2b/basic_experiment_setup.csv (built by
build_experiment2b.py), using the same goal/step-name/occupation lookups as
experiment2a, but filling EXPERIMENT2B_TEMPLATE and saving to
extracted_timestamps/prompts/experiment2b/<experiment_id>.txt.

--setting experiment<ID>_<condition> (ID = 1a or 1b; condition = video_graph, graph_only, video_chat,
chat_only, graph_chat, video_graph_chat) builds the same prompts with the workflow graph and/or the
human<->AI chat added (and, for the *_only / graph_chat ones, no video is sent with the prompt).
Both are cut to the clip's window -- begin_timestamp..end_timestamp of the row's labels_experiment<N> CSV
line, i.e. from the start of the source node to the end of the user's response (experiment 1a) or of the
target AI action (experiment 1b):
  - graph: nodes that start in the window, edges between them (cut_window_subgraph), no timestamps;
  - chat:  rows of session<S>_chats_aligned/ whose video_ts_seconds is in the window
           (chat_common.chat_for_window); rows of sessions with no aligned chat are skipped.
Written to extracted_timestamps/prompts/experiment<ID>_<condition>/ with the same file names as experiment<ID>.
    python prompt_maker_setting.py --setting experiment1b_graph_only
    python prompt_maker_setting.py --setting experiment1a_video_graph_chat --preview exp1_p1_s1
        (--preview prints one prompt and writes nothing; --force overwrites existing files)

Usage:
    python prompt_maker_setting.py --setting 1a --p p1 --session 1
    python prompt_maker_setting.py --setting 1a --p p1        # both sessions for p1
    python prompt_maker_setting.py --setting 1a               # every participant/session found
    python prompt_maker_setting.py --setting 1b               # same, but for experiment 1b
    python prompt_maker_setting.py --setting experiment2a     # Experiment 2a prompts
    python prompt_maker_setting.py --setting experiment2a --p p2 --session 1
    python prompt_maker_setting.py --setting experiment2b     # Experiment 2b prompts

--setting experiment2a_video_graph / experiment2a_video_graph_ts /
experiment2a_graph_only / experiment2a_graph_only_ts build the Experiment 2a
graph-condition prompts for the steps in
extracted_timestamps/experiment2a/annotated_graph_experiment_setup.csv (made from
annotated_graphs/ by build_2a_graph_setup.py, which also writes the per-step
subgraphs in extracted_timestamps/2a_graphs/ that these read). "video_graph"
prompts go with the step's video clip, "graph_only" ones are sent with no video;
"_ts" puts each graph node's time range (on the step clip's clock, so multi-range
steps are offset) in the prompt, the others leave timestamps out. Each writes to
extracted_timestamps/prompts/<setting>/<experiment_id>.txt.
    python prompt_maker_setting.py --setting experiment2a_video_graph_ts
    python prompt_maker_setting.py --setting experiment2a_graph_only --preview p4_session1_step1_basic_setup
        (--preview prints one step's prompt and writes nothing; --force overwrites existing files)

The same four conditions exist for Experiment 2b (experiment2b_video_graph, _video_graph_ts, _graph_only,
_graph_only_ts): same steps and graphs, but the output-quality question, read from
extracted_timestamps/experiment2b/annotated_graph_experiment_setup.csv (build_2a_graph_setup.py
--experiment 2b) and written to prompts/experiment2b_<...>/.
    python prompt_maker_setting.py --setting experiment2b_video_graph_ts

Already-generated prompt files are left untouched (skipped).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

from chat.chat_common import chat_for_window
from labels.experiment1b_extract_user_response_labels import parse_user_response_entry as _parse_user_response_entry

from config import ROOT_DIR
# Graph sources (the first is synced from the annotators by bash_scripts/sync_graphs.sh). A session
# counts as long as it has a graph in either folder; new_annotated_graphs/ wins when both have one.
ANNOTATED_GRAPHS_DIR = ROOT_DIR / "new_annotated_graphs"
FALLBACK_GRAPHS_DIR = ROOT_DIR / "annotated_graphs"
EXPERIMENT2A_CSV = ROOT_DIR / "extracted_timestamps" / "experiment2a" / "basic_experiment_setup.csv"
WITH_GRAPH_CSV = ROOT_DIR / "extracted_timestamps" / "experiment2a" / "with_graph_experiment_setup.csv"
EXPERIMENT2B_CSV = ROOT_DIR / "extracted_timestamps" / "experiment2b" / "basic_experiment_setup.csv"
EXPERIMENT2C_CSV = ROOT_DIR / "extracted_timestamps" / "experiment2c" / "basic_experiment_setup.csv"
GRAPHS_2A_DIR = ROOT_DIR / "extracted_timestamps" / "2a_graphs"
# Built by build_2a_graph_setup.py from annotated_graphs/ -- the steps used by the
# experiment2a video-graph / graph-only conditions (GRAPH_CONDITIONS below).
ANNOTATED_SETUP_CSV = ROOT_DIR / "extracted_timestamps" / "experiment2a" / "annotated_graph_experiment_setup.csv"
# Same, for Experiment 2b (build_2a_graph_setup.py --experiment 2b).
ANNOTATED_SETUP_CSV_2B = ROOT_DIR / "extracted_timestamps" / "experiment2b" / "annotated_graph_experiment_setup.csv"
OCCUPATIONS_CSV = ROOT_DIR / "final_participants_for_iclr.csv"

CSV_NAME_RE = re.compile(r"^session(\d+)_p(\d+)_timestamp_labels\.csv$", re.IGNORECASE)

EXPERIMENT1A_TEMPLATE = """You are given a video snippet of {PROFESSION_PHRASE} working with an AI agent toward the following goal:

Goal: {GOAL}

The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an output in one step, and the user may act on that output either within the same step or in a later step.

AI Action Step: {AI_ACTION_STEP}
User Action Step: {USER_ACTION_STEP}

Target AI Action Timestamp: {START_TIME}–{END_TIME}

You must predict how the user will respond to the output of the target AI action:

* ADOPT: Use the AI output as provided, without any modification.
* MODIFY: Use the AI output but modify it in any way, including minor edits, formatting changes, or other adjustments.
* REJECT/BYPASS: Do not use the AI output; reject, ignore, replace, or proceed without it.

Use the workflow context available to make your prediction.

Output exactly one label:
ADOPT, MODIFY, or REJECT/BYPASS"""

EXPERIMENT1B_TEMPLATE = """You are given a video snippet of {PROFESSION_PHRASE} working with an AI agent toward the following goal:

Goal: {GOAL}

The workflow consists of multiple steps that contribute to achieving this goal.

Current Step: {AI_ACTION_STEP}

Target AI Action Timestamp: {START_TIME}–{END_TIME}

You must predict how the user will respond to the output of this AI action:

* ADOPT: Use the AI output as provided, without any modification.
* MODIFY: Use the AI output but modify it in any way, including minor edits, formatting changes, or other adjustments.
* REJECT/BYPASS: Do not use the AI output; reject, ignore, replace, or proceed without it.

Use the workflow context available to make your prediction.

Output exactly one label:
ADOPT, MODIFY, or REJECT/BYPASS"""

GRAPH_DESCRIPTION = (
    "each node is one action taken by the human or the agent; edges show the order they happened in"
)
CHAT_DESCRIPTION = (
    "in chronological order (each message from the human is wrapped in <human> tags, "
    "each message from the AI agent in <ai> tags)"
)


def build_setting_template(setting, video, graph, chat):
    """Prompt template of one experiment-1a / experiment-1b condition (video, graph and chat
    each on or off; at least one on). Same wording as EXPERIMENT1A_TEMPLATE /
    EXPERIMENT1B_TEMPLATE, plus a graph and/or chat section. The graph and chat cover only
    the span the video clip does (see cut_window_subgraph / chat_for_window), i.e.
    nothing after the user's response (experiment 1a) or the AI action's end (experiment 1b).
    Placeholders: PROFESSION_PHRASE, GOAL, AI_ACTION_STEP, [USER_ACTION_STEP (experiment 1a)],
    [START_TIME, END_TIME (video)], [GRAPH_TEXT (graph)], [CHAT_TEXT (chat)]."""
    sources = (["a video snippet"] if video else []) + (["the workflow graph"] if graph else []) \
        + (["the chat"] if chat else [])
    if len(sources) > 1:
        sources_text = ", ".join(sources[:-1]) + (", and " if len(sources) > 2 else " and ") + sources[-1]
    else:
        sources_text = sources[0]
    only_note = ""
    if not video:
        only_note = " (no video is provided for this task -- rely only on the {} below)".format(
            " and ".join(s.replace("the ", "") for s in sources)
        )

    parts = [f"You are given {sources_text} of {{PROFESSION_PHRASE}} working with an AI agent toward the following goal{only_note}:"]
    parts.append("Goal: {GOAL}")
    if setting == "1a":
        parts.append(
            "The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an "
            "output in one step, and the user may act on that output either within the same step or in a later step."
        )
        parts.append("AI Action Step: {AI_ACTION_STEP}\nUser Action Step: {USER_ACTION_STEP}")
        target = "the target AI action"
    else:
        parts.append("The workflow consists of multiple steps that contribute to achieving this goal.")
        parts.append("Current Step: {AI_ACTION_STEP}")
        target = "this AI action"
    if video:
        parts.append("Target AI Action Timestamp: {START_TIME}\u2013{END_TIME}")
    if graph:
        parts.append(
            f"Workflow graph for this portion of the workflow is defined as follows ({GRAPH_DESCRIPTION}):\n\n{{GRAPH_TEXT}}"
        )
    if chat:
        parts.append(
            f"Chat between the human and the AI agent during this portion of the workflow, {CHAT_DESCRIPTION}:\n\n{{CHAT_TEXT}}"
        )
    node_note = ' (the node above matching the "AI Action Step" described earlier)' if graph and not video else ""
    if graph and not video and setting == "1b":
        node_note = ' (the node above matching the "Current Step" described earlier)'
    parts.append(
        f"You must predict how the user will respond to the output of {target}{node_note}:\n\n"
        "* ADOPT: Use the AI output as provided, without any modification.\n"
        "* MODIFY: Use the AI output but modify it in any way, including minor edits, formatting changes, or other adjustments.\n"
        "* REJECT/BYPASS: Do not use the AI output; reject, ignore, replace, or proceed without it."
    )
    parts.append("Use the workflow context available to make your prediction.")
    parts.append("Output exactly one label:\nADOPT, MODIFY, or REJECT/BYPASS")
    return "\n\n".join(parts)


EXPERIMENT2A_TEMPLATE = """You are given a video snippet of {PROFESSION_PHRASE} working with an AI agent toward the following goal:

Goal: {GOAL}

The video shows a step of a multi-step workflow for accomplishing the goal:

Step: {STEP_NAME}

Watch how the human and the AI agent interact throughout this step. Examine whether the agent understood the task, context, and instructions; whether it adapted appropriately when given feedback or corrections; whether it reduced the human's effort, time, or cognitive load; whether it proactively suggested useful ideas or next steps; whether its output was accurate, relevant, and trustworthy with minimal correction needed; whether the interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work; and whether the human was able to guide, redirect, or override the agent's actions when desired.

Based on this, rate the overall collaboration experience for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the collaboration experience with the AI agent and 5 means that the user is satisfied with the collaboration experience with the AI agent.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2A_GRAPH_TEMPLATE = """The video and graph show actions by {PROFESSION_PHRASE} and AI to complete the step below for achieving the goal.

Goal: {GOAL}

Step: {STEP_NAME}

Workflow graph for this step is defined as follows ({NODE_DESCRIPTION}):

{GRAPH_TEXT}

Watch the video and read the graph above together. Examine whether the agent understood the task, context, and instructions; whether it adapted appropriately when given feedback or corrections; whether it reduced the human's effort, time, or cognitive load; whether it proactively suggested useful ideas or next steps; whether its output was accurate, relevant, and trustworthy with minimal correction needed; whether the interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work; and whether the human was able to guide, redirect, or override the agent's actions when desired.

Based on this, rate the overall collaboration experience for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the collaboration experience with the AI agent and 5 means that the user is satisfied with the collaboration experience with the AI agent.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2A_GRAPH_ONLY_TEMPLATE = """You are given the workflow graph of {PROFESSION_PHRASE} working with an AI agent toward the following goal (no video is provided for this task -- rely only on the graph below):

Goal: {GOAL}

Step: {STEP_NAME}

Workflow graph for this step is defined as follows ({NODE_DESCRIPTION}):

{GRAPH_TEXT}

Read the graph above. Examine whether the agent understood the task, context, and instructions; whether it adapted appropriately when given feedback or corrections; whether it reduced the human's effort, time, or cognitive load; whether it proactively suggested useful ideas or next steps; whether its output was accurate, relevant, and trustworthy with minimal correction needed; whether the interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work; and whether the human was able to guide, redirect, or override the agent's actions when desired.

Based on this, rate the overall collaboration experience for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the collaboration experience with the AI agent and 5 means that the user is satisfied with the collaboration experience with the AI agent.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2A_GRAPH_CHAT_TEMPLATE = """The video, graph, and chat show actions by {PROFESSION_PHRASE} and AI to complete the step below for achieving the goal.

Goal: {GOAL}

Step: {STEP_NAME}

Workflow graph for this step is defined as follows ({NODE_DESCRIPTION}):

{GRAPH_TEXT}

Chat between the human and the AI agent during this step, in chronological order (each message from the human is wrapped in <human> tags, each message from the AI agent in <ai> tags):

{CHAT_TEXT}

Watch the video and read the graph and the chat above together. Examine whether the agent understood the task, context, and instructions; whether it adapted appropriately when given feedback or corrections; whether it reduced the human's effort, time, or cognitive load; whether it proactively suggested useful ideas or next steps; whether its output was accurate, relevant, and trustworthy with minimal correction needed; whether the interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work; and whether the human was able to guide, redirect, or override the agent's actions when desired.

Based on this, rate the overall collaboration experience for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the collaboration experience with the AI agent and 5 means that the user is satisfied with the collaboration experience with the AI agent.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2A_GRAPH_CHAT_ONLY_TEMPLATE = """You are given the workflow graph and the chat of {PROFESSION_PHRASE} working with an AI agent toward the following goal (no video is provided for this task -- rely only on the graph and the chat below):

Goal: {GOAL}

Step: {STEP_NAME}

Workflow graph for this step is defined as follows ({NODE_DESCRIPTION}):

{GRAPH_TEXT}

Chat between the human and the AI agent during this step, in chronological order (each message from the human is wrapped in <human> tags, each message from the AI agent in <ai> tags):

{CHAT_TEXT}

Read the graph and the chat above together. Examine whether the agent understood the task, context, and instructions; whether it adapted appropriately when given feedback or corrections; whether it reduced the human's effort, time, or cognitive load; whether it proactively suggested useful ideas or next steps; whether its output was accurate, relevant, and trustworthy with minimal correction needed; whether the interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work; and whether the human was able to guide, redirect, or override the agent's actions when desired.

Based on this, rate the overall collaboration experience for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the collaboration experience with the AI agent and 5 means that the user is satisfied with the collaboration experience with the AI agent.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2A_CHAT_TEMPLATE = """You are given a video snippet of {PROFESSION_PHRASE} working with an AI agent toward the following goal, together with the chat between them:

Goal: {GOAL}

The video shows a step of a multi-step workflow for accomplishing the goal:

Step: {STEP_NAME}

Chat between the human and the AI agent during this step, in chronological order (each message from the human is wrapped in <human> tags, each message from the AI agent in <ai> tags):

{CHAT_TEXT}

Watch how the human and the AI agent interact throughout this step and read the chat above. Examine whether the agent understood the task, context, and instructions; whether it adapted appropriately when given feedback or corrections; whether it reduced the human's effort, time, or cognitive load; whether it proactively suggested useful ideas or next steps; whether its output was accurate, relevant, and trustworthy with minimal correction needed; whether the interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work; and whether the human was able to guide, redirect, or override the agent's actions when desired.

Based on this, rate the overall collaboration experience for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the collaboration experience with the AI agent and 5 means that the user is satisfied with the collaboration experience with the AI agent.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2A_CHAT_ONLY_TEMPLATE = """You are given the chat between {PROFESSION_PHRASE} and an AI agent while they worked toward the following goal (no video is provided for this task -- rely only on the chat below):

Goal: {GOAL}

Step: {STEP_NAME}

Chat between the human and the AI agent during this step, in chronological order (each message from the human is wrapped in <human> tags, each message from the AI agent in <ai> tags):

{CHAT_TEXT}

Read the chat above. Examine whether the agent understood the task, context, and instructions; whether it adapted appropriately when given feedback or corrections; whether it reduced the human's effort, time, or cognitive load; whether it proactively suggested useful ideas or next steps; whether its output was accurate, relevant, and trustworthy with minimal correction needed; whether the interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work; and whether the human was able to guide, redirect, or override the agent's actions when desired.

Based on this, rate the overall collaboration experience for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the collaboration experience with the AI agent and 5 means that the user is satisfied with the collaboration experience with the AI agent.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2B_TEMPLATE = """You are given a video snippet of {PROFESSION_PHRASE} working with an AI agent toward the following goal:

Goal: {GOAL}

The video shows a step of a multi-step workflow for accomplishing the goal:

Step: {STEP_NAME}

Watch the output the AI agent produced and how it was used during this step. Examine whether the output was accurate, relevant, and directly usable for the task; whether it required significant correction, rework, or was discarded; and whether it met the specific requirements of this step.

Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the output quality and 5 means that the user is satisfied with the output quality.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2B_GRAPH_TEMPLATE = """The video and graph show actions by {PROFESSION_PHRASE} and AI to complete the step below for achieving the goal.

Goal: {GOAL}

Step: {STEP_NAME}

Workflow graph for this step is defined as follows ({NODE_DESCRIPTION}):

{GRAPH_TEXT}

Watch the video and read the graph above together, focusing on the output the AI agent produced and how it was used during this step. Examine whether the output was accurate, relevant, and directly usable for the task; whether it required significant correction, rework, or was discarded; and whether it met the specific requirements of this step.

Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the output quality and 5 means that the user is satisfied with the output quality.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2B_GRAPH_ONLY_TEMPLATE = """You are given the workflow graph of {PROFESSION_PHRASE} working with an AI agent toward the following goal (no video is provided for this task -- rely only on the graph below):

Goal: {GOAL}

Step: {STEP_NAME}

Workflow graph for this step is defined as follows ({NODE_DESCRIPTION}):

{GRAPH_TEXT}

Read the graph above, focusing on the output the AI agent produced and how it was used during this step. Examine whether the output was accurate, relevant, and directly usable for the task; whether it required significant correction, rework, or was discarded; and whether it met the specific requirements of this step.

Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the output quality and 5 means that the user is satisfied with the output quality.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2B_GRAPH_CHAT_TEMPLATE = """The video, graph, and chat show actions by {PROFESSION_PHRASE} and AI to complete the step below for achieving the goal.

Goal: {GOAL}

Step: {STEP_NAME}

Workflow graph for this step is defined as follows ({NODE_DESCRIPTION}):

{GRAPH_TEXT}

Chat between the human and the AI agent during this step, in chronological order (each message from the human is wrapped in <human> tags, each message from the AI agent in <ai> tags):

{CHAT_TEXT}

Watch the video and read the graph and the chat above together, focusing on the output the AI agent produced and how it was used during this step. Examine whether the output was accurate, relevant, and directly usable for the task; whether it required significant correction, rework, or was discarded; and whether it met the specific requirements of this step.

Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the output quality and 5 means that the user is satisfied with the output quality.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2B_CHAT_TEMPLATE = """You are given a video snippet of {PROFESSION_PHRASE} working with an AI agent toward the following goal, together with the chat between them:

Goal: {GOAL}

The video shows a step of a multi-step workflow for accomplishing the goal:

Step: {STEP_NAME}

Chat between the human and the AI agent during this step, in chronological order (each message from the human is wrapped in <human> tags, each message from the AI agent in <ai> tags):

{CHAT_TEXT}

Watch the output the AI agent produced and how it was used during this step, and read the chat above. Examine whether the output was accurate, relevant, and directly usable for the task; whether it required significant correction, rework, or was discarded; and whether it met the specific requirements of this step.

Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the output quality and 5 means that the user is satisfied with the output quality.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2B_GRAPH_CHAT_ONLY_TEMPLATE = """You are given the workflow graph and the chat of {PROFESSION_PHRASE} working with an AI agent toward the following goal (no video is provided for this task -- rely only on the graph and the chat below):

Goal: {GOAL}

Step: {STEP_NAME}

Workflow graph for this step is defined as follows ({NODE_DESCRIPTION}):

{GRAPH_TEXT}

Chat between the human and the AI agent during this step, in chronological order (each message from the human is wrapped in <human> tags, each message from the AI agent in <ai> tags):

{CHAT_TEXT}

Read the graph and the chat above together, focusing on the output the AI agent produced and how it was used during this step. Examine whether the output was accurate, relevant, and directly usable for the task; whether it required significant correction, rework, or was discarded; and whether it met the specific requirements of this step.

Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the output quality and 5 means that the user is satisfied with the output quality.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2B_CHAT_ONLY_TEMPLATE = """You are given the chat between {PROFESSION_PHRASE} and an AI agent while they worked toward the following goal (no video is provided for this task -- rely only on the chat below):

Goal: {GOAL}

Step: {STEP_NAME}

Chat between the human and the AI agent during this step, in chronological order (each message from the human is wrapped in <human> tags, each message from the AI agent in <ai> tags):

{CHAT_TEXT}

Read the chat above, focusing on the output the AI agent produced and how it was used during this step. Examine whether the output was accurate, relevant, and directly usable for the task; whether it required significant correction, rework, or was discarded; and whether it met the specific requirements of this step.

Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is not satisfied with the output quality and 5 means that the user is satisfied with the output quality.

Output exactly one integer, with no other text:
1, 2, 3, 4, or 5"""

EXPERIMENT2C_TEMPLATE = """You are given a video snippet of {PROFESSION_PHRASE} working with an AI agent toward the following goal:

Goal: {GOAL}

The video shows the session from its beginning up to a certain point. The full session is about one hour of the human working with the AI agent, and this video may show only the first part of it.

Watch how the human and the AI agent interact over the course of the video. Examine whether the agent understood the task, context, and instructions; whether the human had to spend significant time getting the agent up to speed or correcting it; how much of the intended work has been completed so far and how much progress is being made toward the goal; and whether the AI's assistance is helping or slowing down progress.

Based on this, predict whether the user achieved their goal after the ~1-hour interaction with the AI agent.

Output exactly one label, with no other text:
Yes or No"""

TEMPLATES = {
    "1a": EXPERIMENT1A_TEMPLATE, "1b": EXPERIMENT1B_TEMPLATE,
    "experiment2a": EXPERIMENT2A_TEMPLATE, "experiment2a_graph": EXPERIMENT2A_GRAPH_TEMPLATE,
    "experiment2a_graph_only": EXPERIMENT2A_GRAPH_ONLY_TEMPLATE,
    "experiment2a_graph_chat": EXPERIMENT2A_GRAPH_CHAT_TEMPLATE,
    "experiment2a_graph_chat_only": EXPERIMENT2A_GRAPH_CHAT_ONLY_TEMPLATE,
    "experiment2a_chat": EXPERIMENT2A_CHAT_TEMPLATE,
    "experiment2a_chat_only": EXPERIMENT2A_CHAT_ONLY_TEMPLATE,
    "experiment2b_graph_chat": EXPERIMENT2B_GRAPH_CHAT_TEMPLATE,
    "experiment2b_chat": EXPERIMENT2B_CHAT_TEMPLATE,
    "experiment2b_graph_chat_only": EXPERIMENT2B_GRAPH_CHAT_ONLY_TEMPLATE,
    "experiment2b_chat_only": EXPERIMENT2B_CHAT_ONLY_TEMPLATE,
    "experiment2b": EXPERIMENT2B_TEMPLATE,
    "experiment2b_graph": EXPERIMENT2B_GRAPH_TEMPLATE,
    "experiment2b_graph_only": EXPERIMENT2B_GRAPH_ONLY_TEMPLATE,
    "experiment2c": EXPERIMENT2C_TEMPLATE,
}

# Experiment 2a / 2b graph conditions built from annotated_graphs/ (see
# build_2a_graph_setup.py). The key is both the --setting value and the folder
# under extracted_timestamps/prompts/ the prompts are written to.
#   csv        -- the annotated-steps setup CSV of that experiment (2a: ANNOTATED_SETUP_CSV,
#                 2b: ANNOTATED_SETUP_CSV_2B)
#   template   -- key into TEMPLATES (video is only mentioned by the video-graph one)
#   timestamps -- whether each graph node line carries its (clip-relative) time range
GRAPH_CONDITIONS = {
    "experiment2a_video_graph":    {"csv": ANNOTATED_SETUP_CSV, "template": "experiment2a_graph",      "timestamps": False},
    "experiment2a_video_graph_ts": {"csv": ANNOTATED_SETUP_CSV, "template": "experiment2a_graph",      "timestamps": True},
    "experiment2a_graph_only":     {"csv": ANNOTATED_SETUP_CSV, "template": "experiment2a_graph_only", "timestamps": False},
    "experiment2a_graph_only_ts":  {"csv": ANNOTATED_SETUP_CSV, "template": "experiment2a_graph_only", "timestamps": True},
    "experiment2b_video_graph":    {"csv": ANNOTATED_SETUP_CSV_2B, "template": "experiment2b_graph",      "timestamps": False},
    "experiment2b_video_graph_ts": {"csv": ANNOTATED_SETUP_CSV_2B, "template": "experiment2b_graph",      "timestamps": True},
    "experiment2b_graph_only":     {"csv": ANNOTATED_SETUP_CSV_2B, "template": "experiment2b_graph_only", "timestamps": False},
    "experiment2b_graph_only_ts":  {"csv": ANNOTATED_SETUP_CSV_2B, "template": "experiment2b_graph_only", "timestamps": True},
}


# Experiment 2a chat ablation (chat_snippets.csv, built by run_chat_pipeline.sh). The key is
# both the --setting value and the folder under extracted_timestamps/prompts/.
#   csv      -- the annotated-steps setup CSV of that experiment (2a or 2b; same experiment_ids)
#   template -- key into TEMPLATES
#   graph    -- whether the step's graph is embedded too (needs graph_filename in the setup CSV)
CHAT_SNIPPETS_CSV = ROOT_DIR / "extracted_timestamps" / "experiment2a" / "chat_snippets.csv"
CHAT_CONDITIONS = {
    "experiment2a_video_graph_chat": {"csv": ANNOTATED_SETUP_CSV, "template": "experiment2a_graph_chat",      "graph": True},
    "experiment2a_video_chat":       {"csv": ANNOTATED_SETUP_CSV, "template": "experiment2a_chat",            "graph": False},
    "experiment2a_graph_chat":       {"csv": ANNOTATED_SETUP_CSV, "template": "experiment2a_graph_chat_only", "graph": True},
    "experiment2a_chat_only":        {"csv": ANNOTATED_SETUP_CSV, "template": "experiment2a_chat_only",       "graph": False},
    # Experiment 2b: same steps/ids/graphs/chat snippets, output-quality wording.
    "experiment2b_video_graph_chat": {"csv": ANNOTATED_SETUP_CSV_2B, "template": "experiment2b_graph_chat",      "graph": True},
    "experiment2b_video_chat":       {"csv": ANNOTATED_SETUP_CSV_2B, "template": "experiment2b_chat",            "graph": False},
    "experiment2b_graph_chat":       {"csv": ANNOTATED_SETUP_CSV_2B, "template": "experiment2b_graph_chat_only", "graph": True},
    "experiment2b_chat_only":        {"csv": ANNOTATED_SETUP_CSV_2B, "template": "experiment2b_chat_only",       "graph": False},
}
NO_CHAT_TEXT = "(There were no chat messages during this step.)"


# Experiment 1a / 1b ablation conditions. The key is both the --setting value
# and the folder under extracted_timestamps/prompts/; plain "1a" / "1b" (folders experiment1a / experiment1b)
# stay the video-only condition. Graph and chat are cut to the clip's window
# (begin_timestamp..end_timestamp of the labels_experiment<N> CSV row).
#   setting -- "1a" or "1b"
#   video / graph / chat -- which sources the prompt (and, for video, the model input) carries
SETTING_CONDITIONS = {}
for _n in ("1a", "1b"):
    for _name, _video, _graph, _chat in (
        ("video_graph",      True,  True,  False),
        ("graph_only",       False, True,  False),
        ("video_chat",       True,  False, True),
        ("chat_only",        False, False, True),
        ("graph_chat",       False, True,  True),
        ("video_graph_chat", True,  True,  True),
    ):
        _key = f"experiment{_n}_{_name}"
        SETTING_CONDITIONS[_key] = {"setting": _n, "video": _video, "graph": _graph, "chat": _chat}
        TEMPLATES[_key] = build_setting_template(_n, _video, _graph, _chat)
NO_CHAT_TEXT_SETTING = "(There were no chat messages during this portion of the workflow.)"


def node_description(with_timestamps):
    """Fills the {NODE_DESCRIPTION} placeholder of the experiment2a graph templates."""
    return (
        "each node is one action taken by the human or the agent"
        + (", with its timestamp range" if with_timestamps else "")
        + "; edges show the order they happened in"
    )


def setting_csv_dir(setting):
    return ROOT_DIR / "extracted_timestamps" / f"labels_experiment{setting}"


def setting_output_dir(setting):
    folder = setting if setting.startswith(("experiment2a", "experiment2b", "experiment2c", "experiment")) else f"experiment{setting}"
    return ROOT_DIR / "extracted_timestamps" / "prompts" / folder


# ---------------------------------------------------------------------------
# Behavior graph helpers
# ---------------------------------------------------------------------------

def split_time_range(range_str):
    for sep in ("–", "—", "-"):
        if sep in range_str:
            start, _, end = range_str.partition(sep)
            if start.strip() and end.strip():
                return start.strip(), end.strip()
    raise ValueError(f"Cannot split time range: {range_str!r}")


def to_seconds(ts):
    parts = [int(p) for p in ts.strip().split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def to_timestamp(seconds):
    seconds = max(int(round(seconds)), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_clip_ranges(timestamp_ranges):
    """Parse a setup-CSV "timestamp_ranges" cell ("HH:MM:SS-HH:MM:SS" pieces,
    ";"-separated when a step's footage is disjoint) into [(start_s, end_s), ...]
    in the original order. The step's clip is these pieces joined end to end."""
    ranges = []
    for part in timestamp_ranges.split(";"):
        part = part.strip()
        if not part:
            continue
        start, end = split_time_range(part)
        ranges.append((to_seconds(start), to_seconds(end)))
    return ranges


def session_range_to_clip_ranges(start_s, end_s, clip_ranges):
    """Map a [start_s, end_s] interval on the full session video's clock onto the
    step clip's clock. clip_ranges is parse_clip_ranges()'s output: piece i of the
    clip starts at the sum of the lengths of pieces 0..i-1, and shows session time
    clip_ranges[i][0]..clip_ranges[i][1].

    Returns [(clip_start_s, clip_end_s), ...] -- one entry per piece the interval
    overlaps (the part outside any piece is clamped away). An empty list means the
    interval lies entirely in a gap between pieces, i.e. is not in the clip at all.

    A zero-length interval (start_s == end_s, an instantaneous node) maps to the
    single instant it falls on, in the first piece that contains it."""
    mapped = []
    offset = 0
    if start_s == end_s:
        for piece_start, piece_end in clip_ranges:
            if piece_start <= start_s <= piece_end:
                return [(offset + start_s - piece_start, offset + start_s - piece_start)]
            offset += piece_end - piece_start
        return []
    for piece_start, piece_end in clip_ranges:
        overlap_start = max(start_s, piece_start)
        overlap_end = min(end_s, piece_end)
        if overlap_start < overlap_end:
            mapped.append((offset + overlap_start - piece_start, offset + overlap_end - piece_start))
        offset += piece_end - piece_start
    return mapped


def node_time_bounds(node):
    """Return (start_seconds, end_seconds) spanning all of a node's time ranges."""
    time_field = node.get("time")
    if not time_field:
        raise ValueError(f"Node {node.get('id')!r} has no 'time' field")
    ranges = [time_field] if isinstance(time_field, str) else list(time_field)
    start, _ = split_time_range(ranges[0])
    _, end = split_time_range(ranges[-1])
    return to_seconds(start), to_seconds(end)


def parse_user_response_entry(entry, node_id, node_index=None):
    """Return (source_node_id, ai_node_id, user_action, ai_ts_override) or None (and print
    a warning) if the dict entry can't be used. Delegates to
    experiment1b_extract_user_response_labels.parse_user_response_entry, so an entry is skipped
    here exactly when it is skipped when the labels_experiment<N> labels are built.

    ai_ts_override is the entry's "ai_node_time" as "HH:MM:SS-HH:MM:SS" (a multi-range runs
    from its first start to its last end), or None when it is empty -- then ai_node's own
    aggregate first/last "time" range is used. It identifies which specific occurrence of
    ai_node (e.g. after a loop back to an earlier node) this response refers to."""
    parsed = _parse_user_response_entry(entry, node_id, node_index)
    if parsed is None:
        return None
    source_node_id, ai_node_id, user_action, _source_ts, ai_ts, _adoption_ts = parsed
    return source_node_id, ai_node_id, user_action, ai_ts


def build_node_index(graph):
    """Map node id -> (node, step_name, step_number) by searching the whole graph.
    step_number comes from the segment's "badge" field (e.g. "Step 3" -> 3),
    falling back to the segment's position in the list if badge has no digits."""
    index = {}
    for position, step in enumerate(graph.get("steps", []), start=1):
        step_name = step.get("step_name", "")
        badge_match = re.search(r"\d+", step.get("badge", ""))
        step_number = int(badge_match.group(0)) if badge_match else position
        for node in step.get("nodes", []):
            index[node["id"]] = (node, step_name, step_number)
    return index


def find_ai_action_for_row(node_index, node_id, target_ai_timestamp):
    """
    Given the CSV row's node_id and its target_ai_timestamp ("HH:MM:SS-HH:MM:SS"),
    find which "user_response" entry on that node points at the matching AI node.

    A single node can carry several user_response entries (several AI actions
    feeding into one user response), so the timestamp is what disambiguates
    which one this CSV row refers to.

    Returns (ai_node_id, ai_action_step, user_action_step, step_name).
    """
    entry = node_index.get(node_id)
    if entry is None:
        raise KeyError(f"node id {node_id!r} not found in behavior graph")
    node, step_name, _step_number = entry

    target_start, target_end = split_time_range(target_ai_timestamp)
    target_bounds = (to_seconds(target_start), to_seconds(target_end))

    responses = node.get("user_response", [])
    plain_node_index = {nid: e[0] for nid, e in node_index.items()}
    best_match = None
    best_diff = None
    for response in responses:
        parsed = parse_user_response_entry(response, node_id, plain_node_index)
        if parsed is None:
            continue
        _source_id, ai_node_id, _label, ai_ts_override = parsed
        ai_entry = node_index.get(ai_node_id)
        if ai_entry is None:
            continue
        ai_node, _, _ = ai_entry
        try:
            if ai_ts_override is not None:
                ai_start, ai_end = split_time_range(ai_ts_override)
                ai_bounds = (to_seconds(ai_start), to_seconds(ai_end))
            else:
                ai_bounds = node_time_bounds(ai_node)
        except ValueError:
            continue
        if ai_bounds == target_bounds:
            return ai_node_id, ai_node["label"], node["label"], step_name
        diff = abs(ai_bounds[0] - target_bounds[0]) + abs(ai_bounds[1] - target_bounds[1])
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_match = (ai_node_id, ai_node["label"])

    if best_match is not None:
        print(
            f"Warning: no exact timestamp match for node {node_id!r} / "
            f"{target_ai_timestamp!r}; using closest user_response entry instead.",
            file=sys.stderr,
        )
        ai_node_id, ai_action_step = best_match
        return ai_node_id, ai_action_step, node["label"], step_name

    raise ValueError(
        f"Node {node_id!r} has no user_response entry whose AI node matches "
        f"timestamp {target_ai_timestamp!r}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prompt_filename(experiment_id, participant_id, session):
    """Prompt filename for one CSV row: <participant_id>_s<session>_exp<i>.txt
    (e.g. p6_s1_exp4.txt), reordering experiment_id's "exp<i>_p<N>_s<S>" parts."""
    match = re.match(r"^(exp\d+)_", experiment_id)
    exp_part = match.group(1) if match else experiment_id
    return f"{participant_id}_s{session}_{exp_part}.txt"


def discover_jobs(participant_filter, session_filter, setting="1a"):
    """Return a list of (csv_path, session, participant_id) to process."""
    jobs = []
    for csv_path in sorted(setting_csv_dir(setting).glob("*.csv")):
        match = CSV_NAME_RE.match(csv_path.name)
        if not match:
            continue
        session = int(match.group(1))
        participant_id = f"p{match.group(2)}"

        if participant_filter and participant_id != participant_filter:
            continue
        if session_filter and session != session_filter:
            continue

        jobs.append((csv_path, session, participant_id))
    return jobs


def load_graph(session, participant_id):
    """Find and load session<S>_<p>_*.json -- the suffix isn't consistent across files
    ("_behavior_graph.json", "_behavior_graph_modified.json",
    "_behavior_graph_modified_check.json", "_edited.json"), so match by the
    session_participant prefix only."""
    matches = sorted(ANNOTATED_GRAPHS_DIR.glob(f"session{session}_{participant_id}_*.json"))
    if not matches:  # no graph in new_annotated_graphs/: fall back to the older annotated_graphs/
        matches = sorted(FALLBACK_GRAPHS_DIR.glob(f"session{session}_{participant_id}_*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No 'session{session}_{participant_id}_*.json' found in {ANNOTATED_GRAPHS_DIR} "
            f"or {FALLBACK_GRAPHS_DIR}"
        )
    if len(matches) > 1:
        print(
            f"Warning: multiple behavior graph files match session{session}_{participant_id}; "
            f"using {matches[0].name!r}: {[m.name for m in matches]}",
            file=sys.stderr,
        )
    with open(matches[0], "r") as f:
        return json.load(f)


def collect_tasks(jobs):
    """
    Read every CSV row for the given jobs and return a flat list of tasks:
    (participant_id, session, graph, node_index, node_id, target_ai_timestamp,
    clip_begin_timestamp, clip_end_timestamp, experiment_id).
    Loads each graph only once even though it may have many rows.
    """
    tasks = []
    for csv_path, session, participant_id in jobs:
        try:
            graph = load_graph(session, participant_id)
        except FileNotFoundError:
            print(
                f"Warning: no graph for {participant_id} session {session} in "
                f"{ANNOTATED_GRAPHS_DIR.name}/; skipping its rows (existing prompts are kept).",
                file=sys.stderr,
            )
            continue
        node_index = build_node_index(graph)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tasks.append(
                    (
                        participant_id,
                        session,
                        graph,
                        node_index,
                        row["user_action_node"],
                        row["target_ai_timestamp"],
                        row["begin_timestamp"],
                        row["end_timestamp"],
                        row["experiment_id"],
                    )
                )
    return tasks


def cut_window_subgraph(graph, begin_s, end_s):
    """The part of an annotated behavior graph inside one clip window, as a list of
    step dicts for render_graph_text() (one per step that has a node in the window;
    more than one when the window crosses steps, which then get "[Step: ...]" headers).

    A node is kept if any of its time ranges starts in [begin_s, end_s) -- so a node
    that starts exactly where the window ends (experiment 1b: the user's response node,
    which starts right as the AI action ends) is out, and looped/revisited nodes with
    several ranges count if any visit is in the window. An edge is kept only if both
    of its ends are. Only id/label/detail/actor fields are carried over: the nodes'
    "user_response" / "source_steps" entries hold the adopt/modify/reject labels."""
    steps = []
    for step in graph.get("steps", []):
        kept_nodes = []
        for node in step.get("nodes", []):
            time_field = node.get("time")
            if not time_field:
                continue
            ranges = [time_field] if isinstance(time_field, str) else list(time_field)
            starts = []
            for r in ranges:
                try:
                    starts.append(to_seconds(split_time_range(r)[0]))
                except ValueError:
                    continue
            if any(begin_s <= t < end_s for t in starts):
                kept_nodes.append({k: v for k, v in node.items() if k not in ("user_response", "source_steps")})
        if not kept_nodes:
            continue
        kept_ids = {n["id"] for n in kept_nodes}
        steps.append(
            {
                "step_name": step.get("step_name", ""),
                "badge": step.get("badge"),
                "nodes": kept_nodes,
                "edges": [e for e in step.get("edges", []) if e.get("from") in kept_ids and e.get("to") in kept_ids],
            }
        )
    return steps


def build_prompt(graph, node_index, node_id, target_ai_timestamp, clip_begin_timestamp, clip_end_timestamp,
                 participant_id, session, setting, occupations, condition=None):
    """Return (ai_node_id, prompt_text) for one CSV row, or None if it can't be resolved.

    condition=None is the video-only prompt (EXPERIMENT1A_TEMPLATE / EXPERIMENT1B_TEMPLATE). Otherwise it
    is a SETTING_CONDITIONS key and the prompt also (or instead) carries:
      - the workflow graph, cut to the clip window [clip_begin_timestamp, clip_end_timestamp] by
        cut_window_subgraph (no timestamps in the graph text);
      - the human<->AI chat inside the same window, from the session's aligned chat file
        (chat_for_window). If the session has no aligned chat, the row is skipped with a warning.
    The window is the truncated clip's span, so nothing after the user's response (experiment 1a) or
    the AI action's end (experiment 1b) reaches any condition."""
    try:
        ai_node_id, ai_action_step, user_action_step, _step_name = find_ai_action_for_row(
            node_index, node_id, target_ai_timestamp
        )
    except (KeyError, ValueError) as exc:
        print(
            f"Warning: skipping node {node_id!r} for {participant_id} session {session}: {exc}",
            file=sys.stderr,
        )
        return None

    goal = graph.get("goal", "")
    # The behavior graph doesn't always carry its own "occupation" field (some
    # freshly-added/re-annotated graphs don't); fall back to
    # final_participants_for_iclr.csv, the same source experiment2a prompts use.
    profession = graph.get("occupation") or occupations.get(participant_id.strip().lower())
    if not profession:
        print(
            f"Warning: no occupation found in the behavior graph or "
            f"{OCCUPATIONS_CSV.name} for {participant_id} session {session}; leaving placeholder.",
            file=sys.stderr,
        )
        profession = "professional"
    article = "an" if profession[:1].lower() in "aeiou" else "a"

    # target_ai_timestamp is measured against the FULL original video, but the
    # video snippet actually shown/sent for this row is the truncated clip
    # extracted_timestamps/truncated_videos/experiment<ID>/..._<begin>_<end>.mp4,
    # whose own timeline starts at 0:00 = clip_begin_timestamp. Re-zero the
    # target timestamp to the clip so it matches what's visible in that clip.
    target_start, target_end = split_time_range(target_ai_timestamp)
    clip_begin_seconds = to_seconds(clip_begin_timestamp)
    start_time = to_timestamp(to_seconds(target_start) - clip_begin_seconds)
    end_time = to_timestamp(to_seconds(target_end) - clip_begin_seconds)

    fields = dict(
        PROFESSION_PHRASE=f"{article} {profession}",
        GOAL=goal,
        AI_ACTION_STEP=ai_action_step,
        START_TIME=start_time,
        END_TIME=end_time,
    )
    # experiment 1b's clip is already trimmed to end exactly where the AI action
    # ends, so no user-action line is shown in its prompts.
    if setting == "1a":
        fields["USER_ACTION_STEP"] = user_action_step

    if condition is None:
        template_key = setting
    else:
        cfg = SETTING_CONDITIONS[condition]
        template_key = condition
        begin_s, end_s = clip_begin_seconds, to_seconds(clip_end_timestamp)
        if cfg["graph"]:
            fields["GRAPH_TEXT"] = render_graph_text(cut_window_subgraph(graph, begin_s, end_s))
        if cfg["chat"]:
            # rescue only applies to experiment 1b's window (ends at the AI action's own end);
            # experiment 1a's window ends at the user's later response, where "last message must
            # be Claude" isn't the right check -- see chat_for_window's docstring.
            chat_text = chat_for_window(
                participant_id, f"session{session}", begin_s, end_s, rescue=(setting == "1b")
            )
            if chat_text is None:
                print(
                    f"Warning: no aligned chat for {participant_id} session {session}; "
                    f"skipping {node_id!r} for {condition}.",
                    file=sys.stderr,
                )
                return None
            fields["CHAT_TEXT"] = chat_text or NO_CHAT_TEXT_SETTING

    return ai_node_id, TEMPLATES[template_key].format(**fields)


# ---------------------------------------------------------------------------
# Experiment 2a (collaboration-rating prediction) helpers
# ---------------------------------------------------------------------------

def load_occupations():
    """Map participant_id (lowercase, e.g. "p2") -> occupation string, read by
    column position since final_participants_for_iclr.csv's header has blank
    cells that break name-based CSV lookup."""
    occupations = {}
    with open(OCCUPATIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) <= 5 or not row[0].strip():
                continue
            participant_id = row[0].strip().lower()
            occupation = row[5].strip()
            if occupation:
                occupations[participant_id] = occupation
    return occupations


def find_graph_clean_path(participant_id, session):
    """participant_id like "p2", session like "1" or 1 -> the session's graph in
    new_annotated_graphs/, or None when it has none. (Name kept from when graph_clean/
    existed; there is only one graph folder now.)"""
    num = re.sub(r"^p", "", participant_id, flags=re.IGNORECASE)
    matches = sorted(ANNOTATED_GRAPHS_DIR.glob(f"session{session}_p{num}_*.json")) or \
        sorted(ANNOTATED_GRAPHS_DIR.glob(f"session{session}_P{num}_*.json"))
    return matches[0] if matches else None


def load_graph_clean(participant_id, session):
    path = find_graph_clean_path(participant_id, session)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def step_name_for_step_id(graph, step_id):
    """step_id like "step1" -> the step_name of the first graph segment whose
    "badge" contains that step number (badges like "Step 1", "Step 1-2",
    "Step 1+2+3" all match step number 1; "custom-*" badges are skipped)."""
    match = re.search(r"\d+", step_id)
    if not match:
        return None
    step_number = int(match.group(0))
    for step in graph.get("steps", []):
        badge = step.get("badge", "") or ""
        if "custom" in badge.lower():
            continue
        if step_number in (int(n) for n in re.findall(r"\d+", badge)):
            return step.get("step_name", "")
    return None


def build_experiment2a_prompt(row, occupations, graph_cache):
    """Return (experiment_id, prompt_text) for one basic_experiment_setup.csv
    row, or None if the goal/step name can't be resolved."""
    experiment_id = row["experiment_id"].strip()
    participant_id = row["participant_id"].strip().lower()
    session = re.sub(r"^session", "", row["session_id"].strip(), flags=re.IGNORECASE)
    step_id = row["step_id"].strip()

    cache_key = (participant_id, session)
    if cache_key not in graph_cache:
        graph_cache[cache_key] = load_graph_clean(participant_id, session)
    graph = graph_cache[cache_key]
    if graph is None:
        print(
            f"Warning: no graph_clean file for {participant_id} session {session}; skipping {experiment_id!r}.",
            file=sys.stderr,
        )
        return None

    goal = graph.get("goal", "")
    step_name = step_name_for_step_id(graph, step_id)
    if not step_name:
        print(
            f"Warning: step {step_id!r} not found in graph_clean for {participant_id} session {session}; "
            f"skipping {experiment_id!r}.",
            file=sys.stderr,
        )
        return None

    profession = occupations.get(participant_id)
    if not profession:
        print(
            f"Warning: no occupation found for {participant_id}; leaving placeholder.",
            file=sys.stderr,
        )
        profession = "professional"
    article = "an" if profession[:1].lower() in "aeiou" else "a"

    prompt = TEMPLATES["experiment2a"].format(
        PROFESSION_PHRASE=f"{article} {profession}",
        GOAL=goal,
        STEP_NAME=step_name,
    )
    return experiment_id, prompt


def generate_experiment2a_prompts(participant_filter, session_filter):
    if not EXPERIMENT2A_CSV.is_file():
        print(f"No such file: {EXPERIMENT2A_CSV}", file=sys.stderr)
        return

    with open(EXPERIMENT2A_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if participant_filter:
        rows = [r for r in rows if r["participant_id"].strip().lower() == participant_filter]
    if session_filter:
        rows = [r for r in rows if r["session_id"].strip().lower() == f"session{session_filter}"]

    if not rows:
        print("No matching rows found in basic_experiment_setup.csv.", file=sys.stderr)
        return

    occupations = load_occupations()
    graph_cache = {}

    output_dir = setting_output_dir("experiment2a")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated, skipped = 0, 0
    for row in tqdm(rows, desc="Generating experiment2a prompts"):
        out_path = output_dir / f"{row['experiment_id'].strip()}.txt"
        if out_path.exists():
            skipped += 1
            continue

        result = build_experiment2a_prompt(row, occupations, graph_cache)
        if result is None:
            continue
        _experiment_id, prompt = result

        out_path.write_text(prompt + "\n")
        generated += 1

    print(f"Generated {generated} prompt file(s); skipped {skipped} already-existing file(s).")


# ---------------------------------------------------------------------------
# Experiment 2b (output-quality-rating prediction) helpers
# ---------------------------------------------------------------------------

def build_experiment2b_prompt(row, occupations, graph_cache):
    """Return (experiment_id, prompt_text) for one experiment2b
    basic_experiment_setup.csv row, or None if the goal/step name can't be
    resolved. Structurally identical to build_experiment2a_prompt, just
    reading from EXPERIMENT2B_CSV and filling EXPERIMENT2B_TEMPLATE."""
    experiment_id = row["experiment_id"].strip()
    participant_id = row["participant_id"].strip().lower()
    session = re.sub(r"^session", "", row["session_id"].strip(), flags=re.IGNORECASE)
    step_id = row["step_id"].strip()

    cache_key = (participant_id, session)
    if cache_key not in graph_cache:
        graph_cache[cache_key] = load_graph_clean(participant_id, session)
    graph = graph_cache[cache_key]
    if graph is None:
        print(
            f"Warning: no graph_clean file for {participant_id} session {session}; skipping {experiment_id!r}.",
            file=sys.stderr,
        )
        return None

    goal = graph.get("goal", "")
    step_name = step_name_for_step_id(graph, step_id)
    if not step_name:
        print(
            f"Warning: step {step_id!r} not found in graph_clean for {participant_id} session {session}; "
            f"skipping {experiment_id!r}.",
            file=sys.stderr,
        )
        return None

    profession = occupations.get(participant_id)
    if not profession:
        print(
            f"Warning: no occupation found for {participant_id}; leaving placeholder.",
            file=sys.stderr,
        )
        profession = "professional"
    article = "an" if profession[:1].lower() in "aeiou" else "a"

    prompt = TEMPLATES["experiment2b"].format(
        PROFESSION_PHRASE=f"{article} {profession}",
        GOAL=goal,
        STEP_NAME=step_name,
    )
    return experiment_id, prompt


def generate_experiment2b_prompts(participant_filter, session_filter):
    if not EXPERIMENT2B_CSV.is_file():
        print(f"No such file: {EXPERIMENT2B_CSV}", file=sys.stderr)
        return

    with open(EXPERIMENT2B_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if participant_filter:
        rows = [r for r in rows if r["participant_id"].strip().lower() == participant_filter]
    if session_filter:
        rows = [r for r in rows if r["session_id"].strip().lower() == f"session{session_filter}"]

    if not rows:
        print("No matching rows found in basic_experiment_setup.csv.", file=sys.stderr)
        return

    occupations = load_occupations()
    graph_cache = {}

    output_dir = setting_output_dir("experiment2b")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated, skipped = 0, 0
    for row in tqdm(rows, desc="Generating experiment2b prompts"):
        out_path = output_dir / f"{row['experiment_id'].strip()}.txt"
        if out_path.exists():
            skipped += 1
            continue

        result = build_experiment2b_prompt(row, occupations, graph_cache)
        if result is None:
            continue
        _experiment_id, prompt = result

        out_path.write_text(prompt + "\n")
        generated += 1

    print(f"Generated {generated} prompt file(s); skipped {skipped} already-existing file(s).")


# ---------------------------------------------------------------------------
# Experiment 2c (session-level goal-achievement prediction, incremental video) helpers
# ---------------------------------------------------------------------------

def build_experiment2c_prompt(row):
    """Return (experiment_id, prompt_text) for one experiment2c
    basic_experiment_setup.csv row, or None if it has no goal text to build a
    prompt from. Unlike experiment2a/2b, the row already carries its own
    'occupation' and 'goal' columns (written by truncate_experiment2c_videos.py
    from session_ground_truth.csv), so there's no graph lookup here. Every row of
    a session (step1, step12, step123, ...) gets the same prompt text; only the
    video differs."""
    experiment_id = row["experiment_id"].strip()
    goal = row.get("goal", "").strip()
    if not goal:
        print(f"Warning: no goal text for {experiment_id!r}; skipping.", file=sys.stderr)
        return None

    profession = row.get("occupation", "").strip() or "professional"
    article = "an" if profession[:1].lower() in "aeiou" else "a"

    prompt = TEMPLATES["experiment2c"].format(
        PROFESSION_PHRASE=f"{article} {profession}",
        GOAL=goal,
    )
    return experiment_id, prompt


def generate_experiment2c_prompts(participant_filter, session_filter):
    if not EXPERIMENT2C_CSV.is_file():
        print(f"No such file: {EXPERIMENT2C_CSV} (run extract_experiment2c_ratings.py and truncate_experiment2c_videos.py first)", file=sys.stderr)
        return

    with open(EXPERIMENT2C_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if participant_filter:
        rows = [r for r in rows if r["participant_id"].strip().lower() == participant_filter]
    if session_filter:
        rows = [r for r in rows if r["session_id"].strip().lower() == f"session{session_filter}"]

    if not rows:
        print("No matching rows found in basic_experiment_setup.csv.", file=sys.stderr)
        return

    output_dir = setting_output_dir("experiment2c")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated, skipped = 0, 0
    for row in tqdm(rows, desc="Generating experiment2c prompts"):
        out_path = output_dir / f"{row['experiment_id'].strip()}.txt"
        if out_path.exists():
            skipped += 1
            continue

        result = build_experiment2c_prompt(row)
        if result is None:
            continue
        _experiment_id, prompt = result

        out_path.write_text(prompt + "\n")
        generated += 1

    print(f"Generated {generated} prompt file(s); skipped {skipped} already-existing file(s).")


ACTOR_DISPLAY_NAME = {"user": "user", "ai": "ai"}


def node_clip_time_text(node, clip_ranges):
    """The "[HH:MM:SS-HH:MM:SS]" text for one node, with its session-video time
    ranges converted to the step clip's clock (see session_range_to_clip_ranges).
    A node spread over several pieces of the clip lists each, comma-separated.
    Returns "" for a node with no time, or one lying entirely in a gap between
    the clip's pieces (its footage is not in the clip)."""
    time_field = node.get("time")
    if not time_field:
        return ""
    session_ranges = [time_field] if isinstance(time_field, str) else list(time_field)
    pieces = []
    for session_range in session_ranges:
        try:
            start, end = split_time_range(session_range)
            start_s, end_s = to_seconds(start), to_seconds(end)
        except ValueError:
            continue
        for clip_start, clip_end in session_range_to_clip_ranges(start_s, end_s, clip_ranges):
            pieces.append(f"{to_timestamp(clip_start)}-{to_timestamp(clip_end)}")
    return f"[{', '.join(pieces)}]" if pieces else ""


def render_graph_text(subgraph_or_subgraphs, with_timestamps=False):
    """Render one cut subgraph (from build_2a_graph_setup.py) or a list of raw
    annotated_graphs step dicts as readable text for a "*_graph" template's
    {GRAPH_TEXT} placeholder: one line per node, then the edge list with a
    one-line explanation of what an edge means.

    By default node lines carry no timestamps (the graph is read as pure
    task/dependency structure). With with_timestamps=True (experiment2a's "*_ts"
    conditions, single subgraph only) each line also carries the node's time range
    on the step clip's clock -- the subgraph's "timestamp_ranges" (the setup CSV's
    cell for that step) says which pieces of the session video the clip is made of.

    A list with more than one entry (the experiment1a/experiment1b graph conditions' clip
    window crossing steps: the AI action and the user's response to it are in
    different steps) gets a "[Step: ...]" header per entry so it's clear which nodes belong to
    which step; a single subgraph (the normal case, and always the case for
    experiment2a_graph) renders exactly as before, with no header."""
    subgraphs = [subgraph_or_subgraphs] if isinstance(subgraph_or_subgraphs, dict) else list(subgraph_or_subgraphs)
    multi = len(subgraphs) > 1

    clip_ranges = None
    if with_timestamps:
        if multi or not subgraphs[0].get("timestamp_ranges"):
            raise ValueError("with_timestamps needs a single subgraph carrying a 'timestamp_ranges' string")
        clip_ranges = parse_clip_ranges(subgraphs[0]["timestamp_ranges"])

    lines = []
    for subgraph in subgraphs:
        if multi:
            header = subgraph.get("step_name") or ""
            badge = subgraph.get("badge")
            lines.append(f"[Step: {badge + ' - ' if badge else ''}{header}]")
        for n in subgraph.get("nodes", []):
            actor_raw = n.get("actor_type") or n.get("actor") or ""
            actor = ACTOR_DISPLAY_NAME.get(actor_raw, actor_raw)
            label = n.get("label", "")
            detail = n.get("detail", "")

            time_text = node_clip_time_text(n, clip_ranges) if clip_ranges is not None else ""
            line = f"[{n.get('id')}] ({actor}) {time_text + ' ' if time_text else ''}{label}"
            if detail:
                line += f": {detail}"
            lines.append(line)
        if multi:
            lines.append("")
    lines.append("")
    lines.append(
        "A directed edge X -> Y means that action Y follows from or depends on action X. "
        "The graph is directed and may contain branches."
    )
    lines.append("")
    lines.append("Edges (order of actions):")
    for subgraph in subgraphs:
        for e in subgraph.get("edges", []):
            edge_type = e.get("type", "")
            suffix = f" ({edge_type})" if edge_type and edge_type != "sequential" else ""
            lines.append(f"{e.get('from')} -> {e.get('to')}{suffix}")
    return "\n".join(lines)


def load_annotated_graph(participant_id, session):
    """The session's graph from new_annotated_graphs/, or None when it has none."""
    try:
        return load_graph(int(session), participant_id)
    except FileNotFoundError:
        return None


def build_experiment2a_graph_prompt(row, occupations, graph_cache, template_key="experiment2a_graph",
                                    with_timestamps=False, annotated_only=False, chat_text=""):
    """Like build_experiment2a_prompt, but also embeds the row's cut subgraph
    (via its graph_filename column) as text.

    template_key picks the TEMPLATES entry ("experiment2a_graph" = video + graph,
    "experiment2a_graph_only" = no video); with_timestamps adds each node's
    clip-relative time range to the graph text (and to the template's node
    description). annotated_only=True reads goal/step name from annotated_graphs/
    (the experiment2a video-graph / graph-only conditions) instead of
    graph_clean/-then-annotated_graphs/ (the original experiment2a_graph run)."""
    experiment_id = row["experiment_id"].strip()
    participant_id = row["participant_id"].strip().lower()
    session = re.sub(r"^session", "", row["session_id"].strip(), flags=re.IGNORECASE)
    step_id = row["step_id"].strip()
    graph_filename = row.get("graph_filename", "").strip()

    if not graph_filename:
        print(
            f"Warning: no graph_filename for {experiment_id!r} (run build_2a_graph_setup.py first); skipping.",
            file=sys.stderr,
        )
        return None

    subgraph_path = GRAPHS_2A_DIR / graph_filename
    if not subgraph_path.is_file():
        print(f"Warning: missing cut graph file {subgraph_path.name}; skipping {experiment_id!r}.", file=sys.stderr)
        return None
    subgraph = json.loads(subgraph_path.read_text(encoding="utf-8"))
    try:
        graph_text = render_graph_text(subgraph, with_timestamps=with_timestamps)
    except ValueError as exc:
        print(f"Warning: {exc}; skipping {experiment_id!r}.", file=sys.stderr)
        return None

    cache_key = (participant_id, session)
    if cache_key not in graph_cache:
        loader = load_annotated_graph if annotated_only else load_graph_clean
        graph_cache[cache_key] = loader(participant_id, session)
    graph = graph_cache[cache_key]
    if graph is None:
        print(
            f"Warning: no graph_clean/annotated_graphs file for {participant_id} session {session}; "
            f"skipping {experiment_id!r}.",
            file=sys.stderr,
        )
        return None

    goal = graph.get("goal", "")
    step_name = step_name_for_step_id(graph, step_id) or subgraph.get("step_name", "")
    if not step_name:
        print(
            f"Warning: step {step_id!r} not found for {participant_id} session {session}; "
            f"skipping {experiment_id!r}.",
            file=sys.stderr,
        )
        return None

    profession = occupations.get(participant_id)
    if not profession:
        print(f"Warning: no occupation found for {participant_id}; leaving placeholder.", file=sys.stderr)
        profession = "professional"
    article = "an" if profession[:1].lower() in "aeiou" else "a"

    prompt = TEMPLATES[template_key].format(
        PROFESSION_PHRASE=f"{article} {profession}",
        GOAL=goal,
        STEP_NAME=step_name,
        GRAPH_TEXT=graph_text,
        NODE_DESCRIPTION=node_description(with_timestamps),
        CHAT_TEXT=chat_text,
    )
    return experiment_id, prompt


def build_experiment2a_chat_prompt(row, occupations, graph_cache, template_key, chat_text):
    """Like build_experiment2a_graph_prompt, but with no graph in the prompt (the
    experiment2a_video_chat / experiment2a_chat_only conditions): goal and step name
    come from annotated_graphs/, the chat from chat_snippets.csv."""
    experiment_id = row["experiment_id"].strip()
    participant_id = row["participant_id"].strip().lower()
    session = re.sub(r"^session", "", row["session_id"].strip(), flags=re.IGNORECASE)
    step_id = row["step_id"].strip()

    cache_key = (participant_id, session)
    if cache_key not in graph_cache:
        graph_cache[cache_key] = load_annotated_graph(participant_id, session)
    graph = graph_cache[cache_key]
    step_name = step_name_for_step_id(graph, step_id) if graph else None
    if not step_name:
        print(f"Warning: no goal/step name for {experiment_id!r} in annotated_graphs; skipping.", file=sys.stderr)
        return None

    profession = occupations.get(participant_id)
    if not profession:
        print(f"Warning: no occupation found for {participant_id}; leaving placeholder.", file=sys.stderr)
        profession = "professional"
    article = "an" if profession[:1].lower() in "aeiou" else "a"

    prompt = TEMPLATES[template_key].format(
        PROFESSION_PHRASE=f"{article} {profession}",
        GOAL=graph.get("goal", ""),
        STEP_NAME=step_name,
        CHAT_TEXT=chat_text,
    )
    return experiment_id, prompt


def generate_experiment2a_chat_prompts(participant_filter, session_filter, condition, force=False, preview_id=None):
    """Prompts of one CHAT_CONDITIONS key, for every annotated step that has a row in
    chat_snippets.csv (steps without one -- session not aligned yet -- are skipped).
    Written to prompts/<condition>/<experiment_id>.txt; existing files are skipped unless
    force. preview_id prints that one step's prompt instead and writes nothing."""
    cfg = CHAT_CONDITIONS[condition]
    setup_csv = cfg["csv"]
    if not setup_csv.is_file():
        builder = "build_2a_graph_setup.py" + (" --experiment 2b" if setup_csv == ANNOTATED_SETUP_CSV_2B else "")
        print(f"No such file: {setup_csv} (run {builder} first)", file=sys.stderr)
        return
    if not CHAT_SNIPPETS_CSV.is_file():
        print(f"No such file: {CHAT_SNIPPETS_CSV} (run run_chat_pipeline.sh first)", file=sys.stderr)
        return

    with open(setup_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(CHAT_SNIPPETS_CSV, newline="", encoding="utf-8") as f:
        snippets = {r["experiment_id"]: r["chat_text"] for r in csv.DictReader(f)}

    if participant_filter:
        rows = [r for r in rows if r["participant_id"].strip().lower() == participant_filter]
    if session_filter:
        rows = [r for r in rows if r["session_id"].strip().lower() == f"session{session_filter}"]
    if preview_id:
        rows = [r for r in rows if r["experiment_id"].strip() == preview_id]
    rows = [r for r in rows if r["experiment_id"].strip() in snippets]
    if not rows:
        print(f"No matching rows with a chat snippet found for {condition}.", file=sys.stderr)
        return

    occupations = load_occupations()
    graph_cache = {}

    def build(row):
        chat_text = snippets[row["experiment_id"].strip()].strip() or NO_CHAT_TEXT
        if cfg["graph"]:
            return build_experiment2a_graph_prompt(
                row, occupations, graph_cache, cfg["template"], with_timestamps=False,
                annotated_only=True, chat_text=chat_text,
            )
        return build_experiment2a_chat_prompt(row, occupations, graph_cache, cfg["template"], chat_text)

    if preview_id:
        result = build(rows[0])
        if result is not None:
            print(result[1])
        return

    output_dir = setting_output_dir(condition)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated, skipped = 0, 0
    for row in tqdm(rows, desc=f"Generating {condition} prompts"):
        out_path = output_dir / f"{row['experiment_id'].strip()}.txt"
        if out_path.exists() and not force:
            skipped += 1
            continue
        result = build(row)
        if result is None:
            continue
        out_path.write_text(result[1] + "\n")
        generated += 1
    print(f"Generated {generated} prompt file(s) in {output_dir}; skipped {skipped} already-existing file(s).")


def generate_experiment2a_graph_prompts(participant_filter, session_filter, condition="experiment2a_graph",
                                        force=False, preview_id=None):
    """condition "experiment2a_graph" is the original video+graph run (reads
    with_graph_experiment_setup.csv, graph_clean/-first); any GRAPH_CONDITIONS key
    reads annotated_graph_experiment_setup.csv (annotated_graphs/ only). Prompts are
    written to prompts/<condition>/<experiment_id>.txt; existing files are skipped
    unless force. preview_id prints that one step's prompt instead and writes nothing."""
    if condition in GRAPH_CONDITIONS:
        setup_csv = GRAPH_CONDITIONS[condition]["csv"]
        template_key = GRAPH_CONDITIONS[condition]["template"]
        with_timestamps = GRAPH_CONDITIONS[condition]["timestamps"]
        annotated_only = True
        builder = "build_2a_graph_setup.py" + (" --experiment 2b" if setup_csv == ANNOTATED_SETUP_CSV_2B else "")
    else:
        setup_csv = WITH_GRAPH_CSV
        template_key, with_timestamps, annotated_only = "experiment2a_graph", False, False
        builder = "build_2a_graphs.py"

    if not setup_csv.is_file():
        print(f"No such file: {setup_csv} (run {builder} first)", file=sys.stderr)
        return

    with open(setup_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if participant_filter:
        rows = [r for r in rows if r["participant_id"].strip().lower() == participant_filter]
    if session_filter:
        rows = [r for r in rows if r["session_id"].strip().lower() == f"session{session_filter}"]
    if preview_id:
        rows = [r for r in rows if r["experiment_id"].strip() == preview_id]

    if not rows:
        print(f"No matching rows found in {setup_csv.name}.", file=sys.stderr)
        return

    occupations = load_occupations()
    graph_cache = {}

    if preview_id:
        result = build_experiment2a_graph_prompt(
            rows[0], occupations, graph_cache, template_key, with_timestamps, annotated_only
        )
        if result is not None:
            print(result[1])
        return

    output_dir = setting_output_dir(condition)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated, skipped = 0, 0
    for row in tqdm(rows, desc=f"Generating {condition} prompts"):
        out_path = output_dir / f"{row['experiment_id'].strip()}.txt"
        if out_path.exists() and not force:
            skipped += 1
            continue

        result = build_experiment2a_graph_prompt(
            row, occupations, graph_cache, template_key, with_timestamps, annotated_only
        )
        if result is None:
            continue
        _experiment_id, prompt = result

        out_path.write_text(prompt + "\n")
        generated += 1

    print(f"Generated {generated} prompt file(s) in {output_dir}; skipped {skipped} already-existing file(s).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--setting", required=True,
        choices=["1a", "1b", "experiment2a", "experiment2a_graph", "experiment2b", "experiment2c"]
        + list(GRAPH_CONDITIONS) + list(CHAT_CONDITIONS) + list(SETTING_CONDITIONS),
        help="Which setting to generate prompts for",
    )
    parser.add_argument("--p", required=False, default=None, help="Participant id, e.g. p1")
    parser.add_argument(
        "--session", required=False, default=None, choices=["1", "2"], help="Session number: 1 or 2"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="graph/chat conditions only: overwrite prompt files that already exist",
    )
    parser.add_argument(
        "--preview", metavar="EXPERIMENT_ID", default=None,
        help="graph/chat conditions only: print this one prompt (an experiment_id, e.g. exp1_p1_s1 for the "
             "setting conditions) to the terminal and write nothing",
    )
    args = parser.parse_args()

    participant_filter = args.p.strip().lower() if args.p else None
    session_filter = int(args.session) if args.session else None

    if (args.force or args.preview) and not (
        args.setting in GRAPH_CONDITIONS or args.setting in CHAT_CONDITIONS or args.setting in SETTING_CONDITIONS
        or args.setting == "experiment2a_graph"
    ):
        parser.error("--force/--preview only apply to the graph/chat condition settings")

    if args.setting == "experiment2a":
        generate_experiment2a_prompts(participant_filter, session_filter)
        return
    if args.setting in CHAT_CONDITIONS:
        generate_experiment2a_chat_prompts(
            participant_filter, session_filter, args.setting, force=args.force, preview_id=args.preview
        )
        return
    if args.setting == "experiment2a_graph" or args.setting in GRAPH_CONDITIONS:
        generate_experiment2a_graph_prompts(
            participant_filter, session_filter, args.setting, force=args.force, preview_id=args.preview
        )
        return
    if args.setting == "experiment2b":
        generate_experiment2b_prompts(participant_filter, session_filter)
        return
    if args.setting == "experiment2c":
        generate_experiment2c_prompts(participant_filter, session_filter)
        return

    if args.setting in SETTING_CONDITIONS:
        base_setting, condition = SETTING_CONDITIONS[args.setting]["setting"], args.setting
    else:
        base_setting, condition = args.setting, None

    jobs = discover_jobs(participant_filter, session_filter, base_setting)
    if not jobs:
        print(f"No matching timestamp CSVs found in labels_experiment{base_setting}.", file=sys.stderr)
        return

    tasks = collect_tasks(jobs)
    if args.preview:
        tasks = [t for t in tasks if t[-1] == args.preview]
    occupations = load_occupations()

    output_dir = setting_output_dir(args.setting)
    if not args.preview:
        output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    for (
        participant_id,
        session,
        graph,
        node_index,
        node_id,
        target_ai_timestamp,
        clip_begin_timestamp,
        clip_end_timestamp,
        experiment_id,
    ) in tqdm(tasks, desc=f"Generating {args.setting} prompts", disable=bool(args.preview)):
        out_path = output_dir / prompt_filename(experiment_id, participant_id, session)
        if out_path.exists() and not args.force and not args.preview:
            skipped += 1
            continue

        result = build_prompt(
            graph, node_index, node_id, target_ai_timestamp, clip_begin_timestamp, clip_end_timestamp,
            participant_id, session, base_setting, occupations, condition=condition,
        )
        if result is None:
            continue
        _ai_node_id, prompt = result

        if args.preview:
            print(prompt)
            return
        out_path.write_text(prompt + "\n")
        generated += 1

    print(f"Generated {generated} prompt file(s); skipped {skipped} already-existing file(s).")


if __name__ == "__main__":
    main()
