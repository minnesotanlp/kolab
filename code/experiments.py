"""
Shared definition of the seven input modes ("conditions"), used by experiments 1a, 1b, 1c and 1d by
run_qwen_setting.py, run_gemini_setting.py, run_gpt_setting.py and run_experiments.py.

A condition is named like run_experiment2a.py's --condition values (hyphenated). Each maps to the
prompt_maker_setting.py folder its prompts live in (extracted_timestamps/prompts/<folder>/, same
file names as the plain experiment<ID> prompts), the results folder
(experiment_results/<folder>/<model>_<folder>_results[_run<K>].csv -- what eval_results.py reads),
and whether the video clip is sent with the prompt.

    condition          folder suffix        video sent
    video-only         (none)               yes   (the plain experiment<ID> run)
    video-graph        _video_graph         yes
    graph-only         _graph_only          no
    video-chat         _video_chat          yes
    chat-only          _chat_only           no
    graph-chat         _graph_chat          no
    video-graph-chat   _video_graph_chat    yes
"""

from pathlib import Path

# The classification experiments (adopt / modify / reject-bypass), as they are named on every command line
# and in every folder name (experiment<ID>, experiment<ID>_<condition>):
#   1a  predict the user's response from the clip up to that response
#   1b  clip ends where the AI action ends
#   1c  1b with the context cut to the last 1 / 2 / 3 steps, or the whole workflow ("full")
#   1d  1b with every clip starting at 0:00 (the whole workflow up to the AI action)
CLASSIFICATION_EXPERIMENTS = ("1a", "1b", "1c_last1", "1c_last2", "1c_last3", "1c_full", "1d")

from config import ROOT_DIR

# condition -> (folder suffix after "experiment<ID>", whether the video is sent)
SETTING_CONDITIONS = {
    "video-only": ("", True),
    "video-graph": ("_video_graph", True),
    "graph-only": ("_graph_only", False),
    "video-chat": ("_video_chat", True),
    "chat-only": ("_chat_only", False),
    "graph-chat": ("_graph_chat", False),
    "video-graph-chat": ("_video_graph_chat", True),
}
CONDITION_CHOICES = tuple(SETTING_CONDITIONS)


def condition_folder(setting, condition):
    """"experiment1a", "experiment1a_graph_only", ... -- the prompts folder AND the results folder."""
    return f"experiment{setting}{SETTING_CONDITIONS[condition][0]}"


def needs_video(condition):
    return SETTING_CONDITIONS[condition][1]


# experiment -> max clip size (MB) a video condition may send; larger clips are skipped.
MAX_VIDEO_MB = {}


# experiment -> Qwen's cap on sampled video frames per clip (run_experiments.py uses it unless
# --max_frames is given; every other experiment uses DEFAULT_MAX_VIDEO_FRAMES = 32). Experiment 1d's clips
# run from 00:00 (median ~35 min), so 32 frames would be ~1 per minute.
QWEN_MAX_FRAMES = {"1c_full": 128, "1d": 128}


def video_too_large(setting, video_path):
    """True if this setting has a size cap and video_path's clip exceeds it."""
    limit = MAX_VIDEO_MB.get(str(setting))
    return limit is not None and video_path.stat().st_size > limit * 1024 * 1024


def condition_prompts_dir(setting, condition):
    return ROOT_DIR / "extracted_timestamps" / "prompts" / condition_folder(setting, condition)


def condition_results_path(setting, condition, model_slug, run=None):
    """experiment_results/<folder>/<model_slug>_<folder>_results[_run<run>].csv"""
    folder = condition_folder(setting, condition)
    suffix = "" if run is None else f"_run{run}"
    return ROOT_DIR / "experiment_results" / folder / f"{model_slug}_{folder}_results{suffix}.csv"
