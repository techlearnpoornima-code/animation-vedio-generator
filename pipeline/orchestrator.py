"""
pipeline/orchestrator.py
LangGraph Orchestrator — wires all 6 agents into a directed graph.

Flow:
  init → trend_research → script_writing → quality_gate
       → story_character → animation → voiceover → editor_upload → done

With retry logic: if quality_gate fails, re-run script_writing (up to 2 retries).
"""

import os
from typing import Annotated
from langgraph.graph import StateGraph, END
from pipeline.state import EpisodeState
from agents.trend_researcher import TrendResearcher
from agents.script_writer import ScriptWriter
from agents.story_character import StoryCharacterAgent
from agents.animation_agent import AnimationAgent
from agents.voiceover_agent import VoiceoverAgent
from agents.editor_upload import EditorUploadAgent
from pipeline.quality_gate import QualityGate

MAX_SCRIPT_RETRIES = 2


# ------------------------------------------------------------------ #
#  Node functions (each wraps one agent)                               #
# ------------------------------------------------------------------ #

def node_trend_research(state: dict) -> dict:
    ep = EpisodeState(**state)
    genre_hint = state.get("_genre_hint", "")
    ep = TrendResearcher().run(ep, genre_hint=genre_hint)
    return ep.model_dump()


def node_script_writing(state: dict) -> dict:
    ep = EpisodeState(**state)
    ep = ScriptWriter().run(ep)
    return ep.model_dump()


def node_story_character(state: dict) -> dict:
    ep = EpisodeState(**state)
    generate_images = state.get("_generate_images", False)
    ep = StoryCharacterAgent().run(ep, generate_images=generate_images)
    return ep.model_dump()


def node_quality_gate(state: dict) -> dict:
    ep = EpisodeState(**state)
    ep, passed = QualityGate().run(ep)

    # Track retry count in state extras
    retries = state.get("_script_retries", 0)
    result = ep.model_dump()
    result["_qa_passed"] = passed
    result["_script_retries"] = retries
    return result


def node_animation(state: dict) -> dict:
    ep = EpisodeState(**state)
    ep = AnimationAgent().run(ep)
    return ep.model_dump()


def node_voiceover(state: dict) -> dict:
    ep = EpisodeState(**state)
    ep = VoiceoverAgent().run(ep)
    return ep.model_dump()


def node_editor_upload(state: dict) -> dict:
    ep = EpisodeState(**state)
    upload = state.get("_upload_to_youtube", False)
    ep = EditorUploadAgent().run(ep, upload=upload)
    return ep.model_dump()


# ------------------------------------------------------------------ #
#  Conditional routing                                                  #
# ------------------------------------------------------------------ #

def route_after_qa(state: dict) -> str:
    """Route to story_character if QA passed, back to script if failed (with retry limit)."""
    passed = state.get("_qa_passed", True)
    retries = state.get("_script_retries", 0)

    if passed:
        return "story_character"
    elif retries < MAX_SCRIPT_RETRIES:
        print(f"  [orchestrator] QA failed. Retrying script (attempt {retries + 1}/{MAX_SCRIPT_RETRIES})...")
        state["_script_retries"] = retries + 1
        return "script_writing"
    else:
        print(f"  [orchestrator] QA failed after {MAX_SCRIPT_RETRIES} retries. Proceeding anyway.")
        return "story_character"


def route_after_errors(state: dict) -> str:
    """If critical errors exist after an agent, skip to end."""
    ep = EpisodeState(**state)
    critical = [e for e in ep.errors if "trend_research" in e or "script_writer" in e]
    if critical:
        print(f"  [orchestrator] Critical errors found. Aborting pipeline.")
        return END
    return "continue"


# ------------------------------------------------------------------ #
#  Graph construction                                                   #
# ------------------------------------------------------------------ #

def build_graph() -> StateGraph:
    graph = StateGraph(dict)

    graph.add_node("trend_research",  node_trend_research)
    graph.add_node("script_writing",  node_script_writing)
    graph.add_node("quality_gate",    node_quality_gate)
    graph.add_node("story_character", node_story_character)
    graph.add_node("animation",       node_animation)
    graph.add_node("voiceover",       node_voiceover)
    graph.add_node("editor_upload",   node_editor_upload)

    graph.set_entry_point("trend_research")

    graph.add_edge("trend_research",  "script_writing")
    graph.add_edge("script_writing",  "quality_gate")

    graph.add_conditional_edges(
        "quality_gate",
        route_after_qa,
        {
            "story_character": "story_character",
            "script_writing":  "script_writing",
        },
    )

    graph.add_edge("story_character", "animation")
    graph.add_edge("animation",       "voiceover")
    graph.add_edge("voiceover",       "editor_upload")
    graph.add_edge("editor_upload",   END)

    return graph.compile()


# ------------------------------------------------------------------ #
#  Public run function                                                  #
# ------------------------------------------------------------------ #

def run_pipeline(
    genre_hint: str = "",
    generate_images: bool = False,
    upload_to_youtube: bool = False,
    output_dir: str = "./output",
) -> EpisodeState:
    """
    Run the full cartoon production pipeline.

    Args:
        genre_hint:         Optional genre/tone hint for the trend researcher.
                            e.g. "dark comedy", "action thriller", "surreal horror"
        generate_images:    Whether to generate character reference images via SD.
        upload_to_youtube:  Whether to upload the final video to YouTube.
        output_dir:         Base output directory.

    Returns:
        Final EpisodeState with all results and paths.
    """
    print("\n" + "=" * 60)
    print("  CARTOON PRODUCTION PIPELINE")
    print("=" * 60)

    initial_state = {
        **EpisodeState(output_dir=output_dir).model_dump(),
        "_genre_hint":          genre_hint,
        "_generate_images":     generate_images,
        "_upload_to_youtube":   upload_to_youtube,
        "_qa_passed":           True,
        "_script_retries":      0,
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    final_ep = EpisodeState(**{k: v for k, v in final_state.items() if not k.startswith("_")})

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Episode ID:    {final_ep.episode_id}")
    print(f"  Title:         {final_ep.script.title if final_ep.script else 'N/A'}")
    print(f"  Steps done:    {', '.join(final_ep.completed_steps)}")
    print(f"  Final video:   {final_ep.final_video_path or 'N/A'}")
    print(f"  YouTube URL:   {final_ep.youtube_url or 'Not uploaded'}")
    if final_ep.errors:
        print(f"  Errors:        {len(final_ep.errors)}")
        for e in final_ep.errors:
            print(f"    - {e}")
    if final_ep.warnings:
        print(f"  Warnings:      {len(final_ep.warnings)}")
    print("=" * 60 + "\n")

    return final_ep
