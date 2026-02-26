"""
pipeline/state.py
Shared state object passed between all agents in the LangGraph pipeline.
Each agent reads from and writes to this state.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class TrendData(BaseModel):
    topic: str
    genre: str                      # e.g. "action", "comedy", "horror", "satire"
    tone: str                       # e.g. "dark", "absurdist", "heartfelt"
    target_audience: str            # e.g. "adults", "teens", "general"
    style_tags: List[str] = []      # e.g. ["anime", "retro", "2D flat"]
    hooks: List[str] = []           # audience engagement angles
    search_volume: Optional[str] = None
    source_urls: List[str] = []


class ScriptData(BaseModel):
    title: str
    episode_number: int
    synopsis: str
    scenes: List[dict] = []         # [{scene_num, location, characters, action, dialogue}]
    total_scenes: int = 0
    estimated_duration_mins: float = 0.0
    tone: str = ""
    raw_text: str = ""


class CharacterData(BaseModel):
    characters: List[dict] = []     # [{name, description, visual_prompt, voice_profile}]
    style_reference: str = ""       # SD style prompt prefix for all scenes
    color_palette: List[str] = []
    scene_prompts: List[str] = []   # one SD prompt per scene


class AnimationData(BaseModel):
    clip_paths: List[str] = []      # paths to generated video clips
    total_clips: int = 0
    fps: int = 24
    resolution: str = "512x512"


class AudioData(BaseModel):
    audio_paths: List[str] = []     # paths to generated audio files per scene
    total_duration_secs: float = 0.0
    voice_map: dict = {}            # character_name -> voice_id


class EpisodeState(BaseModel):
    """Master state object for the entire pipeline run."""

    # Metadata
    episode_id: str = Field(default_factory=lambda: f"EP_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    output_dir: str = "./output"

    # Agent outputs (filled progressively)
    trend: Optional[TrendData] = None
    script: Optional[ScriptData] = None
    characters: Optional[CharacterData] = None
    animation: Optional[AnimationData] = None
    audio: Optional[AudioData] = None
    final_video_path: Optional[str] = None
    youtube_url: Optional[str] = None

    # Quality gate scores
    script_score: float = 0.0
    character_score: float = 0.0

    # Pipeline control
    current_step: str = "init"
    errors: List[str] = []
    warnings: List[str] = []
    completed_steps: List[str] = []

    def mark_done(self, step: str):
        self.completed_steps.append(step)
        self.current_step = step

    def add_error(self, step: str, msg: str):
        self.errors.append(f"[{step}] {msg}")

    def is_healthy(self) -> bool:
        return len(self.errors) == 0
