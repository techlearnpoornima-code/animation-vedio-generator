"""
agents/trend_researcher.py
Agent 1 — Trend Researcher

Identifies trending topics and returns a structured creative brief.
Uses SerpAPI (free tier) for live trend data with Google Trends RSS as fallback.
"""

import os
import json
import feedparser
import requests
from dotenv import load_dotenv
from pipeline.state import EpisodeState, TrendData
from pipeline.ollama_client import OllamaClient

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss?geo=US"

SYSTEM_PROMPT = """You are a creative director for an animated cartoon studio.
Your job is to analyze trending topics and turn them into compelling cartoon episode concepts.
You create briefs that work across all audiences — from absurdist adult comedy to dark thriller animation.
Be bold, specific, and creative. Avoid generic or safe ideas."""


class TrendResearcher:
    def __init__(self):
        self.llm = OllamaClient(model=os.getenv("OLLAMA_MODEL_FAST", "llama3"))

    # ------------------------------------------------------------------ #
    #  Data fetching                                                        #
    # ------------------------------------------------------------------ #

    def _fetch_serpapi_trends(self) -> list[str]:
        """Fetch trending searches via SerpAPI (uses free quota)."""
        if not SERPAPI_KEY:
            return []
        try:
            resp = requests.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google_trends_trending_now",
                    "api_key": SERPAPI_KEY,
                    "geo": "US",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item.get("query", "") for item in data.get("trending_searches", [])[:15]]
        except Exception as e:
            print(f"  [trend_researcher] SerpAPI failed: {e}")
            return []

    def _fetch_rss_trends(self) -> list[str]:
        """Fallback: parse Google Trends RSS feed (no API key needed)."""
        try:
            feed = feedparser.parse(GOOGLE_TRENDS_RSS)
            return [entry.title for entry in feed.entries[:15]]
        except Exception as e:
            print(f"  [trend_researcher] RSS fallback failed: {e}")
            return []

    def _get_raw_trends(self) -> list[str]:
        trends = self._fetch_serpapi_trends()
        if not trends:
            print("  [trend_researcher] Falling back to Google Trends RSS...")
            trends = self._fetch_rss_trends()
        if not trends:
            # Last resort: ask the LLM for culturally relevant topics
            print("  [trend_researcher] No live data. Using LLM cultural awareness...")
            result = self.llm.chat_json(
                "List 10 currently culturally relevant topics that would make great cartoon episodes.",
                system=SYSTEM_PROMPT,
            )
            trends = result if isinstance(result, list) else result.get("topics", [])
        return trends

    # ------------------------------------------------------------------ #
    #  Core analysis                                                        #
    # ------------------------------------------------------------------ #

    def _analyze_trends(self, raw_trends: list[str], genre_hint: str = "") -> dict:
        """Ask Ollama to pick the best topic and build a creative brief."""
        trends_str = "\n".join(f"- {t}" for t in raw_trends)
        genre_note = f"\nPreferred genre/tone hint: {genre_hint}" if genre_hint else ""

        prompt = f"""Here are current trending topics:
{trends_str}
{genre_note}

Pick the single best topic for a cartoon episode and return a JSON creative brief with this exact structure:
{{
  "topic": "the chosen topic",
  "genre": "one of: action, comedy, horror, satire, drama, sci-fi, fantasy, thriller",
  "tone": "one of: dark, absurdist, heartfelt, gritty, whimsical, surreal, tense",
  "target_audience": "one of: adults, teens, general, mature",
  "style_tags": ["up to 4 visual style tags like anime, retro, 2D flat, noir, cel-shaded"],
  "hooks": ["3 specific audience engagement angles or emotional beats"],
  "search_volume": "high / medium / low",
  "source_urls": []
}}"""

        return self.llm.chat_json(prompt, system=SYSTEM_PROMPT)

    # ------------------------------------------------------------------ #
    #  Public entrypoint                                                    #
    # ------------------------------------------------------------------ #

    def run(self, state: EpisodeState, genre_hint: str = "") -> EpisodeState:
        print("\n[1/6] Trend Researcher starting...")

        try:
            raw_trends = self._get_raw_trends()
            print(f"  Found {len(raw_trends)} trends. Analyzing with Ollama...")

            brief = self._analyze_trends(raw_trends, genre_hint)

            state.trend = TrendData(**brief)
            state.mark_done("trend_research")

            print(f"  Topic: {state.trend.topic}")
            print(f"  Genre: {state.trend.genre} | Tone: {state.trend.tone}")
            print(f"  Audience: {state.trend.target_audience}")
            print("  [1/6] DONE\n")

        except Exception as e:
            state.add_error("trend_research", str(e))
            print(f"  [trend_researcher] ERROR: {e}")

        return state


# ------------------------------------------------------------------ #
#  Standalone test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    state = EpisodeState()
    agent = TrendResearcher()
    state = agent.run(state, genre_hint="dark comedy")
    if state.trend:
        print(json.dumps(state.trend.model_dump(), indent=2))
