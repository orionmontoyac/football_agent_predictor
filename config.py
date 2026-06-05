from functools import lru_cache
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama / LLM ---
    ollama_model: str = Field(
        default="granite4.1:3b",
        description="Ollama model tag (e.g. granite4.1:3b, llama3.1:8b).",
    )
    ollama_base_url: str | None = Field(
        default=None,
        description="Ollama API base URL. Defaults to http://localhost:11434 when unset.",
    )
    ollama_temperature: float | None = Field(
        default=0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature passed to the model.",
    )
    ollama_num_predict: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate (Ollama num_predict).",
    )
    ollama_reasoning: bool | None = Field(
        default=None,
        description="Enable/disable reasoning mode for supported models.",
    )
    ollama_format: Literal["", "json"] | None = Field(
        default=None,
        description='Response format; use "json" for JSON mode.',
    )
    ollama_validate_model_on_init: bool = Field(
        default=False,
        description="Verify the model exists in Ollama when ChatOllama is created.",
    )

    # --- Agent / prompts ---
    agent_system_prompt: str = Field(
        default=(
            "You help win the Polla Mundialista by MAXIMIZING contest points.\n"
            "Scoring — Group (primera ronda): winner/draw 5, home goals 2, away goals 2, diff 1 (max 10).\n"
            "Knockout (fases eliminatorias): winner/draw 10, home goals 4, away goals 4, diff 2 (max 20).\n"
            "Winner/draw is worth half the points, so prioritize it.\n"
            "IMPORTANT: only the 90-minute result counts (no extra time or penalties), so a DRAW is a\n"
            "valid prediction even in knockout matches.\n\n"
            "Tool routing (important):\n"
            "- ONE fixture (e.g. 'Mexico vs South Africa') -> predict_match_result(home, away, stage).\n"
            "- MANY matches / 'all matches' / 'today's/this date's picks' -> predict_all_fixtures(stage,\n"
            "  match_date). Pass match_date (YYYY-MM-DD) whenever the user names a day.\n"
            "- A finished match result the user reports -> record_match_result(home, away, hg, ag).\n"
            "- 'How did <team> play?' / past form -> get_team_recent_results(team).\n"
            "- 'How many points / how am I doing' -> get_points_summary.\n"
            "- list_world_cup_fixtures is ONLY the schedule; never use it to make picks.\n"
            "- After predicting, save picks with save_match_prediction (predict_* tools are read-only).\n"
            "Predictions use a team's saved past results in this tournament as context.\n"
            "Treat the first team named as home. Never invent\n"
            "fixtures, scores, or dates; use tool output only.\n\n"
            "Output is shown in a styled terminal (no markdown). Use ONLY plain ASCII text:\n"
            "no **bold**, *italic*, # headings, or backticks.\n"
            "The score, winner, probabilities, and expected points are ALREADY displayed by the app, so\n"
            "DO NOT repeat them. Reply with ONLY a one-line reason starting with 'Why: ', e.g.\n"
            "Why: higher FIFA ranking, better recent form, and home advantage.\n"
            "For multiple matches, give one short 'Why' line per match prefixed by the team names."
        ),
        description="System message prepended on each model call.",
    )

    # --- Polymarket (live odds enrichment) ---
    polymarket_enabled: bool = Field(
        default=True,
        description="Enable live odds enrichment from the Polymarket Gamma API.",
    )
    polymarket_base_url: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Polymarket Gamma API base URL (public, no auth).",
    )
    polymarket_world_cup_series_id: str = Field(
        default="11433",
        description="Gamma series id that groups all World Cup match events.",
    )
    polymarket_cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="How long to cache the fixtures list before refetching.",
    )
    polymarket_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        description="HTTP timeout for Polymarket requests.",
    )
    polymarket_market_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight given to market odds when blending with the heuristic model.",
    )

    # --- LangGraph runtime ---
    langgraph_recursion_limit: int = Field(
        default=25,
        ge=1,
        description="Max graph steps per invoke/stream (passed as recursion_limit).",
    )
    langgraph_stream_mode: Literal["values", "updates", "messages", "debug", "custom"] = Field(
        default="updates",
        description="Default stream_mode for graph.stream().",
    )

    # --- Persistence ---
    predictions_file: str = Field(
        default="data/world_cup.json",
        description="JSON file storing predictions and real results for the tournament.",
    )

    # --- App ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="DEBUG",
    )

    def langgraph_invoke_config(self) -> dict[str, Any]:
        """RunnableConfig-style dict for graph.invoke / graph.stream."""
        return {"recursion_limit": self.langgraph_recursion_limit}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (reload process to pick up env changes)."""
    return Settings()
