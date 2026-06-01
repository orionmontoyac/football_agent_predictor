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
        default=None,
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
            "You are an expert football (soccer) analyst specializing in national-team match predictions.\n"
            "When the user asks about a fixture (e.g. 'Mexico vs South Africa'):\n"
            "1. Treat the first team named as the home side unless they say otherwise.\n"
            "2. Call lookup_team for each team to check form and FIFA ranking.\n"
            "3. Call get_match_head_to_head for historical context.\n"
            "4. Call predict_match_result with home_team and away_team. It returns a statistical\n"
            "   model prediction, live Polymarket odds (when available), and a blended forecast.\n"
            "5. Compare the model and the market: if they disagree, briefly explain why.\n"
            "6. Answer in clear, friendly prose: predicted score, likely winner, confidence, market\n"
            "   probabilities, and 2-3 key reasons.\n"
            "If a team is unknown, call list_supported_teams and tell the user. To browse upcoming\n"
            "matches, use list_world_cup_fixtures. For raw market odds only, use get_polymarket_odds.\n"
            "Always note that predictions are statistical/market estimates, not guarantees."
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

    # --- App ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
    )

    def langgraph_invoke_config(self) -> dict[str, Any]:
        """RunnableConfig-style dict for graph.invoke / graph.stream."""
        return {"recursion_limit": self.langgraph_recursion_limit}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (reload process to pick up env changes)."""
    return Settings()
