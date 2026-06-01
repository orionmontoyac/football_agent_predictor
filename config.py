from functools import lru_cache
from typing import Any, Literal

from langchain_ollama import ChatOllama
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
        default="You are a helpful assistant. Please respond to the user's request.",
        description="System message prepended on each model call.",
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

    def ollama_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ChatOllama, omitting unset optional fields."""
        kwargs: dict[str, Any] = {
            "model": self.ollama_model,
            "validate_model_on_init": self.ollama_validate_model_on_init,
        }
        if self.ollama_base_url is not None:
            kwargs["base_url"] = self.ollama_base_url
        if self.ollama_temperature is not None:
            kwargs["temperature"] = self.ollama_temperature
        if self.ollama_num_predict is not None:
            kwargs["num_predict"] = self.ollama_num_predict
        if self.ollama_reasoning is not None:
            kwargs["reasoning"] = self.ollama_reasoning
        if self.ollama_format is not None:
            kwargs["format"] = self.ollama_format
        return kwargs

    def create_chat_model(self, *, bind_tools: list | None = None) -> ChatOllama:
        """Build a ChatOllama instance from current settings."""
        model = ChatOllama(**self.ollama_kwargs())
        if bind_tools:
            return model.bind_tools(bind_tools)
        return model

    def langgraph_invoke_config(self) -> dict[str, Any]:
        """RunnableConfig-style dict for graph.invoke / graph.stream."""
        return {"recursion_limit": self.langgraph_recursion_limit}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (reload process to pick up env changes)."""
    return Settings()
