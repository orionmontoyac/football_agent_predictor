from typing import Any, Sequence

from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from config import Settings, get_settings


class OllamaModelBuilder:
    """Builds a ChatOllama instance from application settings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    def ollama_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ChatOllama, omitting unset optional fields."""
        s = self._settings
        kwargs: dict[str, Any] = {
            "model": s.ollama_model,
            "validate_model_on_init": s.ollama_validate_model_on_init,
        }
        if s.ollama_base_url is not None:
            kwargs["base_url"] = s.ollama_base_url
        if s.ollama_temperature is not None:
            kwargs["temperature"] = s.ollama_temperature
        if s.ollama_num_predict is not None:
            kwargs["num_predict"] = s.ollama_num_predict
        if s.ollama_reasoning is not None:
            kwargs["reasoning"] = s.ollama_reasoning
        if s.ollama_format is not None:
            kwargs["format"] = s.ollama_format
        return kwargs

    def build(self, *, tools: Sequence[BaseTool] | None = None) -> ChatOllama:
        """Create the Ollama chat model, optionally bound to tools."""
        model = ChatOllama(**self.ollama_kwargs())
        if tools:
            return model.bind_tools(list(tools))
        return model
