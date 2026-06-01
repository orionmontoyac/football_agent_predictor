"""CLI for the football match prediction agent."""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage

from config import get_settings
from graph import build_graph


def print_stream(stream) -> None:
    for step in stream:
        for node_name, update in step.items():
            if "messages" not in update:
                continue
            message = update["messages"][-1]
            print(f"\n--- {node_name} ---")
            if hasattr(message, "pretty_print"):
                message.pretty_print()
            else:
                print(message)


def run_query(query: str, *, stream: bool = True) -> None:
    settings = get_settings()
    app = build_graph(settings)
    input_state = {"messages": [HumanMessage(content=query)]}
    config = settings.langgraph_invoke_config()

    if stream:
        print_stream(
            app.stream(
                input_state,
                stream_mode=settings.langgraph_stream_mode,
                config=config,
            )
        )
    else:
        result = app.invoke(input_state, config=config)
        result["messages"][-1].pretty_print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Football match prediction agent (LangGraph + Ollama).",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What will be the result of Mexico vs South Africa?",
        help='Match question, e.g. "What will be the result of Mexico vs South Africa?"',
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print only the final answer instead of streaming node updates.",
    )
    args = parser.parse_args(argv)

    try:
        run_query(args.query, stream=not args.no_stream)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
