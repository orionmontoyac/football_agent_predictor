"""CLI for the football match prediction agent."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import terminal
from config import get_settings
from graph import build_graph
from logging_config import configure_logging, get_logger, log_event

logger = get_logger("main")


def _pct(value) -> str:
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return "?"


def _extract_predictions(messages) -> list[dict]:
    """Pull structured predictions (teams, score, winner, probs, points) from tool outputs."""
    preds: list[dict] = []
    seen: set[tuple] = set()

    def add(pred: dict) -> None:
        key = (pred["home"], pred["away"], pred["score"])
        if pred["home"] and pred["away"] and pred["score"] and key not in seen:
            seen.add(key)
            preds.append(pred)

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            data = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue

        bet = data.get("recommended_bet")
        if isinstance(bet, dict):
            model = data.get("model_prediction", {})
            probs = bet.get("result_probabilities", {})
            add(
                {
                    "home": str(model.get("home_team", "")),
                    "away": str(model.get("away_team", "")),
                    "score": str(bet.get("score", "")),
                    "winner": bet.get("winner"),
                    "probs": probs,
                    "expected_points": bet.get("expected_points"),
                    "max_points": bet.get("max_points"),
                    "stage": bet.get("stage") or data.get("stage"),
                }
            )
        for pick in data.get("picks", []) or []:
            if not isinstance(pick, dict):
                continue
            parts = re.split(r"\s+vs\.?\s+", str(pick.get("match", "")), maxsplit=1)
            if len(parts) == 2:
                add(
                    {
                        "home": parts[0].strip(),
                        "away": parts[1].strip(),
                        "score": str(pick.get("score", "")),
                        "winner": pick.get("winner"),
                        "probs": {},
                        "expected_points": pick.get("expected_points"),
                        "max_points": data.get("max_points"),
                        "stage": data.get("stage"),
                    }
                )
    return preds


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)
    return str(content).strip()


def format_for_terminal(text: str) -> str:
    """Strip common markdown so CLI output reads cleanly in a terminal."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("•", "-")
    return text.strip()


def print_terminal(message, predictions=None) -> None:
    text = format_for_terminal(_message_text(message))
    predictions = predictions or []
    print()

    for pred in predictions:
        for line in terminal.banner(pred["home"], pred["away"], pred["score"]):
            print(line)
        # Show key stats deterministically (single match only, to stay compact).
        if len(predictions) == 1:
            if pred.get("winner"):
                print(terminal.stat("Winner", pred["winner"], "green"))
            probs = pred.get("probs") or {}
            if probs:
                value = (
                    f"H {_pct(probs.get('home_win'))} / "
                    f"D {_pct(probs.get('draw'))} / "
                    f"A {_pct(probs.get('away_win'))}"
                )
                print(terminal.stat("Win prob", value, "magenta"))
            if pred.get("expected_points") is not None:
                stage = f" ({pred['stage']})" if pred.get("stage") else ""
                value = f"{pred['expected_points']}/{pred.get('max_points', '?')}{stage}"
                print(terminal.stat("Expected points", value, "yellow"))
        elif pred.get("expected_points") is not None:
            print(terminal.stat("Expected points", str(pred["expected_points"]), "yellow"))

    if text:
        print()
        print(terminal.render(text, skip_banner=bool(predictions)))


def _format_args(args: dict) -> str:
    if not isinstance(args, dict):
        return ""
    parts = [f"{k}={v!r}" for k, v in args.items() if v not in (None, "")]
    return ", ".join(parts)


def print_stream(stream) -> None:
    tool_messages: list = []
    for step in stream:
        for node_name, update in step.items():
            if "messages" not in update:
                continue
            message = update["messages"][-1]
            if node_name == "tools":
                tool_messages.append(message)
                name = getattr(message, "name", "tool")
                print(terminal.step(f"{name} done", "ok"))
            elif node_name == "agent" and isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    for call in tool_calls:
                        label = f"{call.get('name', 'tool')}({_format_args(call.get('args', {}))})"
                        print(terminal.step(f"calling {label}", "run"))
                else:
                    print_terminal(message, _extract_predictions(tool_messages))


def run_query(query: str, *, stream: bool = True) -> None:
    """ This function get the settings, build the graph, and run the query"""

    settings = get_settings()
    # Build the graph
    app = build_graph(settings)
    # Create the input state
    input_state = {"messages": [HumanMessage(content=query)]}
    # Get the config
    config = settings.langgraph_invoke_config()

    log_event(logger, logging.INFO, "run_start", stream=stream, query_len=len(query))
    start = time.perf_counter()
    try:
        if stream:
            print_stream(
                app.stream(
                    input_state,
                    stream_mode=settings.langgraph_stream_mode,
                    config=config,
                )
            )
        else:
            # Invoke the graph and print the result
            result = app.invoke(input_state, config=config)
            print_terminal(
                result["messages"][-1], _extract_predictions(result["messages"])
            )
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000)
        log_event(logger, logging.INFO, "run_end", duration_ms=duration_ms)


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
    # If the user wants to print only the final answer instead of streaming node updates.
    # node update are useful for debugging and development, but not for the end user.
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print only the final answer instead of streaming node updates.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging for this run (overrides LOG_LEVEL).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings, verbose=args.verbose)

    try:
        run_query(args.query, stream=not args.no_stream)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    # Since the agent is a CLI, we use SystemExit to exit the program with a status code.
    raise SystemExit(main())
