import json
import logging
import time
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

from config import Settings, get_settings
from logging_config import get_logger, log_event
from model_builder import OllamaModelBuilder
from tools import TOOLS

logger = get_logger("graph")


class FootballAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _tool_status(message: ToolMessage) -> str:
    """Return 'error' when a tool's JSON output has a top-level error, else 'ok'."""
    try:
        data = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return "ok"
    return "error" if isinstance(data, dict) and "error" in data else "ok"


def build_graph(
    settings: Settings | None = None,
    *,
    model: ChatOllama | None = None,
):
    """Compile the football prediction LangGraph agent."""
    settings = settings or get_settings()
    # Build the model
    model = OllamaModelBuilder(settings).build(tools=TOOLS)

    tool_node = ToolNode(TOOLS)

    # Define the agent node
    def agent_node(state: FootballAgentState) -> FootballAgentState:
        system = SystemMessage(content=settings.agent_system_prompt)
        log_event(
            logger, logging.DEBUG, "agent_invoke", messages=len(state["messages"])
        )
        start = time.perf_counter()
        response = model.invoke([system, *state["messages"]])
        duration_ms = round((time.perf_counter() - start) * 1000)
        tool_calls = getattr(response, "tool_calls", None)
        log_event(logger, logging.INFO, "agent_done", duration_ms=duration_ms)
        if tool_calls:
            log_event(
                logger,
                logging.INFO,
                "agent_tool_plan",
                tools=",".join(call.get("name", "?") for call in tool_calls),
            )
            for call in tool_calls:
                log_event(
                    logger,
                    logging.DEBUG,
                    "agent_tool_call",
                    tool=call.get("name", "?"),
                    args=call.get("args", {}),
                )
        else:
            log_event(
                logger,
                logging.INFO,
                "agent_final",
                content_len=len(getattr(response, "content", "") or ""),
            )
        return {"messages": [response]}

    # Wrap ToolNode to log each tool's name, args, latency, and status.
    def tools_node(
        state: FootballAgentState, config: RunnableConfig | None = None
    ) -> FootballAgentState:
        last = state["messages"][-1]
        for call in getattr(last, "tool_calls", None) or []:
            log_event(
                logger,
                logging.INFO,
                "tool_start",
                tool=call.get("name", "?"),
                args=call.get("args", {}),
            )
        start = time.perf_counter()
        out = tool_node.invoke(state, config)
        duration_ms = round((time.perf_counter() - start) * 1000)
        for message in out.get("messages", []):
            if isinstance(message, ToolMessage):
                log_event(
                    logger,
                    logging.INFO,
                    "tool_done",
                    tool=getattr(message, "name", "?"),
                    duration_ms=duration_ms,
                    status=_tool_status(message),
                )
        return out

    # Define the route node to route the messages to the tools or the end node
    def route(state: FootballAgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        decision = "tools" if tool_calls else "end"
        log_event(
            logger,
            logging.DEBUG,
            "route",
            decision=decision,
            tool_call_count=len(tool_calls or []),
        )
        return decision

    graph = StateGraph(FootballAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    # Add the conditional edges to route the messages to the tools or the end node
    graph.add_conditional_edges(
        "agent",
        route,
        {"tools": "tools", "end": END},
    )
    # Add the edge to the tools node to the agent node (return to the agent node) this is needed to complete the loop
    graph.add_edge("tools", "agent")
    # Compile the graph
    return graph.compile()
