from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

from config import Settings, get_settings
from model_builder import OllamaModelBuilder
from tools import TOOLS


class FootballAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_model(settings: Settings | None = None) -> ChatOllama:
    settings = settings or get_settings()
    return OllamaModelBuilder(settings).build(tools=TOOLS)


def build_graph(
    settings: Settings | None = None,
    *,
    model: ChatOllama | None = None,
):
    """Compile the football prediction LangGraph agent."""
    settings = settings or get_settings()
    model = model or build_model(settings)

    def agent_node(state: FootballAgentState) -> FootballAgentState:
        system = SystemMessage(content=settings.agent_system_prompt)
        response = model.invoke([system, *state["messages"]])
        return {"messages": [response]}

    def route(state: FootballAgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    graph = StateGraph(FootballAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
