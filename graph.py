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


def build_graph(
    settings: Settings | None = None,
    *,
    model: ChatOllama | None = None,
):
    """Compile the football prediction LangGraph agent."""
    settings = settings or get_settings()
    # Build the model
    model = OllamaModelBuilder(settings).build(tools=TOOLS)

    # Define the agent node
    def agent_node(state: FootballAgentState) -> FootballAgentState:
        system = SystemMessage(content=settings.agent_system_prompt)
        response = model.invoke([system, *state["messages"]])
        return {"messages": [response]}

    # Define the route node to route the messages to the tools or the end node
    def route(state: FootballAgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    graph = StateGraph(FootballAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
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
