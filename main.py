from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import get_settings

settings = get_settings()


class Agent8State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@tool
def subtract_numbers(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

tools = [add_numbers, subtract_numbers, multiply_numbers]

model = settings.create_chat_model(bind_tools=tools)


def model_call(state: Agent8State) -> Agent8State:
    system_prompt = SystemMessage(content=settings.agent_system_prompt)

    response = model.invoke([system_prompt] + state["messages"]) # concatenate the system prompt and the messages form user

    return {"messages": [response]}


def should_continue(state: Agent8State) -> Literal["end", "model_call"]:
    message = state["messages"]
    last_message = message[-1]

    if not last_message.tool_calls:
        return "end"

    return "model_call"

graph = StateGraph(Agent8State)
graph.add_node("agent", model_call)


tool_node = ToolNode(tools)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "end": END,
        "model_call": "tool",
    }
)

graph.add_edge("tool", "agent")

app = graph.compile()


input_message = {"messages": [HumanMessage(content="What is the sum of 1 and 99 multiplied by 3?")]}

def print_stream(stream):
    for s in stream:
        # Default stream_mode is "updates": {node_name: {state_delta}}
        for node_name, update in s.items():
            if "messages" not in update:
                continue
            message = update["messages"][-1]
            print(f"Update from node '{node_name}':")
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

print_stream(
    app.stream(
        input_message,
        stream_mode=settings.langgraph_stream_mode,
        config=settings.langgraph_invoke_config(),
    )
)