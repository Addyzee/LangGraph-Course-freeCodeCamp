from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b


@tool
def subtract(a: int, b: int):
    """This is subtraction function that subtracts the first argument from the second"""
    return a - b


@tool
def multiply(a: int, b: int):
    """This is a multiplication function that multiplies 2 numbers"""
    return a * b


tools = [add, subtract, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are a helpful assistant. Please answer my query to the best of your ability"
    )
    response = model.invoke([system_prompt] + list(state["messages"]))
    return AgentState(messages=[response])


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent", should_continue, {"continue": "tools", "end": END}
)
graph.add_edge("tools", "our_agent")
app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = AgentState(messages=[HumanMessage("Add 40 and 12 aand subtract 5 from the answer")])
print_stream(app.stream(inputs, stream_mode="values"))
