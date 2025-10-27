from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the content with the provided content"""
    global document_content
    document_content = content
    return f"Document updated successfully\nCurrent content: {document_content}"


@tool
def save(filename: str) -> str:
    """Save the current content to a text file

    Arge:
    - filename for the txt file
    """
    if not (filename.endswith("txt")):
        filename = f"{filename}.txt"
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        print(f"Document has been saved to {filename}")
        return f"Operation successful. Document has been saved to {filename}"
    except Exception as e:
        return f"Process failed. Error {str(e)}"


tools = [update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def the_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """
    )
    if not state["messages"]:
        user_input = input(
            "I'm ready to help you update a document. What would you like to create? \n"
        )
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? \n")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation"""

    messages = state["messages"]

    if not messages:
        return "continue"

    last_message = messages[-1]

    if (
        isinstance(last_message, ToolMessage)
        and "saved" in last_message.content.lower()
        and "document" in last_message.content.lower()
    ):
        return "end"

    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("the_agent", the_agent)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.add_edge("the_agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "the_agent",
        "end": END,
    },
)
graph.set_entry_point("the_agent")
app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER =====")

    state = AgentState(messages=[])

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
