from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing import Optional

from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import create_agent

from typing import TypedDict, Any, Annotated, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_use import MCPAgent, MCPClient

import graphviz
import pydot
try:
    from IPython.display import display, Image, SVG  # for notebook rendering
except Exception:
    display = None
    Image = None
    SVG = None

'''
HELPFUL TIPS
1. If functions return a field matching a field from the BrainState class, that field is updated in the state. 
2. Good idea to include all fields in BrainState but not necessary. If a new field is returned, it can still be accessed in the state. 
'''


Role = Literal["planner", "toolcaller", "summarizer"]

class BrainState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_type: str | None
    next: str | None
    user_id: str | None

class ToolClassifier(BaseModel):
    tool_name: Literal["project_decomposition", "select_freelancer", "write_to_database", "read_from_database", "search_knowledge_base", "None"] = Field(
        ...,
        description="Classify the tool to be used based on the user's request. If no tool is needed, return 'None'."
    )


def get_llm(role: Role, provider: str = "openai"):
    if provider == "openai":
        model = "gpt-4o-mini" if role != "summarizer" else "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=0.2)
    if provider == "anthropic":
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)
    if provider == "vertex":
        # Use this path in Vertex deployments
        return ChatVertexAI(model="gemini-2.0-flash", temperature=0.2)
    raise ValueError("Unknown provider")

@tool
def get_account_info(state: BrainState, runtime: ToolRuntime) -> str:
    '''Get the current user's account information.'''
    
    return {"messages": [llm.invoke(state["messages"])]}

@tool
def clear_conversation(state: BrainState) -> Command:
    """Clear the conversation history."""

    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

@tool
def project_decomposition_tool(state: BrainState, runtime: ToolRuntime):
    '''
    Decompose the project into phases and tasks. 
    Args: 
        project_id: The id of the project.
        client_id: The id of the client.
        input_text: The input text of the project.
    '''
    # Add guardrails (rbac/idempotency) if needed
    return {"messages": [llm.invoke(state["messages"])]}

@tool
def select_freelancer_tool(state: BrainState, runtime: ToolRuntime,role: str, phase: list[str], top_k: int = 5):
    '''
    Select Freelancers from the database based on the role and required skills.
    Args:
        role: The role of the freelancer. If not provided, default is all roles.
        phase: The phase of the project. If not provided, default is all phases.
        top_k: The number of freelancer to select. If not provided, default is 5.
    '''
    return {"messages": [llm.invoke(state["messages"])]}
    #return adk_select_freelancer(role=role, skills=required_skills, run_id=run_id, top_k=top_k)

@tool
def write_to_database(table: str, data: dict):
    '''
    Write data to the database.
    Args:
        table: The table to write to.
        data: The data to write to the database.
    '''
    # Add guardrails (rbac/idempotency) if needed
    return {"messages": [llm.invoke(state["messages"])]}

@tool#("Read Database", description="Read data from the database.")
def read_from_database(table: str, query: str = None):
    '''
    Read data from the database.
    Args:
        table: The table to read from.
        query: The query to read from the database.
    '''
    # Add guardrails (rbac/idempotency) if needed
    return {"messages": [llm.invoke(state["messages"])]}

@tool
def search_knowledge_base(query: str):
    '''
    Search the knowledge base.
    Args:
        query: The query to search the knowledge base.
    '''
    # Add guardrails (rbac/idempotency) if needed
    return {"messages": [llm.invoke(state["messages"])]}

def planner_node(state: BrainState) -> BrainState:
    # Use a plain chat model and return an AI message to append to history.
        
    # Create configuration dictionary
    CONFIG = {
        "mcpServers": {
            "pactly-scoring": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-remote", "https://flashteams-selectfreelancer.onrender.com/mcp"]
            },
        }
    }

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_dict(CONFIG)

    # Create LLM
    llm = get_llm("planner", provider="vertex")

    # Create agent with the client
    agent = MCPAgent(llm=llm,client=client,max_steps=30)

    return {"messages": [agent.invoke(state["messages"])]}

def postprocess_node(state: BrainState) -> BrainState:
    # Pull structured outputs from tool calls in previous step (if you store them in state)
    # Or run deterministic business logic to build a proposal
    state["proposal"] = {"summary": "Draft proposal based on decomposition & selection"}
    return state

def tool_classifier_node(state: BrainState) -> BrainState:
    pass

def router(state: BrainState) -> BrainState:
    pass

def human_feedback(state: BrainState) -> BrainState:
    # TODO: Implement human feedback

    return {"messages": [llm.invoke(state["messages"])]}

def build_graph():
    g = StateGraph(BrainState)
    g.add_edge(START, "plan")
    g.add_node("plan", planner_node)
    #g.add_node("act", tool_node)
    #g.add_node("finalize", postprocess_node)
    #g.add_edge("plan", "act")
    #g.add_edge("act", "plan")
    #g.add_node("human_feedback", human_feedback)
    g.add_edge("plan", END)
    return g.compile()


if __name__ == "__main__": 
    graph = build_graph()

    #initial state
    #state = graph.invoke({"messages": [{"role": "user", "content": "My project is 'A inventory management system that warns when the stock is low based on future demand projections and automatically finds the cheapest supplier. '"}]})
    user_input = input("Enter your message: ")
    state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(state["messages"][-1].content)
    print(state)

