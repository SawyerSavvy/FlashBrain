from typing import Annotated, List, Literal, Optional, TypedDict, Union, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate

from prompts import GATEWAY_ROUTER_PROMPT, ORCHESTRATOR_PROMPT, RESPONSE_AGENT_PROMPT
# --- A2A Integration ---
from python_a2a import run_server, AgentCard, AgentSkill
from a2a_client_helper import call_project_decomp_agent, call_freelancer_agent, call_summarization_agent
import matplotlib.pyplot as plt



import os
import logging
logger = logging.getLogger(__name__)
# --- 1. State Definitions ---

class ReasoningMetadata(BaseModel):
    """The 'Reasoning Sidecar' - metadata explaining agent decisions."""
    reasoning: str = Field(..., description="Explanation of why this decision was made")
    alternatives_dropped: List[str] = Field(default_factory=list, description="Options considered but rejected")
    confidence_score: float = Field(0.0, description="Confidence in the decision (0-1)")

class AgentResponse(BaseModel):
    """Standard wrapper for all agent outputs."""
    status: Literal["success", "failed", "queued"]
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[ReasoningMetadata] = None
    job_id: Optional[str] = None # For async jobs

class GatewayRouterOutput(BaseModel):
    """Schema for the gateway router's output."""
    model_type: Literal["fast_model", "powerful_model"] = Field(..., description="The type of model to use for the next step.")
    next_step_category: Literal["responder", "orchestrator"] = Field(..., description="The category of the next step to take.")

class OrchestratorOutput(BaseModel):
    """Schema for the orchestrator's output."""
    next_step_route: Literal["planning_agent", "team_selection_agent", "finops_agent", "answer_directly"] = Field(..., description="The next agent to route to.")
    response: Optional[str] = Field(None, description="The direct response to the user, if applicable.")

class BrainState(TypedDict):
    """Core state for the FlashBrain system."""
    # Message history (Router adds messages to a list)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Context
    user_id: Optional[str]  # user_id is used for authentication and authorization
    project_id: Optional[str]  # project_id is the current project that the user is working on 
    
    # Internal routing state
    next_step: Optional[str]  # for routing to the next step
    previous_step: Optional[str]  # for looping back to the previous step
    selected_model_key: Optional[str] # Key for the selected model from MODEL_CLIENTS
    
    # Job tracking for async patterns
    active_jobs: Dict[str, str]  # job_id -> status
    
    # Most recent structured output from a sub-agent
    last_agent_response: Optional[Dict[str, Any]]
    
    # Token usage from the last model run
    last_token_usage: Optional[int]

class GeminiModel:
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.client = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    def call(self, prompt: Union[str, List[Union[str, Dict]]], **kwargs) -> str:
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        elif isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            # Assuming the list of dicts can be converted to BaseMessage
            messages = [HumanMessage(content=m["content"]) if m["type"] == "human" else AIMessage(content=m["content"]) for m in prompt] # Simplified conversion
        else:
            messages = prompt # Assume it's already a list of BaseMessage
            
        try:
            response = self.client.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            print(f"Error calling {self.model_name}: {e}")
            raise

# --- 3. Shared Resources (LLM) ---
# Initialize LLM clients once at module level to avoid re-init overhead.
# This centralizes model management and allows easy routing.
MODEL_CLIENTS: Dict[str, GeminiModel] = {}
try:
    MODEL_CLIENTS["routing"] = GeminiModel(model_name="gemini-2.5-flash-lite", temperature=0.0) # For gateway routing
    MODEL_CLIENTS["fast"] = GeminiModel(model_name="gemini-2.5-flash", temperature=0.0) # For basic tasks
    MODEL_CLIENTS["powerful"] = GeminiModel(model_name="gemini-3-pro-preview", temperature=0.1) # For complex orchestrator tasks
    print("All Gemini LLM clients initialized.")
except Exception as e:
    print(f"Error initializing one or more Gemini LLM clients: {e}. Some functionalities might be degraded.")

# --- 4. Nodes & Agents ---

# --- Level 0: Intelligent Gateway ---
def gateway_router(state: BrainState) -> Dict[str, Any]:
    """
    Lightweight classifier to route between simple Workflow and complex Orchestrator,
    and to select the appropriate model (fast or powerful) for subsequent steps.
    """
    last_msg_content = state["messages"][-1].content
    logger.info(f"DEBUG: gateway_router last_msg_content type: {type(last_msg_content)}, value: {last_msg_content}")

    # Default to fast model and orchestrator if routing model is not available
    model_choice = "fast"
    next_step_category = "orchestrator"
    token_usage = 0

    routing_model_client = MODEL_CLIENTS.get("routing")

    if routing_model_client:
        try:

            # Use .with_structured_output for reliable JSON parsing
            agent = routing_model_client.client.with_structured_output(GatewayRouterOutput)

            #Invoke agent with the system prompt and the user's message
            messages = [
                {"role": "system", "content": GATEWAY_ROUTER_PROMPT},
                {"role": "user", "content": last_msg_content}
            ]
            response = agent.invoke(messages)

            # Parse the response
            model_choice = response.model_type.replace("_model", "") # Convert to "fast" or "powerful"
            next_step_category = response.next_step_category
            
            # Calculate token usage (approximate since structured output hides metadata)
            # We reconstruct the messages to count tokens
            if hasattr(routing_model_client.client, "get_num_tokens_from_messages"):
                # Convert dict messages to BaseMessage for counting
                msgs_for_count = [SystemMessage(content=GATEWAY_ROUTER_PROMPT), HumanMessage(content=last_msg_content)]
                input_tokens = routing_model_client.client.get_num_tokens_from_messages(msgs_for_count)
                # Estimate output tokens from the Pydantic model (approx)
                output_tokens = len(str(response.model_dump())) // 4
                token_usage = input_tokens + output_tokens

            # Update the BrainState 
            return {
                "next_step": next_step_category,
                "previous_step": "gateway",
                "selected_model_key": model_choice,
                "last_token_usage": token_usage
            }

        except Exception as e:
            logger.error(f"FlashBrain:Gateway routing LLM execution failed ({e}), falling back to heuristic.")
    
    # Fallback heuristic for model choice (if routing model fails or is unavailable)
    if any(x in last_msg_content for x in ["plan", "analyze", "complex", "deep dive"]):
        model_choice = "powerful"
    else:
        model_choice = "fast" # Default to fast for basic tasks

    # Fallback heuristic for next step category (if routing model fails or is unavailable)
    if any(x in last_msg_content for x in ["mark", "create event", "update status"]):
        next_step_category = "responder"
    else:
        next_step_category = "orchestrator"

    return {
        "next_step": next_step_category,
        "previous_step": "gateway",
        "selected_model_key": model_choice,
        "last_token_usage": 0
    }

    # --- Level 1: FlashBrain Orchestrator ---
def orchestrator_node(state: BrainState) -> Dict[str, Any]:
    """
    The central brain. Decides which reasoning subsystem to call or answers the user,
    using the appropriate LLM based on the selected_model_key.
    """

    # Get the last message content
    last_msg_content = state["messages"][-1].content
    print(f"FlashBrain:orchestrator_node last_msg_content type: {type(last_msg_content)}, value: {last_msg_content}")

    # Use the selected model key to get the appropriate model client
    selected_model_key = state.get("selected_model_key", "powerful") # Orchestrator defaults to powerful
    orchestrator_model = MODEL_CLIENTS.get(selected_model_key)
    token_usage = 0

    if orchestrator_model:
        try:
            # Use .with_structured_output for reliable JSON parsing
            agent = orchestrator_model.client.with_structured_output(OrchestratorOutput)

            # Invoke agent
            messages = [
                {"role": "system", "content": ORCHESTRATOR_PROMPT},
                {"role": "user", "content": last_msg_content}
            ]
            response = agent.invoke(messages)

            # Parse the response
            next_step_route = response.next_step_route
            
            # Calculate token usage
            if hasattr(orchestrator_model.client, "get_num_tokens_from_messages"):
                msgs_for_count = [SystemMessage(content=ORCHESTRATOR_PROMPT), HumanMessage(content=last_msg_content)]
                input_tokens = orchestrator_model.client.get_num_tokens_from_messages(msgs_for_count)
                output_tokens = len(str(response.model_dump())) // 4
                token_usage = input_tokens + output_tokens

            # If the next step is to answer directly, return the response, else return the next step
            if next_step_route == "answer_directly":
                return {
                    "next_step": END,
                    "messages": [AIMessage(content=response.response or "I have processed your request.")],
                    "last_token_usage": token_usage
                }
            else:
                return {
                    "next_step": next_step_route,
                    "last_token_usage": token_usage
                }

        except Exception as e:
            print(f"Orchestrator LLM execution failed ({e}), falling back to heuristic. Error: {e}")

    # Fallback heuristic (if LLM fails or is unavailable)
    if "plan" in last_msg_content or "decompose" in last_msg_content:
        return {"next_step": "planning_agent"}
    elif "team" in last_msg_content or "hire" in last_msg_content:
        return {"next_step": "team_selection_agent"}
    elif "budget" in last_msg_content or "cost" in last_msg_content:
        return {"next_step": "finops_agent"}
    else:
        return {
            "next_step": END,
            "messages": [AIMessage(content="Error in orchestrator node. Please try again.")],
            "last_token_usage": 0
        }
    

def response_agent_node(state: BrainState) -> Dict[str, Any]:
    """
    Answers the user's query directly.
    """
    # Get the last message content
    last_msg_content = state["messages"][-1].content 

    # Use the selected model key to get the appropriate model client
    selected_model_key = state.get("selected_model_key", "fast") # Response agent defaults to powerful
    response_model = MODEL_CLIENTS.get(selected_model_key)
    token_usage = 0

    if response_model:
        try:
            # Invoke agent
            response = response_model.client.invoke([
                {"role": "system", "content": RESPONSE_AGENT_PROMPT},
                {"role": "user", "content": last_msg_content}
            ])
            
            # Extract token usage from metadata
            if response.usage_metadata:
                token_usage = response.usage_metadata.get("total_tokens", 0)

        except Exception as e:
            print(f"Response agent LLM execution failed ({e}), falling back to heuristic. Error: {e}")
            response = AIMessage(content="I'm sorry, I encountered an error.")
            
    return {
        "messages": [AIMessage(content=response.content)],
        "next_step": END,
        "last_token_usage": token_usage
    }

# --- Level 2: Reasoning Subsystems ---

# 2a. Planning Agent (Async Job Pattern)
def planning_agent_node(state: BrainState) -> Dict[str, Any]:
    """
    Calls the external Planning Agent via A2A protocol HTTP endpoint.
    """
    last_msg_content = state["messages"][-1].content
    agent_url = os.getenv("PROJECT_DECOMP_AGENT_URL", "http://localhost:8011")

    # Use helper function to call agent
    response_text = call_project_decomp_agent(
        agent_url=agent_url,
        message=last_msg_content,
        project_id=state.get("project_id"),
        client_id=state.get("user_id"),
        exist=False,
        context_id=state.get("user_id", "default")
    )

    return {
        "messages": [AIMessage(content=response_text)],
        "next_step": END
    }

# 2b. Team Selection Agent (Reasoning Sidecar)
def team_selection_node(state: BrainState) -> Dict[str, Any]:
    """
    Calls the external Select Freelancer Agent via A2A protocol HTTP endpoint.
    """
    last_msg_content = state["messages"][-1].content
    agent_url = os.getenv("SELECT_FREELANCER_AGENT_URL", "http://localhost:8012")

    # Use helper function to call agent
    response_text = call_freelancer_agent(
        agent_url=agent_url,
        message=last_msg_content,
        project_id=state.get("project_id"),
        context_id=state.get("user_id", "default")
    )

    # Create AgentResponse object for metadata
    agent_response = AgentResponse(
        status="success",
        data={"response": response_text},
        metadata=ReasoningMetadata(
            reasoning="Called external freelancer selection agent via A2A.",
            confidence_score=1.0
        )
    )

    return {
        "messages": [AIMessage(content=response_text)],
        "last_agent_response": agent_response.model_dump(),
        "next_step": END
    }

# 2c. FinOps Agent (Code Sandbox)  # TODO: Not sure what to do here yet, need to discuss with Nisi. 
def finops_node(state: BrainState) -> Dict[str, Any]:
    """
    Executes Python/SQL for math. Never uses raw LLM for calculation.
    """
    # Simulation: Run python calculation
    budget = 50000
    burn_rate = 5000
    runway = budget / burn_rate
    
    response = AgentResponse(
        status="success",
        data={"forecast_months": runway, "burn_rate": burn_rate},
        metadata=ReasoningMetadata(
            reasoning="Calculated based on current monthly burn rate from DB.",
            confidence_score=1.0
        )
    )
    
    msg = f"Based on the data, you have {runway} months of runway left."
    return {
        "messages": [AIMessage(content=msg)],
        "last_agent_response": response.model_dump(),
        "next_step": END
    }


# --- Level 3: Deterministic Workflow Node --- #TODO: Simple Supabase tools
def basic_tools_node(state: BrainState) -> Dict[str, Any]:
    """
    Executes simple deterministic actions via tools.
    """
    # In a real graph, this would call ToolNode(WORKFLOW_TOOLS)
    # Here we simulate a successful tool call.
    return {
        "messages": [AIMessage(content="Executed workflow action successfully.")],
        "next_step": END
    }

# --- Summarization Node ---
def summarize_conversation(state: BrainState) -> Dict[str, Any]:
    """
    Summarizes the conversation if the PREVIOUS model run exceeded a token limit.
    Calls the external Summarization Agent via A2A HTTP endpoint.
    """
    messages = state["messages"]
    last_token_usage = state.get("last_token_usage", 0)

    # Threshold for summarization (e.g., > 20000 tokens in the last run)
    if last_token_usage > 20000:
        # Keep the last 2 messages (user's latest input + potential system context)
        messages_to_summarize = messages[:-2]

        if not messages_to_summarize:
            return {}

        # Get agent URL from environment
        agent_url = os.getenv("SUMMARIZATION_AGENT_URL", "http://localhost:8013")

        # Serialize messages to dicts
        serialized_messages = []
        for m in messages:
            serialized_messages.append({
                "role": m.type,
                "content": m.content,
                "id": m.id if hasattr(m, "id") else None
            })

        # Call summarization agent using helper
        result = call_summarization_agent(
            agent_url=agent_url,
            messages=serialized_messages,
            keep_last=2,
            context_id=state.get("user_id", "default")
        )

        if result:
            print(f"Summarization completed: {result}")

            # Create a summary message placeholder
            summary_message = SystemMessage(content="[Previous conversation summarized]")

            # Remove old messages and add summary
            messages_to_remove = messages[:-2]
            delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove if hasattr(m, 'id')]

            return {"messages": [summary_message] + delete_messages}

    return {}

# --- 5. Graph Construction --- 
def build_flashbrain_graph(): 
    workflow = StateGraph(BrainState) 

    # Add Nodes
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("gateway", gateway_router)
    workflow.add_node("orchestrator", orchestrator_node)
    
    # Use the A2A wrapper node
    workflow.add_node("planning_agent", planning_agent_node)
    
    workflow.add_node("team_selection_agent", team_selection_node)
    workflow.add_node("finops_agent", finops_node)
    workflow.add_node("responder", response_agent_node)

    # Set Entry Point
    workflow.add_edge(START, "summarize_conversation")
    workflow.add_edge("summarize_conversation", "gateway")

    # Edges from Gateway
    workflow.add_conditional_edges(
        "gateway",
        lambda x: x["next_step"],
        {
            "orchestrator": "orchestrator",
            "responder": "responder"
        }
    )

    # Edges from Orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: x["next_step"],
        {
            "planning_agent": "planning_agent",
            "team_selection_agent": "team_selection_agent",
            "finops_agent": "finops_agent",
            END: END
        }
    )

    # Edges from Sub-Agents (return to END for now, could loop back to Orchestrator)
    workflow.add_edge("planning_agent", END)
    workflow.add_edge("team_selection_agent", END)
    workflow.add_edge("finops_agent", END)
    workflow.add_edge("responder", END)

    # Add Checkpointer for Memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

flashbrain_skill = AgentSkill(
    id='flashbrain_orchestrator',
    name="flashbrain_orchestrator",
    description="Orchestrates the entire FlashBrain system.",
    tags=["orchestrator", "brain"],
)

AgentCard(
    name="FlashBrain Orchestrator",
    description="The central orchestrator for the FlashBrain system.",
    url="http://localhost:8010",  # Required parameter
    version="0.1.0",
    capabilities={"streaming": True},
    skills=[flashbrain_skill],
)

class FlashBrainAgent:

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    async def stream(self, query: str, session_id: str = "default", user_id: str = None, project_id: str = None):
        """
        Streams responses from the FlashBrain Graph for A2A protocol.

        Yields:
            Dict with keys:
                - is_task_complete: bool
                - require_user_input: bool
                - content: str
        """
        from collections.abc import AsyncIterable

        app = build_flashbrain_graph()
        config = {"configurable": {"thread_id": session_id}}

        try:
            final_response = ""

            for event in app.stream(
                {"messages": [HumanMessage(content=query)], "user_id": user_id, "project_id": project_id},
                config=config
            ):
                logger.info(f"DEBUG: Stream query: {query}, type: {type(query)}")
                for key, value in event.items():
                    logger.info(f"DEBUG: Stream event key: {key}, value type: {type(value)}")
                    if value is None:
                         logger.info("DEBUG: Value is None!")
                         continue
                    if "messages" in value and value["messages"]:
                        last_msg = value["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            final_response = last_msg.content
                            # Yield intermediate updates
                            yield {
                                'is_task_complete': False,
                                'require_user_input': False,
                                'content': f"Processing: {key}..."
                            }

            # Final response
            if final_response:
                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': final_response
                }
            else:
                yield {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': "I need more information to process your request."
                }

        except Exception as e:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"An error occurred: {str(e)}"
            }

    def invoke(self, message: str, user_id: str = None, thread_id: str = "default", project_id: str = None) -> Dict[str, Any]:
        """
        Invokes the FlashBrain Graph.
        """
        # Handle JSON payload parsing if input is a string (A2A compat)
        if isinstance(message, str):
            try:
                import json
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    message = parsed.get("message", message)
                    user_id = parsed.get("user_id", user_id)
                    thread_id = parsed.get("thread_id", thread_id)
                    project_id = parsed.get("project_id", project_id)
            except:
                pass

        app = build_flashbrain_graph()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}} 
        
        # We need to collect the output since it streams
        final_response = ""
        full_state = {}
        
        for event in app.stream(
            {"messages": [HumanMessage(content=message)], "user_id": user_id, "project_id": project_id},
            config=config
        ):
            for key, value in event.items():
                if "messages" in value and value["messages"]:
                    last_msg = value["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        final_response = last_msg.content
                full_state.update(value)
                
        return {
            "response": final_response,
            "full_state": str(full_state) # Convert to string for safe serialization
        }
def test():
    agent = FlashBrainAgent()
    response = agent.invoke("Hello, how are you?")
    print(response)

if __name__ == "__main__":
    # Uncomment to run test
    # test()

    # Start A2A server
    port = int(os.getenv("PORT", 8010))
    print(f"Starting FlashBrain Orchestrator on port {port}...")
    run_server(FlashBrainAgent(), port=port)