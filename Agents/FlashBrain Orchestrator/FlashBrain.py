from typing import Annotated, List, Literal, Optional, TypedDict, Union, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import logging

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
import json

from prompts import GATEWAY_ROUTER_PROMPT, ORCHESTRATOR_PROMPT, RESPONSE_AGENT_PROMPT, INTENT_CLASSIFICATION_PROMPT
# --- A2A Integration ---
from a2a_client_helper import call_project_decomp_agent, call_freelancer_agent, call_summarization_agent

load_dotenv(override=True)

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
    next_step_category: Literal["orchestrator"] = Field(..., description="The category of the next step to take.")

class TaskIntent(BaseModel):
    """Schema for a single task in multi-intent requests."""
    type: Literal["PROJECT_DECOMP", "SELECT_FREELANCER", "FINOPS", "DIRECT_ANSWER"] = Field(..., description="The type of task to execute.")
    question: str = Field(..., description="The specific question or request for this task.")
    priority: int = Field(..., description="Priority level (lower number = higher priority).")

class OrchestratorOutput(BaseModel):
    """Schema for the orchestrator's output - supports both single and multi-intent."""
    # Single-intent fields
    next_step_route: Optional[Literal["planning_agent", "team_selection_agent", "finops_agent", "answer_directly"]] = Field(None, description="The next agent to route to (single-intent only).")
    response: Optional[str] = Field(None, description="The direct response to the user, if applicable.")

    # Multi-intent fields
    is_multi_intent: bool = Field(False, description="Whether multiple distinct tasks were detected.")
    tasks: Optional[List[TaskIntent]] = Field(None, description="List of tasks for multi-intent requests. Each task has: type, question, priority.")
    
class BrainState(TypedDict):
    """Core state for the FlashBrain system."""
    # Message history (Router adds messages to a list)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Context
    client_id: Optional[str]  # client_id is used for authentication and authorization
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

    # Short conversation history for the last 10 messages
    short_conversation_history: Optional[List[BaseMessage]]

    # UI messages
    ui_messages: Optional[List[BaseMessage]]
    
    # Multi-Intent Orchestration (v1)
    pending_tasks: Optional[List[Dict[str, Any]]]  # [{type, question, priority, status, result}, ...]
    current_task_idx: Optional[int]  # Index into pending_tasks for sequential execution
    aggregated_response: Optional[str]  # Final combined response from all tasks
    
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

class FlashBrainAgent:

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/json', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/markdown', 'image/jpeg', 'image/png', 'image/webp'] 

    def __init__(self, supabase_url: str = None, supabase_key: str = None, postgres_connection: str = None):
        """
        Initialize FlashBrain Agent with optional Supabase persistence.
        
        Args:
            supabase_url: Supabase project URL (or from SUPABASE_URL env)
            supabase_key: Supabase service role key (or from SUPABASE_SERVICE_ROLE_KEY env)
            postgres_connection: PostgreSQL connection string (or from SUPABASE_TRANSACTION_POOLER env)
        """
        logger.info("FlashBrainAgent initialized")
        
        # Supabase configuration
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.postgres_connection = postgres_connection or os.getenv("SUPABASE_TRANSACTION_POOLER")
        self.supabase_client = None
        
        # Initialize Supabase client for conversation history
        if self.supabase_url and self.supabase_key:
            try:
                from supabase import create_client
                self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
        
        # Initialize LLM clients
        self.model_clients: Dict[str, GeminiModel] = {}
        try:
            self.model_clients["routing"] = GeminiModel(model_name="gemini-2.5-flash-lite", temperature=0.0) # For gateway routing
            self.model_clients["fast"] = GeminiModel(model_name="gemini-2.5-flash", temperature=0.0) # For basic tasks
            self.model_clients["powerful"] = GeminiModel(model_name="gemini-2.5-pro", temperature=0.1) # For complex orchestrator tasks
            logger.info("All Gemini LLM clients initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini clients: {e}")
            raise
        
        # Build the graph (checkpointer will be initialized here)
        # Use asyncio.run since we're in sync __init__ but need async _build_graph
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # We're already in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.graph = executor.submit(lambda: loop.run_until_complete(self._build_graph())).result()
        else:
            self.graph = loop.run_until_complete(self._build_graph())

    async def _build_graph(self):
        workflow = StateGraph(BrainState) 

        # Add Nodes
        workflow.add_node("summarize_conversation", self.summarize_conversation)
        workflow.add_node("gateway", self.gateway_router)
        workflow.add_node("orchestrator", self.orchestrator_node)
        
        # Sub-agent nodes
        workflow.add_node("planning_agent", self.planning_agent_node)
        workflow.add_node("team_selection_agent", self.team_selection_node)
        workflow.add_node("finops_agent", self.finops_node)
        workflow.add_node("responder", self.response_agent_node)

        # Multi-Intent Orchestration nodes (task_executor and response_aggregator only)
        workflow.add_node("task_executor", self.task_executor_node)
        workflow.add_node("response_aggregator", self.response_aggregator_node)

        # Set Entry Point
        workflow.add_edge(START, "summarize_conversation")
        workflow.add_edge("summarize_conversation", "gateway")

        # Routing Functions
        def route_from_gateway(state: BrainState) -> str:
            """Route from gateway based on next_step."""
            next_step = state.get("next_step", "orchestrator")
            routing_map = {
                "orchestrator": "orchestrator",
                "answer_directly": "responder",
                "planning_agent": "planning_agent",
                "team_selection_agent": "team_selection_agent",
            }
            return routing_map.get(next_step, "orchestrator")

        def route_from_orchestrator(state: BrainState) -> str:
            """Route from orchestrator based on single-intent or multi-intent detection."""
            # Check if multi-intent was detected by orchestrator
            if state.get("pending_tasks"):
                # Multi-intent: go to task_executor
                return "task_executor"

            # Single-intent: route to appropriate agent
            next_step = state.get("next_step", END)
            routing_map = {
                "planning_agent": "planning_agent",
                "team_selection_agent": "team_selection_agent",
                "finops_agent": "finops_agent",
                "answer_directly": "responder",
            }
            return routing_map.get(next_step, END)
        
        def route_from_task_executor(state: BrainState) -> str:
            """Route to appropriate sub-agent or aggregator."""
            next_step = state.get("next_step")
            if next_step == "response_aggregator":
                return "response_aggregator"
            # Route to sub-agents
            routing_map = {
                "planning_agent": "planning_agent",
                "team_selection_agent": "team_selection_agent",
                "finops_agent": "finops_agent",
                "responder": "responder"
            }
            return routing_map.get(next_step, "response_aggregator")
        
        def route_from_subagents(state: BrainState) -> str:
            """Sub-agents either loop back to task_executor or go to END."""
            next_step = state.get("next_step")
            if next_step == "task_executor":
                return "task_executor"
            return END

        # Edge Configuration
        workflow.add_conditional_edges(
            "gateway",
            route_from_gateway,
            {
                "orchestrator": "orchestrator",
                "responder": "responder",
                "planning_agent": "planning_agent",
                "team_selection_agent": "team_selection_agent",
            }
        )

        # Orchestrator routes based on single vs multi-intent
        workflow.add_conditional_edges(
            "orchestrator",
            route_from_orchestrator,
            {
                "task_executor": "task_executor",
                "planning_agent": "planning_agent",
                "team_selection_agent": "team_selection_agent",
                "finops_agent": "finops_agent",
                "responder": "responder",
                END: END
            }
        )
        
        # Task executor routes to sub-agents or aggregator
        workflow.add_conditional_edges(
            "task_executor",
            route_from_task_executor,
            {
                "planning_agent": "planning_agent",
                "team_selection_agent": "team_selection_agent",
                "finops_agent": "finops_agent",
                "responder": "responder",
                "response_aggregator": "response_aggregator"
            }
        )
        
        # Sub-agents loop back to task_executor or END
        workflow.add_conditional_edges("planning_agent", route_from_subagents, 
                                      {"task_executor": "task_executor", END: END})
        workflow.add_conditional_edges("team_selection_agent", route_from_subagents,
                                      {"task_executor": "task_executor", END: END})
        workflow.add_conditional_edges("finops_agent", route_from_subagents,
                                      {"task_executor": "task_executor", END: END})
        workflow.add_conditional_edges("responder", route_from_subagents,
                                      {"task_executor": "task_executor", END: END})
        
        # Aggregator ends
        workflow.add_edge("response_aggregator", END)

        # Add Checkpointer for Memory - Use PostgreSQL for persistence (REQUIRED for production)
        if not self.postgres_connection:
            raise RuntimeError(
                "SUPABASE_TRANSACTION_POOLER environment variable is required for production. "
                "FlashBrain cannot run without persistent state storage."
            )

        try:
            # Use connection pool approach with autocommit to avoid transaction issues
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                conninfo=self.postgres_connection,
                max_size=20,
                min_size=1,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": None,
                    "connect_timeout": 10,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                },
                open=False,
                timeout=30.0,  # Pool checkout timeout
                max_waiting=10,  # Max clients waiting for a connection
                max_idle=300.0,  # Max time a connection can stay idle (5 min)
                reconnect_timeout=60.0,  # Time to wait before reconnecting
            )

            await pool.open()  # Explicitly open the pool

            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()  # Create tables (works with autocommit)
            logger.info("AsyncPostgreSQL checkpointer initialized successfully")

        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize PostgreSQL checkpointer: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to initialize PostgreSQL checkpointer. FlashBrain cannot start without persistent state. "
                f"Error: {e}"
            ) from e

        return workflow.compile(checkpointer=checkpointer)
    

    async def load_conversation_history(self, conversation_id: str, limit: int = 50) -> List[BaseMessage]:
        """Load conversation history from Supabase brain_messages table."""
        if not self.supabase_client:
            return []

        try:
            response = self.supabase_client.table("brain_messages")\
                .select("role, content, metadata, created_at")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=False)\
                .limit(limit)\
                .execute()

            messages = []
            for msg in response.data:
                if msg["role"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))

            return messages
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            return []
    
    async def save_message(self, conversation_id: str, role: str, content: str, metadata: dict = None):
        """Save a message to Supabase brain_messages table."""
        if not self.supabase_client:
            return

        try:
            self.supabase_client.table("brain_messages").insert({
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {}
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
    
    async def ensure_conversation_exists(self, conversation_id: str, user_id: str = None):
        """Ensure a conversation record exists in the database."""
        if not self.supabase_client:
            return

        try:
            # Check if conversation exists
            response = self.supabase_client.table("brain_conversations")\
                .select("id")\
                .eq("id", conversation_id)\
                .execute()

            if not response.data:
                # Create new conversation
                self.supabase_client.table("brain_conversations").insert({
                    "id": conversation_id,
                    "user_id": user_id or "anonymous",
                    "title": "New Conversation"
                }).execute()
        except Exception as e:
            logger.error(f"Failed to ensure conversation exists: {e}")

    # --- Level 1: Gateway ---
    
    def filter_meaningful_messages(self, messages: List[BaseMessage], max_count: int = 10) -> List[BaseMessage]:
        """
        Filter messages to keep only meaningful conversation.
        Excludes:
        - SystemMessages
        - Short AI placeholder messages (e.g., "Processing Conversation...", "Processing: xxx...")
        
        Returns the last max_count meaningful messages.
        """
        meaningful = []
        
        for msg in messages:
            # Skip system messages
            if isinstance(msg, SystemMessage):
                continue
            
            meaningful.append(msg)
        
        # Return the last max_count messages
        return meaningful[-max_count:] if len(meaningful) > max_count else meaningful

    def gateway_router(self, state: BrainState) -> Dict[str, Any]:
        """
        Lightweight classifier to route between simple Workflow and complex Orchestrator,
        and to select the appropriate model (fast or powerful) for subsequent steps.
        
        Function:
        1. Analyzes conversation history to determine complexity.
        2. Routes simple queries to 'responder' (fast model).
        3. Routes complex queries to 'orchestrator' (powerful model).
        4. Calculates token usage for cost tracking.
        """
        messages = state["messages"]
        
        # Filter to get only meaningful messages (last 10, excluding system and placeholders)
        recent_messages = self.filter_meaningful_messages(messages, max_count=10)
        
        # Default to fast model and orchestrator if routing model is not available
        model_choice = "fast"
        next_step_category = "orchestrator"
        token_usage = 0

        routing_model_client = self.model_clients.get("routing")

        if routing_model_client and messages:
            try:
                # Use .with_structured_output for reliable JSON parsing
                agent = routing_model_client.client.with_structured_output(GatewayRouterOutput)

                # Convert state messages to LLM format, sending full conversation history
                llm_messages = [{"role": "system", "content": GATEWAY_ROUTER_PROMPT}]
            
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        llm_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        llm_messages.append({"role": "assistant", "content": msg.content})
                
                response = agent.invoke(llm_messages) #TODO: streaming for LLMs

                # Parse the response
                model_choice = response.model_type.replace("_model", "")  # Convert to "fast" or "powerful"
                next_step_category = response.next_step_category
                
                # Calculate token usage (approximate since structured output hides metadata)
                if hasattr(routing_model_client.client, "get_num_tokens_from_messages"):
                    # Convert dict messages to BaseMessage for counting
                    msgs_for_count = [SystemMessage(content=GATEWAY_ROUTER_PROMPT)] + list(recent_messages)
                    input_tokens = routing_model_client.client.get_num_tokens_from_messages(msgs_for_count)
                    # Estimate output tokens from the Pydantic model (approx)
                    output_tokens = len(str(response.model_dump())) // 4
                    token_usage = input_tokens + output_tokens

                # Update the BrainState 
                return {
                    "next_step": next_step_category,
                    "previous_step": "gateway",
                    "selected_model_key": model_choice,
                    "last_token_usage": token_usage,
                    "short_conversation_history": recent_messages,  # Save filtered messages
                    "ui_messages": [AIMessage(content="🎯 Routing your request...")]
                }

            except Exception as e:
                logger.error(f"FlashBrain:Gateway routing LLM execution failed ({e}), falling back to heuristic.")
        
        return {
            "next_step": END,
            "previous_step": "gateway",
            "last_token_usage": 0
        }

    # --- Level 1: FlashBrain Orchestrator ---
    def orchestrator_node(self, state: BrainState) -> Dict[str, Any]:
        """
        The central brain with integrated intent planning.
        Decides which reasoning subsystem to call AND detects multi-intent requests.

        Function:
        1. Uses a powerful LLM (Gemini Pro) to analyze the user's request.
        2. Detects if multiple distinct tasks exist (multi-intent).
        3. For single-intent: routes to appropriate agent (planning_agent, team_selection_agent, etc.)
        4. For multi-intent: creates pending_tasks list and routes to task_executor.
        """

        # Use pre-filtered messages from gateway if available, otherwise filter now
        recent_messages = state.get("short_conversation_history")
        if not recent_messages:
            messages = state["messages"]
            recent_messages = self.filter_meaningful_messages(messages, max_count=10)

        # Use the selected model key to get the appropriate model client
        selected_model_key = state.get("selected_model_key", "powerful") # Orchestrator defaults to powerful
        orchestrator_model = self.model_clients.get(selected_model_key)

        if orchestrator_model:
            try:
                # Use .with_structured_output for reliable JSON parsing
                agent = orchestrator_model.client.with_structured_output(OrchestratorOutput)

                # Build message list with system prompt and conversation history
                llm_messages = [{"role": "system", "content": ORCHESTRATOR_PROMPT}]

                # Add recent conversation history
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        llm_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        llm_messages.append({"role": "assistant", "content": msg.content})

                response = agent.invoke(llm_messages)

                # Validate response
                if response is None:
                    logger.error("LLM returned None response")
                    raise ValueError("LLM returned None response")

                logger.info(f"Orchestrator response: is_multi_intent={response.is_multi_intent}, next_step={response.next_step_route}, tasks={len(response.tasks) if response.tasks else 0}")

                # Calculate token usage
                if hasattr(orchestrator_model.client, "get_num_tokens_from_messages"):
                    msgs_for_count = [SystemMessage(content=ORCHESTRATOR_PROMPT)] + list(recent_messages)
                    input_tokens = orchestrator_model.client.get_num_tokens_from_messages(msgs_for_count)
                    output_tokens = len(str(response.model_dump())) // 4
                    token_usage = input_tokens + output_tokens
                else:
                    token_usage = 0

                # Check if multi-intent was detected
                if response.is_multi_intent and response.tasks:
                    # Multi-intent path: prepare tasks for execution
                    # Convert TaskIntent objects to dicts for state storage
                    tasks = [task.model_dump() for task in response.tasks]
                    logger.info(f"Multi-intent detected. Tasks from LLM: {tasks}")

                    # Sort tasks by priority (lower number = higher priority)
                    tasks_sorted = sorted(tasks, key=lambda x: x.get("priority", 999))

                    # Initialize task tracking (fields already validated by Pydantic)
                    for task in tasks_sorted:
                        task["status"] = "pending"
                        task["result"] = None

                    return {
                        "pending_tasks": tasks_sorted,
                        "current_task_idx": 0,
                        "last_token_usage": token_usage,
                        "previous_step": "orchestrator",
                        "ui_messages": [AIMessage(content=f"📋 Detected {len(tasks_sorted)} tasks to complete...")]
                    }
                else:
                    # Single-intent path: route to appropriate agent
                    next_step = response.next_step_route

                    return {
                        "next_step": next_step,
                        "last_token_usage": token_usage,
                        "previous_step": "orchestrator",
                        "ui_messages": [AIMessage(content="🧠 Determining next step...")]
                    }

            except Exception as e:
                logger.error(f"Orchestrator LLM execution failed ({e}), falling back to heuristic. Error: {e}")

        # Fallback heuristic (if LLM fails or is unavailable)
        return {
            "next_step": END,
            "messages": [AIMessage(content="Error. Please try again.")],
            "last_token_usage": 0,
            "previous_step": "orchestrator"
        }

    # --- Multi-Intent Orchestration Nodes ---

    def task_executor_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Phase 2: Sequential Task Execution
        Routes to the appropriate sub-agent based on the current task.

        Function:
        1. Checks `current_task_idx` against `pending_tasks`.
        2. If tasks remain, retrieves the current task type.
        3. Maps task type to the correct agent node (e.g., PROJECT_DECOMP -> planning_agent).
        4. Updates `next_step` to route to that agent.
        5. If all tasks done, routes to `response_aggregator`.
        """
        pending_tasks = state.get("pending_tasks", [])
        current_idx = state.get("current_task_idx", 0)

        logger.info(f"[TASK_EXECUTOR] Current index: {current_idx}, Total tasks: {len(pending_tasks)}")

        # Check if all tasks are complete
        if current_idx >= len(pending_tasks):
            logger.info(f"[TASK_EXECUTOR] All tasks complete. Routing to response_aggregator")
            return {"next_step": "response_aggregator"}

        # Get current task
        current_task = pending_tasks[current_idx]
        logger.info(f"[TASK_EXECUTOR] Current task: {current_task}")

        task_type = current_task.get("type")
        task_question = current_task.get("question", "Unknown")

        # Validate task has required fields
        if not task_type:
            logger.error(f"[TASK_EXECUTOR] Task missing 'type' field: {current_task}")
            raise ValueError(f"Task {current_idx} missing 'type' field")
        if not current_task.get("question"):
            logger.error(f"[TASK_EXECUTOR] Task missing 'question' field: {current_task}")
            raise ValueError(f"Task {current_idx} missing 'question' field")

        # Route based on task type
        route_map = {
            "PROJECT_DECOMP": "planning_agent",
            "SELECT_FREELANCER": "team_selection_agent",
            "FINOPS": "finops_agent",
            "DIRECT_ANSWER": "responder"
        }

        next_step = route_map.get(task_type, "responder")
        logger.info(f"[TASK_EXECUTOR] Routing task {current_idx + 1}/{len(pending_tasks)} (type={task_type}) to {next_step}")

        return {
            "next_step": next_step,
            "ui_messages": [AIMessage(content=f"⚙️ Processing task {current_idx + 1}/{len(pending_tasks)}: {task_question}")]
        }

    def response_aggregator_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Phase 3: Response Aggregation
        Combines results from all completed tasks into a unified response.
        
        Function:
        1. Collects results from all items in `pending_tasks`.
        2. Formats them into a single readable message (Task 1: ... Result: ...).
        3. Updates `aggregated_response` in state.
        4. Returns the final combined message to the user.
        """
        pending_tasks = state.get("pending_tasks", [])

        if not pending_tasks:
            return {
                "messages": [AIMessage(content="No tasks to aggregate.")],
                "next_step": END
            }

        # Simple concatenation (v1 approach)
        aggregated_parts = []
        for i, task in enumerate(pending_tasks):
            question = task.get("question", "Unknown task")
            result = task.get("result", "No result available")

            aggregated_parts.append(f"**Task {i + 1}**: {question}\n\n{result}")

        aggregated_response = "\n\n---\n\n".join(aggregated_parts)

        return {
            "messages": [AIMessage(content=aggregated_response)],
            "aggregated_response": aggregated_response,
            "next_step": END,
            "previous_step": "response_aggregator"
        }

    def response_agent_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Answers the user's query directly with streaming output.
        Enhanced for multi-intent support.

        Function:
        1. Acts as a general-purpose chatbot for direct questions.
        2. Supports both single-intent (uses conversation history) and multi-intent (uses specific task question).
        3. Streams the LLM response back to the user.
        4. Stores the result in `pending_tasks` if in multi-intent mode.
        """
        pending_tasks = state.get("pending_tasks")
        current_idx = state.get("current_task_idx")

        logger.info(f"[RESPONDER] Multi-intent mode: {pending_tasks is not None and current_idx is not None}")

        # For multi-intent, use task question directly
        if pending_tasks and current_idx is not None:
            logger.info(f"[RESPONDER] Multi-intent: Processing task {current_idx + 1}/{len(pending_tasks)}")
            try:
                user_question = pending_tasks[current_idx]["question"]
                logger.info(f"[RESPONDER] Task question: {user_question}")
            except KeyError as e:
                logger.error(f"[RESPONDER] Task missing 'question' field. Task: {pending_tasks[current_idx]}")
                raise ValueError(f"Task {current_idx} missing 'question' field") from e

            # Create simple message list for direct answer
            llm_messages = [
                {"role": "system", "content": RESPONSE_AGENT_PROMPT},
                {"role": "user", "content": user_question}
            ]
        else:
            # Single-intent: use full conversation history
            recent_messages = state.get("short_conversation_history")
            if not recent_messages:
                messages = state["messages"]
                recent_messages = self.filter_meaningful_messages(messages, max_count=10)
            
            llm_messages = [{"role": "system", "content": RESPONSE_AGENT_PROMPT}]
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    llm_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    llm_messages.append({"role": "assistant", "content": msg.content})

        # Use the selected model key to get the appropriate model client
        selected_model_key = state.get("selected_model_key", "fast")
        response_model = self.model_clients.get(selected_model_key)

        if response_model:
            try:
                # Stream response
                full_response = ""
                for chunk in response_model.client.stream(llm_messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content

                # Multi-intent: store result and advance
                if pending_tasks and current_idx is not None:
                    pending_tasks[current_idx]["status"] = "completed"
                    pending_tasks[current_idx]["result"] = full_response
                    
                    return {
                        "pending_tasks": pending_tasks,
                        "current_task_idx": current_idx + 1,
                        "next_step": "task_executor"
                    }
                
                # Single-intent: return normally
                return {
                    "messages": [AIMessage(content=full_response)],
                    "next_step": END,
                    "previous_step": "response_agent"
                }

            except Exception as e:
                logger.error(f"Response agent LLM execution failed ({e}), falling back to error message.")
                error_msg = "I'm sorry, I encountered an error. Please try again."
                
                if pending_tasks and current_idx is not None:
                    pending_tasks[current_idx]["status"] = "completed"
                    pending_tasks[current_idx]["result"] = error_msg
                    return {
                        "pending_tasks": pending_tasks,
                        "current_task_idx": current_idx + 1,
                        "next_step": "task_executor"
                    }
                
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "next_step": END,
                    "previous_step": "response_agent"
                }
        
        # Fallback if no model available
        error_msg = "Response model not available."
        if pending_tasks and current_idx is not None:
            pending_tasks[current_idx]["status"] = "completed"
            pending_tasks[current_idx]["result"] = error_msg
            return {
                "pending_tasks": pending_tasks,
                "current_task_idx": current_idx + 1,
                "next_step": "task_executor"
            }
        
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_step": END,
            "previous_step": "response_agent"
        }

    # --- Level 2: Reasoning Subsystems ---

    # 2a. Planning Agent (Async Job Pattern)
    async def planning_agent_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Calls the external Planning Agent via A2A protocol HTTP endpoint.
        Enhanced for multi-intent support.

        Function:
        1. Connects to the Project Decomposition Agent service.
        2. Sends the user's request (or specific task question) to decompose a project.
        3. Streams the agent's progress and final result.
        4. Stores the result in `pending_tasks` if in multi-intent mode.
        """
        pending_tasks = state.get("pending_tasks")
        current_idx = state.get("current_task_idx")

        logger.info(f"[PLANNING_AGENT] Multi-intent mode: {pending_tasks is not None and current_idx is not None}")

        # Determine input message
        if pending_tasks and current_idx is not None:
            # Multi-intent mode: use current task's question
            logger.info(f"[PLANNING_AGENT] Multi-intent: Processing task {current_idx + 1}/{len(pending_tasks)}")
            try:
                message = pending_tasks[current_idx]["question"]
                logger.info(f"[PLANNING_AGENT] Task question: {message}")
            except KeyError as e:
                logger.error(f"[PLANNING_AGENT] Task missing 'question' field. Task: {pending_tasks[current_idx]}")
                raise ValueError(f"Task {current_idx} missing 'question' field") from e
        else:
            # Single-intent mode: use last message
            message = state["messages"][-1].content
            logger.info(f"[PLANNING_AGENT] Single-intent mode. Message: {message}")
        
        agent_url = os.getenv("PROJECT_DECOMP_AGENT_URL", "http://localhost:8011")

        # Collect all streaming updates from the A2A agent
        all_updates = []
        all_updates.append("🔄 Processing with Planning Agent...")
        
        async for response_text in call_project_decomp_agent(
            agent_url=agent_url,
            message=message,
            project_id=state.get("project_id"),
            client_id=state.get("client_id"),
            exist=False,
            context_id=state.get("client_id", "default")
        ):
            all_updates.append(response_text)
            final_response = response_text

        # Multi-intent: store result and advance
        if pending_tasks and current_idx is not None:
            pending_tasks[current_idx]["status"] = "completed"
            pending_tasks[current_idx]["result"] = final_response
            
            return {
                "pending_tasks": pending_tasks,
                "current_task_idx": current_idx + 1,
                "next_step": "task_executor"  # Loop back
            }
        
        # Single-intent: return normally
        combined_message = "\n".join(all_updates)
        return {
            "messages": [AIMessage(content=combined_message)],
            "next_step": END
        }

    # 2b. Team Selection Agent (Reasoning Sidecar)
    async def team_selection_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Calls the external Select Freelancer Agent via A2A protocol HTTP endpoint.
        Enhanced for multi-intent support.
        
        Function:
        1. Connects to the Select Freelancer Agent service.
        2. Sends the request to find freelancers matching project requirements.
        3. Streams the search process and results.
        4. Stores the result in `pending_tasks` if in multi-intent mode.
        """
        pending_tasks = state.get("pending_tasks")
        current_idx = state.get("current_task_idx")
        
        # Determine input message
        if pending_tasks and current_idx is not None:
            message = pending_tasks[current_idx]["question"]
        else:
            message = state["messages"][-1].content
        
        agent_url = os.getenv("SELECT_FREELANCER_AGENT_URL", "http://localhost:8012")

        # Collect all streaming updates from the A2A agent
        all_updates = []
        all_updates.append("🔍 Searching for freelancers...")
        
        async for response_text in call_freelancer_agent(
            agent_url=agent_url,
            message=message,
            project_id=state.get("project_id"),
            client_id=state.get("client_id", "default")
        ):
            all_updates.append(response_text)
            final_response = response_text

        # Multi-intent: store result and advance
        if pending_tasks and current_idx is not None:
            pending_tasks[current_idx]["status"] = "completed"
            pending_tasks[current_idx]["result"] = final_response
            
            return {
                "pending_tasks": pending_tasks,
                "current_task_idx": current_idx + 1,
                "next_step": "task_executor"
            }
        
        # Single-intent: create AgentResponse object for metadata
        agent_response = AgentResponse(
            status="success",
            data={"response": final_response},
            metadata=ReasoningMetadata(
                reasoning="Called external freelancer selection agent via A2A.",
                confidence_score=1.0
            )
        )

        # Return all updates as a single message
        combined_message = "\n".join(all_updates)
        return {
            "messages": [AIMessage(content=combined_message)],
            "last_agent_response": agent_response.model_dump(),
            "next_step": END
        }

    # 2c. FinOps Agent (Code Sandbox)
    def finops_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Executes Python/SQL for math. Never uses raw LLM for calculation.
        Enhanced for multi-intent support.
        
        Function:
        1. Performs deterministic financial calculations (budget, runway, burn rate).
        2. Currently a simulation/placeholder for demonstration.
        3. Returns structured financial data.
        """
        pending_tasks = state.get("pending_tasks")
        current_idx = state.get("current_task_idx")
        
        # Simulation: Run python calculation
        budget = 50000
        burn_rate = 5000
        runway = budget / burn_rate
        
        msg = f"Based on the data, you have {runway} months of runway left."
        
        # Multi-intent: store result and advance
        if pending_tasks and current_idx is not None:
            pending_tasks[current_idx]["status"] = "completed"
            pending_tasks[current_idx]["result"] = msg
            
            return {
                "pending_tasks": pending_tasks,
                "current_task_idx": current_idx + 1,
                "next_step": "task_executor"
            }
        
        # Single-intent: return normally
        response = AgentResponse(
            status="success",
            data={"forecast_months": runway, "burn_rate": burn_rate},
            metadata=ReasoningMetadata(
                reasoning="Calculated based on current monthly burn rate from DB.",
                confidence_score=1.0
            )
        )
        
        return {
            "messages": [AIMessage(content=msg)],
            "last_agent_response": response.model_dump(),
            "next_step": END
        }


    # --- Level 3: Deterministic Workflow Node --- #TODO: Simple Supabase tools
    def basic_tools_node(self, state: BrainState) -> Dict[str, Any]:
        """
        Executes simple deterministic actions via tools.
        
        Function:
        1. Placeholder for future tool execution node.
        2. Would handle simple, non-agentic tools (e.g., search, calculator).
        """
        # In a real graph, this would call ToolNode(WORKFLOW_TOOLS)
        # Here we simulate a successful tool call.
        return {
            "messages": [AIMessage(content="Executed workflow action successfully.")],
            "next_step": END
        }

    # --- Summarization Node ---
    def summarize_conversation(self, state: BrainState) -> Dict[str, Any]:
        """
        Summarizes the conversation if the PREVIOUS model run exceeded a token limit.
        Calls the external Summarization Agent via A2A HTTP endpoint.
        
        Function:
        1. Checks `last_token_usage` against a threshold (20k tokens).
        2. If exceeded, calls the Summarization Agent to condense history.
        3. Replaces old messages with a summary to free up context window.
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
                client_id=state.get("client_id", "default")
            )

            if result:
                print(f"Summarization completed: {result}")

                # Extract the summarized messages from the A2A response
                from a2a_client_helper import extract_a2a_response_text
                import json
                
                try:
                    # Get the text content from the A2A response
                    response_text = extract_a2a_response_text(result)
                    
                    # Parse the JSON to get the list of summarized message dicts
                    summarized_message_dicts = json.loads(response_text)
                    
                    # Convert the message dicts back to BaseMessage objects
                    summarized_messages = []
                    for msg_dict in summarized_message_dicts:
                        msg_type = msg_dict.get("type", "human")
                        content = msg_dict.get("data", {}).get("content", msg_dict.get("content", ""))
                        
                        if msg_type == "ai":
                            summarized_messages.append(AIMessage(content=content))
                        else:  # human or default
                            summarized_messages.append(HumanMessage(content=content))
                    
                    # Remove old messages and add the summarized ones
                    messages_to_remove = messages[:-2]
                    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove if hasattr(m, 'id')]
                    
                    # Return the summarized messages plus delete operations
                    return {"messages": summarized_messages + delete_messages}
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"WARNING: Failed to parse summarized messages: {e}. Using placeholder.")
                    
                    # Fallback to placeholder if parsing fails
                    #summary_message = SystemMessage(content="[Previous conversation summarized]")
                    messages_to_remove = messages[:-2]
                    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove if hasattr(m, 'id')]
                    return {"messages": delete_messages}

        return {'ui_messages': [AIMessage(content="Processing Conversation...")]}

    async def stream(self, query: str, context_id: str = "default", client_id: str = None, project_id: str = None):
        """
        Streams responses from the FlashBrain Graph for A2A protocol.

        Yields:
            Dict with keys:
                - is_task_complete: bool
                - require_user_input: bool
                - content: str
        """
        config = {"configurable": {"thread_id": context_id}}

        try:
            # Ensure conversation exists in Supabase
            await self.ensure_conversation_exists(context_id, client_id)
            
            # Save user message to Supabase
            await self.save_message(
                conversation_id=context_id,
                role="human",
                content=query,
                metadata={"client_id": client_id, "project_id": project_id}
            )
            
            final_response = ""

            # Use 'updates' mode to capture state changes including ui_messages
            # subgraphs=True ensures we get updates from A2A agents too
            async for event in self.graph.astream(
                {"messages": [HumanMessage(content=query)], "client_id": client_id, "project_id": project_id},
                config=config, 
                stream_mode='updates',
                subgraphs=True
            ):
                # In updates mode, events are tuples: (namespace, node_updates)
                # namespace is () for main graph or ('node_name:task_id',) for subgraphs
                # node_updates is a dict with the node name as key and state updates as value
                
                if len(event) == 2:
                    namespace, node_updates = event
                    
                    # Iterate through each node's updates
                    for node_name, state_update in node_updates.items():
                        # Check for ui_messages in the state update (intermediate status messages)
                        if "ui_messages" in state_update and state_update["ui_messages"]:
                            for ui_msg in state_update["ui_messages"]:
                                if isinstance(ui_msg, AIMessage) and ui_msg.content:
                                    final_response = ui_msg.content
                                    # Yield each ui_message
                                    yield {
                                        'is_task_complete': False,
                                        'require_user_input': False,
                                        'content': ui_msg.content
                                    }
                        
                        # Also check regular messages for all AI responses
                        if "messages" in state_update and state_update["messages"]:
                            for msg in state_update["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    # Skip if this is just a repeat of what we already yielded
                                    if msg.content != final_response or not final_response:
                                        final_response = msg.content
                                        yield {
                                            'is_task_complete': False,
                                            'require_user_input': False,
                                            'content': msg.content
                                        }

            # Final response
            if final_response:
                # Save AI response to Supabase
                await self.save_message(
                    conversation_id=context_id,
                    role="ai",
                    content=final_response,
                    metadata={"client_id": client_id, "project_id": project_id}
                )
                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': final_response
                }
            else:
                # Save default completion message
                await self.save_message(
                    conversation_id=context_id,
                    role="ai",
                    content="Task completed.",
                    metadata={"client_id": client_id, "project_id": project_id}
                )
                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': "Task completed."
                }

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"An error occurred: {str(e)}"
            }

    def invoke(self, message: str, client_id: str = None, thread_id: str = "default", project_id: str = None) -> Dict[str, Any]:
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
                    client_id = parsed.get("client_id", client_id)
                    thread_id = parsed.get("thread_id", thread_id)
                    project_id = parsed.get("project_id", project_id)
            except:
                pass

        config = {"configurable": {"thread_id": thread_id}} 
        
        # We need to collect the output since it streams
        final_response = ""
        full_state = {}
        
        for event in self.graph.stream(
            {"messages": [HumanMessage(content=message)], "client_id": client_id, "project_id": project_id},
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

async def test():
    agent = FlashBrainAgent()
    message = """The Project Manager needs to know accounting skills. Can you help me find a freelancer with accounting skills?"""
    project_id = '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc'
    client_id = '9a76be62-0d44-4a34-913d-08dcac008de5'
    
    print("\n=== Streaming Response ===")
    async for item in agent.stream(message, project_id=project_id, client_id=client_id):
        print(f"[{item.get('is_task_complete')}] {item.get('content')}")

if __name__ == "__main__":

    asyncio.run(test())# Uncomment to run test

    # Start A2A server
    #port = int(os.getenv("PORT", 8010))
    #print(f"Starting FlashBrain Orchestrator on port {port}...")
    #run_server(FlashBrainAgent(), port=port)

