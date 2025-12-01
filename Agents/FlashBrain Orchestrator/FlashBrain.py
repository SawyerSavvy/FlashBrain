"""
FlashBrain ReAct Agent - Simplified Architecture

This module implements FlashBrain as a single ReAct (Reasoning + Acting) agent
with tools for calling sub-agents and database operations.

Architecture:
    User Request -> ReAct Agent (single LLM with tools)
                        |
                        +-- Tool: create_project_plan (A2A -> Project Decomp Agent)
                        +-- Tool: find_freelancers (A2A -> Select Freelancer Agent)
                        +-- Tool: get_project_data (Supabase read)
                        +-- Tool: calculate_budget (FinOps)
                        +-- Tool: search_knowledge_base (future MCP)

Benefits over custom graph:
    - 1 LLM call instead of 3+ (faster, cheaper)
    - 1 graph node instead of 10 (simpler state)
    - LangGraph handles tool calling automatically
    - Built-in retry logic for tools
    - Automatic streaming support
    - Multi-intent handled naturally by LLM reasoning
"""

from typing import Annotated, List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import logging
import asyncio
import uuid
import json
import threading
from prompts import FLASHBRAIN_SYSTEM_PROMPT
from supabase import create_client
from available_agents import AGENTS_URLS
from a2a.client import A2ACardResolver
from remote_agent_connections import RemoteAgentConnections

from psycopg_pool import AsyncConnectionPool

# Create A2A message

from a2a.types import (
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    SendMessageRequest,
    MessageSendParams,
    Message,
    SendMessageSuccessResponse
)

from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import httpx

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- FlashBrain ReAct Agent Class ---

class FlashBrainReActAgent:
    """
    Simplified FlashBrain agent using LangGraph's ReAct pattern.
    
    This replaces the complex custom graph with a single ReAct agent
    that uses tools for all operations.
    """

    SUPPORTED_CONTENT_TYPES = [
        'text', 'text/plain', 'application/json'
    ]

    def __init__(
        self,
        supabase_url: str = None,
        supabase_key: str = None,
        postgres_connection: str = None
    ):
        """
        Initialize the FlashBrain ReAct Agent.
        
        Args:
            supabase_url: Supabase project URL (or from SUPABASE_URL env)
            supabase_key: Supabase service role key (or from SUPABASE_SERVICE_ROLE_KEY env)
            postgres_connection: PostgreSQL connection string (or from SUPABASE_TRANSACTION_POOLER env)
        """
        logger.info("Initializing FlashBrain ReAct Agent")
        
        # Configuration
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.postgres_connection = postgres_connection or os.getenv("SUPABASE_TRANSACTION_POOLER")
        self.supabase_client = None
        self.remote_agent_connections = {} # {card name: RemoteAgentConneciton}
        self.cards = {} # {card name: AgentCard}
        self.agents = None
        self.tools = []
        
        # Initialize Supabase client for conversation persistence
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
        
        # Initialize LLM using the same pattern as FlashBrainAgent
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1
            )
            logger.info("Gemini LLM initialized (gemini-2.5-flash)")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            raise
        
        result = [None]
        exception = [None]
        
        def build_in_new_thread():
            """Run _build_graph in a separate thread with its own event loop."""
            try:
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result[0] = new_loop.run_until_complete(self._build_graph())
                # DON'T close the loop - PostgreSQL pool needs it to stay alive
                # The loop will be cleaned up when the thread ends naturally
            except Exception as e:
                exception[0] = e
        
        # Run in a separate thread to avoid event loop conflicts
        thread = threading.Thread(target=build_in_new_thread, daemon=True)
        thread.start()
        thread.join()
        
        if exception[0]:
            raise exception[0]
        
        self.graph = result[0]
    
    async def close(self):
        """
        Close resources and cleanup connections.
        Call this on server shutdown to prevent connection leaks.
        """
        if hasattr(self, 'pool') and self.pool:
            try:
                await self.pool.close()
                logger.info("PostgreSQL connection pool closed")
            except Exception as e:
                logger.error(f"Failed to close pool: {e}")
      
    def _create_request_human_input_tool(self):
        """
        Creates a tool that allows the agent to request input from the user.
        Returns a structured JSON signal that the orchestrator can detect.
        
        Returns:
            A LangChain tool function
        """
        @tool
        def request_human_input(question: str, context: Optional[str] = None) -> str:
            """
            Request input or clarification from the human user.
            Use this when you need more information to complete a task.
            
            Args:
                question: The question to ask the user
                context: Optional context about why you need this information
            
            Returns:
                JSON signal that input is required
            """
            # Return structured JSON that the orchestrator can detect
            payload = {
                "type": "HUMAN_INPUT_REQUIRED",
                "question": question,
                "context": context or ""
            }
            return json.dumps(payload)
        
        return request_human_input
    
    def _create_send_message_tool(self):
        """
        Creates the send_message tool with access to self.remote_agent_connections.
        
        Returns:
            A LangChain tool function
        """
        # Capture self in closure
        agent_instance = self
        
        @tool
        async def send_message(
            agent_name: str,
            task: str,
            project_id: Optional[str] = None,
            client_id: Optional[str] = None
        ) -> str:
            """
            Send a message to a remote agent via A2A protocol.
            
            Args:
                agent_name: The name of the agent to call (e.g., "Project Decomposition Agent", "Select Freelancer Agent")
                task: The task or question to send to the agent
                project_id: Optional project ID for context
                client_id: Optional client ID for authentication
            
            Returns:
                Response from the remote agent
            """
            # Get the remote connection for this agent
            connection = agent_instance.remote_agent_connections.get(agent_name)
            
            if not connection:
                available = list(agent_instance.remote_agent_connections.keys())
                return f"Agent '{agent_name}' not found. Available agents: {', '.join(available)}"
            
            message_id = str(uuid.uuid4())

            message_data = {
                "message": {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': task}],
                    'message_id': message_id,
                    'metadata': {
                        'project_id': project_id,
                        'client_id': client_id
                    }
                }
            }

            message_request = SendMessageRequest(id=message_id, params=MessageSendParams.model_validate(message_data))
            
            send_response: SendMessageResponse = await connection.send_message(message_request)

            if not isinstance(
                send_response.root, SendMessageSuccessResponse
            ) or not isinstance(send_response.root.result, Task):
                return f"Failed to send message to {agent_name}"

            response_content = send_response.root.model_dump_json(exclude_none=True)
            json_content = json.loads(response_content)

            resp = []
            if json_content.get("result", {}).get("artifacts"):
                for artifact in json_content["result"]["artifacts"]:
                    if artifact.get("parts"):
                        resp.extend(artifact["parts"])
            
            # Return as string, not list
            if resp:
                return '\n'.join([str(part) for part in resp])
            else:
                return f"No response content from {agent_name}"
        
        return send_message
    
    async def _build_graph(self):
        """
        Builds the ReAct agent graph with PostgreSQL checkpointer.
        Also initializes remote agent connections.
        
        Returns:
            Compiled LangGraph agent
        """
        # Initialize remote agent connections
        async with httpx.AsyncClient(timeout=30.0) as client:
            for address in AGENTS_URLS:
                card_resolver = A2ACardResolver(httpx_client=client, base_url=address)
                try:
                    # Await the async method
                    card = await card_resolver.get_agent_card()
                    
                    # Create remote connection
                    remote_connection = RemoteAgentConnections(
                        agent_card=card,
                        agent_url=address  # Correct parameter name
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    
                    self.cards[card.name] = card
                    logger.info(f"Connected to agent: {card.name} at {address}")
                except Exception as e:
                    logger.error(f"Failed to get agent card for {address}: {e}")
                    continue
        
        # Build agent info string for system prompt
        agent_info = [
            json.dumps({"name": name, "description": card.description}) 
            for name, card in self.cards.items()
        ]
        self.agents = "\n".join(agent_info) if agent_info else "No agents available"
        logger.info(f"Initialized {len(self.cards)} agent connections")

        # Define tools now that we have agent connections
        self.tools = [
            self._create_send_message_tool(),  # Generic tool for calling any agent
            self._create_request_human_input_tool(),  # Request user input
        ]
        logger.info(f"Created {len(self.tools)} tools ({len(self.remote_agent_connections)} remote agents)")
        
        # Initialize PostgreSQL checkpointer for state persistence
        if not self.postgres_connection:
            raise RuntimeError(
                "SUPABASE_TRANSACTION_POOLER environment variable is required. "
                "FlashBrain cannot run without persistent state storage."
            )

        try:
            

            # Cloud Run optimized connection pool settings
            # Increased max_size and max_waiting for test environment
            pool = AsyncConnectionPool(
                conninfo=self.postgres_connection,
                max_size=10,  # Increased to 10 to prevent exhaustion during dev/testing
                min_size=0,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": None,
                    "connect_timeout": 20,
                    "keepalives": 1,
                    "keepalives_idle": 10,
                    "keepalives_interval": 5,
                    "keepalives_count": 3,
                },
                open=False,
                timeout=20.0,
                max_waiting=20,  # Allow more queued requests
                max_idle=30.0,   # Recycle idle connections quickly
                max_lifetime=300.0, # Close connections after 5 mins to prevent leaks
                reconnect_timeout=10.0,
                check=AsyncConnectionPool.check_connection,
                reset=False,
            )
            
            await pool.open()
            logger.info(f"PostgreSQL connection pool opened ({pool.min_size}-{pool.max_size} connections)")
            
            # Store pool reference for cleanup
            self.pool = pool
            
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            logger.info("AsyncPostgresSaver checkpointer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize checkpointer: {e}") from e
        
        # Create the ReAct agent - this is the magic!
        # LangGraph handles all the complexity of tool calling, retries, and state management
        # Note: System prompt is injected via messages in the stream() method
        agent = create_agent(
            self.llm,
            tools=self.tools,
            checkpointer=checkpointer
        )
        
        logger.info("ReAct agent created successfully")
        return agent
    
    async def ensure_conversation_exists(self, conversation_id: str, user_id: str = None):
        """
        Ensures a conversation record exists in the database.
        
        Args:
            conversation_id: Unique conversation/thread ID
            user_id: Optional user ID
        """
        if not self.supabase_client:
            return

        try:
            response = self.supabase_client.table("brain_conversations")\
                .select("id")\
                .eq("id", conversation_id)\
                .execute()

            if not response.data:
                self.supabase_client.table("brain_conversations").insert({
                    "id": conversation_id,
                    "user_id": user_id or "anonymous",
                    "title": "New Conversation"
                }).execute()
        except Exception as e:
            logger.error(f"Failed to ensure conversation exists: {e}")

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict = None
    ):
        """
        Saves a message to the conversation history.
        
        Args:
            conversation_id: Conversation/thread ID
            role: Message role (human, ai)
            content: Message content
            metadata: Optional metadata dict
        """
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
    
    async def stream(
        self,
        query: str,
        context_id: str = "default",
        client_id: str = None,
        project_id: str = None
    ):
        """
        Streams responses from the FlashBrain ReAct Agent.
        
        Args:
            query: User's message/query
            context_id: Conversation thread ID
            client_id: Client ID for authentication
            project_id: Current project ID context

        Yields:
            Dict with streaming response data:
                - is_task_complete: bool
                - require_user_input: bool
                - content: str
        """
        config = {"configurable": {"thread_id": context_id}}

        try:
            # Ensure conversation exists
            await self.ensure_conversation_exists(context_id, client_id)
            
            # Save user message
            await self.save_message(
                conversation_id=context_id,
                role="human",
                content=query,
                metadata={"client_id": client_id, "project_id": project_id}
            )
            
            # Build message list with system prompt and context
            messages = [SystemMessage(content=f"{FLASHBRAIN_SYSTEM_PROMPT}\n\nAvailable agents: {self.agents}")]
            
            # Add context information if provided
            if project_id or client_id:
                context_info = []
                if project_id:
                    context_info.append(f"Current project_id: {project_id}")
                if client_id:
                    context_info.append(f"Client ID: {client_id}")
                messages.append(SystemMessage(content=f"[Context: {', '.join(context_info)}]"))
            
            # Add user query
            messages.append(HumanMessage(content=query))
            
            final_response = ""
            seen_content = set()  # Track already-yielded content to avoid duplicates

            # Stream from the ReAct agent
            async for event in self.graph.astream(
                {"messages": messages},
                config=config,
                stream_mode="messages"
            ):
                # Handle different event types
                if isinstance(event, tuple):
                    message, _ = event

                    # Stream AI message content
                    if isinstance(message, AIMessage) and message.content:
                        # Skip tool call messages (they have tool_calls attribute)
                        if not message.tool_calls:
                            # Handle both string and list content
                            content = message.content
                            if isinstance(content, list):
                                # Convert list to string
                                content = '\n'.join([str(item) for item in content])

                            # Skip if we've already seen this exact content
                            if content in seen_content:
                                continue

                            seen_content.add(content)

                            # Check if human input is required by detecting JSON signal
                            requires_input = False
                            human_request = None

                            try:
                                # Try to parse as JSON (in case the content is the tool output)
                                data = json.loads(content)
                                if data.get("type") == "HUMAN_INPUT_REQUIRED":
                                    requires_input = True
                                    human_request = data
                            except (json.JSONDecodeError, TypeError):
                                # Not JSON, check for plain text signal
                                if '"type": "HUMAN_INPUT_REQUIRED"' in content or '"type":"HUMAN_INPUT_REQUIRED"' in content:
                                    requires_input = True
                                    # Try to extract from the text
                                    try:
                                        start_idx = content.find('{')
                                        end_idx = content.rfind('}') + 1
                                        if start_idx >= 0 and end_idx > start_idx:
                                            human_request = json.loads(content[start_idx:end_idx])
                                    except:
                                        pass

                            if requires_input and human_request:
                                # Extract question from the structured payload
                                question = human_request.get("question", "Please provide more information")
                                context_info = human_request.get("context", "")

                                response_content = question
                                if context_info:
                                    response_content += f"\n\nContext: {context_info}"

                                yield {
                                    'is_task_complete': False,
                                    'require_user_input': True,
                                    'content': response_content
                                }
                                # Stop streaming - wait for user response
                                logger.info(f"Human input requested: {question}")
                                # Save the response before stopping
                                await self.save_message(
                                    conversation_id=context_id,
                                    role="ai",
                                    content=response_content,
                                    metadata={"client_id": client_id, "project_id": project_id}
                                )
                                yield {
                                    'is_task_complete': True,
                                    'require_user_input': True,
                                    'content': ''
                                }
                                return
                            else:
                                # Normal message
                                final_response = content
                                yield {
                                    'is_task_complete': False,
                                    'require_user_input': False,
                                    'content': content
                                }

            # Save final AI response
            if final_response:
                await self.save_message(
                    conversation_id=context_id,
                    role="ai",
                    content=final_response,
                    metadata={"client_id": client_id, "project_id": project_id}
                )
                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': ''  # Don't repeat content, just signal completion
                }
            else:
                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': ''
                }

        except GeneratorExit:
            logger.warning(f"Stream cancelled by client for thread {context_id}")
            # Checkpointer context manager should handle cleanup
            raise
        except GeneratorExit:
            logger.warning(f"Stream cancelled by client for thread {context_id}. Cleaning up.")
            # Ensure the generator is closed properly
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"An error occurred: {str(e)}"
            }

    def invoke(
        self,
        message: str,
        client_id: str = None,
        thread_id: str = "default",
        project_id: str = None
    ) -> Dict[str, Any]:
        """
        Synchronous invocation of the FlashBrain agent.
        
        Args:
            message: User message
            client_id: Client ID for authentication
            thread_id: Conversation thread ID
            project_id: Current project ID context
        
        Returns:
            Dict with response and state information
        """
        # Handle JSON payload parsing
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    message = parsed.get("message", message)
                    client_id = parsed.get("client_id", client_id)
                    thread_id = parsed.get("thread_id", thread_id)
                    project_id = parsed.get("project_id", project_id)
            except json.JSONDecodeError:
                pass

        config = {"configurable": {"thread_id": thread_id}} 
        
        # Build message list with system prompt and context (same as stream method)
        messages = [SystemMessage(content=FLASHBRAIN_SYSTEM_PROMPT)]
        
        # Add context information if provided
        if project_id or client_id:
            context_info = []
            if project_id:
                context_info.append(f"Current project_id: {project_id}")
            if client_id:
                context_info.append(f"Client ID: {client_id}")
            messages.append(SystemMessage(content=f"[Context: {', '.join(context_info)}]"))
        
        # Add user message
        messages.append(HumanMessage(content=message))
        
        # Invoke synchronously
        final_response = ""
        
        for event in self.graph.stream(
            {"messages": messages},
            config=config
        ):
            for key, value in event.items():
                if "messages" in value and value["messages"]:
                    last_msg = value["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        if not last_msg.tool_calls:
                            final_response = last_msg.content
                
        return {
            "response": final_response,
            "thread_id": thread_id
        }


# --- Test Function ---

async def test():
    """Test the FlashBrain ReAct Agent."""
    agent = FlashBrainReActAgent()
    
    # Generate a valid UUID for the test session
    import uuid
    test_session_id = str(uuid.uuid4())
    print(f"\nTest session ID: {test_session_id}\n")
    
    test_queries = [
        "What is agile methodology?",
        "Create a project plan for an e-commerce website with React frontend and Node.js backend",
        "What is my project budget runway if I have $50,000 and spend $5,000 per month?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        async for item in agent.stream(query, context_id=test_session_id, client_id="9a76be62-0d44-4a34-913d-08dcac008de5", project_id="058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc"):
            if item.get('content'):
                print(item['content'])
            if item.get('is_task_complete'):
                print("\n[Task Complete]")


if __name__ == "__main__":
    asyncio.run(test())

