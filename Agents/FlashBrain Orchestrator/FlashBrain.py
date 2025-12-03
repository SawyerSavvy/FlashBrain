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
                        +-- Tool: process_document (MCP -> RAG Service)
                        +-- Tool: search_knowledge_base (MCP -> RAG Service)

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
            postgres_connection: PostgreSQL connection string (from SUPABASE_POOLER env)
        """
        logger.info("Initializing FlashBrain ReAct Agent")

        # Configuration
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        # Use SUPABASE_POOLER for checkpointer connection
        self.postgres_connection = postgres_connection or os.getenv("SUPABASE_POOLER")
        self.supabase_client = None
        self.remote_agent_connections = {} # {card name: RemoteAgentConneciton}
        self.cards = {} # {card name: AgentCard}
        self.agents = None
        self.tools = []
        self.rag_service_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8014")
        
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
        if self.checkpointer_cm and self.checkpointer:
            try:
                # Exit the async context manager properly
                await self.checkpointer_cm.__aexit__(None, None, None)
                logger.info("PostgreSQL checkpointer closed")
            except Exception as e:
                logger.error(f"Failed to close checkpointer: {e}")
            finally:
                self.checkpointer = None
                self.checkpointer_cm = None
        
        if hasattr(self, 'pool') and self.pool:
            try:
                await self.pool.close()
                logger.info("PostgreSQL connection pool closed")
            except Exception as e:
                logger.error(f"Failed to close pool: {e}")
      
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
    
    def _create_process_document_tool(self):
        """
        Creates a tool for processing documents via MCP RAG service.
        
        Returns:
            A LangChain tool function
        """
        agent_instance = self
        
        @tool
        async def process_document(
            file_path: Optional[str] = None,
            file_url: Optional[str] = None,
            content: Optional[str] = None,
            supabase_bucket_name: Optional[str] = None,
            supabase_bucket_path: Optional[str] = None,
            source_type: str = "pdf",
            title: Optional[str] = None,
            project_id: Optional[str] = None,
            client_id: Optional[str] = None
        ) -> str:
            """
            Process a document with intelligent decomposition into semantic chunks.
            The document will be stored in the knowledge base for future searches.
            
            Priority order:
            1. content (direct text)
            2. supabase_bucket_name + supabase_bucket_path (RECOMMENDED for production - no API timeouts, files persist)
            3. file_path (local file)
            4. file_url (external URL)
            
            Args:
                file_path: Path to local file (e.g., "/path/to/document.pdf")
                file_url: URL to fetch file from (e.g., "https://example.com/doc.pdf")
                content: Direct text content (if source_type is 'text')
                supabase_bucket_name: Supabase Storage bucket name (e.g., "documents") - RECOMMENDED
                supabase_bucket_path: Path to file in Supabase Storage (e.g., "project-123/doc.pdf")
                source_type: Type of document ('pdf', 'text', 'url', 'file')
                title: Optional document title
                project_id: Optional project ID for context
                client_id: Optional client ID for context
            
            Returns:
                JSON string with document_id and chunk_count
            """
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    request_data = {
                        "source_type": source_type
                    }
                    if file_path:
                        request_data["file_path"] = file_path
                    if file_url:
                        request_data["file_url"] = file_url
                    if content:
                        request_data["content"] = content
                    if supabase_bucket_name:
                        request_data["supabase_bucket_name"] = supabase_bucket_name
                    if supabase_bucket_path:
                        request_data["supabase_bucket_path"] = supabase_bucket_path
                    if title:
                        request_data["title"] = title
                    if project_id:
                        request_data["project_id"] = project_id
                    if client_id:
                        request_data["client_id"] = client_id
                    
                    response = await client.post(
                        f"{agent_instance.rag_service_url}/mcp/tools/process_document",
                        json=request_data
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    return json.dumps({
                        "status": "success",
                        "document_id": result["document_id"],
                        "chunk_count": result["chunk_count"],
                        "title": result["title"],
                        "source": result["source"]
                    })
            except httpx.HTTPError as e:
                error_msg = f"Failed to process document: {str(e)}"
                logger.error(error_msg)
                return json.dumps({"status": "error", "message": error_msg})
            except Exception as e:
                error_msg = f"Unexpected error processing document: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return json.dumps({"status": "error", "message": error_msg})
        
        return process_document
    
    def _create_search_knowledge_base_tool(self):
        """
        Creates a tool for semantic search via MCP RAG service.
        
        Returns:
            A LangChain tool function
        """
        agent_instance = self
        
        @tool
        async def search_knowledge_base(
            query: str,
            match_threshold: float = 0.7,
            match_count: int = 10,
            document_id: Optional[str] = None,
            project_id: Optional[str] = None,
            client_id: Optional[str] = None
        ) -> str:
            """
            Search the knowledge base for semantically similar document chunks.
            Use this to find relevant information from previously processed documents.
            
            Args:
                query: Search query text
                match_threshold: Minimum similarity threshold (0-1, default 0.7)
                match_count: Maximum number of results (default 10)
                document_id: Optional filter by specific document ID
                project_id: Optional filter by project ID
                client_id: Optional filter by client ID
            
            Returns:
                JSON string with search results including content and similarity scores
            """
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    request_data = {
                        "query": query,
                        "match_threshold": match_threshold,
                        "match_count": match_count
                    }
                    if document_id:
                        request_data["document_id"] = document_id
                    if project_id:
                        request_data["project_id"] = project_id
                    if client_id:
                        request_data["client_id"] = client_id
                    
                    response = await client.post(
                        f"{agent_instance.rag_service_url}/mcp/tools/search_knowledge_base",
                        json=request_data
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Format results for readability
                    formatted_results = []
                    for r in result["results"]:
                        formatted_results.append({
                            "content": r["content"],
                            "similarity": round(r["similarity"], 3),
                            "document_id": r["document_id"],
                            "chunk_index": r["chunk_index"],
                            "metadata": r.get("metadata", {})
                        })
                    
                    return json.dumps({
                        "status": "success",
                        "query": result["query"],
                        "total_results": result["total_results"],
                        "results": formatted_results
                    }, indent=2)
            except httpx.HTTPError as e:
                error_msg = f"Failed to search knowledge base: {str(e)}"
                logger.error(error_msg)
                return json.dumps({"status": "error", "message": error_msg})
            except Exception as e:
                error_msg = f"Unexpected error searching knowledge base: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return json.dumps({"status": "error", "message": error_msg})
        
        return search_knowledge_base
    
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
            self._create_process_document_tool(),  # MCP: Process documents
            self._create_search_knowledge_base_tool(),  # MCP: Semantic search
        ]
        logger.info(f"Created {len(self.tools)} tools ({len(self.remote_agent_connections)} remote agents, 2 MCP RAG tools)")
        
        # Initialize PostgreSQL checkpointer for state persistence
        if not self.postgres_connection:
            raise RuntimeError(
                "SUPABASE_POOLER environment variable is required. "
                "FlashBrain cannot run without persistent state storage."
            )

        # DON'T create the graph here - it will be created fresh per request in stream()
        # This ensures proper connection lifecycle and avoids stale connections
        # create_agent() returns a CompiledStateGraph, so we can't compile it again
        # Instead, we'll call create_agent() fresh in stream() with a fresh checkpointer

        logger.info("ReAct agent builder ready (graph will be created per-request with fresh checkpointer)")
        return None  # No pre-compiled graph
    
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

            # Create fresh checkpointer for THIS request only
            # This ensures proper connection lifecycle and avoids stale/closed connections
            async with AsyncPostgresSaver.from_conn_string(self.postgres_connection) as checkpointer:
                await checkpointer.setup()

                # Create the ReAct agent WITH fresh checkpointer
                # This returns a CompiledStateGraph ready to use
                graph = create_agent(
                    self.llm,
                    tools=self.tools,
                    checkpointer=checkpointer
                )

                # Stream from the ReAct agent with fresh checkpointer
                async for event in graph.astream(
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

                                # Parse Gemini's response format
                                if isinstance(content, list):
                                    # Handle list of content items (Gemini format)
                                    text_parts = []
                                    for item in content:
                                        if isinstance(item, dict):
                                            # Extract text from dict format: {'type': 'text', 'text': '...', 'extras': {...}}
                                            if item.get('type') == 'text' and 'text' in item:
                                                text_parts.append(item['text'])
                                            elif 'text' in item:
                                                text_parts.append(item['text'])
                                            else:
                                                # Fallback for other dict formats
                                                text_parts.append(str(item))
                                        elif isinstance(item, str):
                                            text_parts.append(item)
                                        else:
                                            text_parts.append(str(item))
                                    content = '\n'.join(text_parts)
                                elif isinstance(content, dict):
                                    # Handle single dict format: {'type': 'text', 'text': '...', 'extras': {...}}
                                    if content.get('type') == 'text' and 'text' in content:
                                        content = content['text']
                                    elif 'text' in content:
                                        content = content['text']
                                    else:
                                        content = str(content)

                                # Skip if we've already seen this exact content
                                if content in seen_content:
                                    continue

                                seen_content.add(content)

                                # Normal message - just stream it
                                final_response = content
                                yield {
                                    'is_task_complete': False,
                                    'require_user_input': False,
                                    'content': content
                                }

                # Save final AI response (after streaming completes)
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
            logger.warning(f"Stream cancelled by client for thread {context_id}. Cleaning up.")
            # Checkpointer context manager should handle cleanup
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
