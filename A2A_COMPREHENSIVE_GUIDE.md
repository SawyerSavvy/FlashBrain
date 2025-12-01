# A2A (Agent-to-Agent) Protocol - Comprehensive Python Guide

## Overview

The A2A (Agent-to-Agent) protocol is Google's standard for enabling communication between AI agents. It provides a structured way for agents to discover, connect to, and interact with each other over HTTP/gRPC.

**Key Benefits:**
- **Interoperability**: Agents from different platforms can communicate
- **Standardization**: Common protocol for agent communication
- **Flexibility**: Supports text, files, structured data, and streaming
- **Discovery**: Agents can be discovered dynamically via agent cards

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Installation](#installation)
3. [Agent Card](#agent-card)
4. [Building a Server](#building-a-server)
5. [Building a Client](#building-a-client)
6. [Message Types](#message-types)
7. [Task Management](#task-management)
8. [Streaming](#streaming)
9. [File Handling](#file-handling)
10. [Authentication](#authentication)
11. [Push Notifications](#push-notifications)
12. [Error Handling](#error-handling)
13. [Production Deployment](#production-deployment)

---

## Core Concepts

### Agent Card

Every A2A agent exposes an **Agent Card** - metadata describing its capabilities:

```python
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

agent_card = AgentCard(
    name="My Agent",
    description="An agent that does X",
    url="https://my-agent.com",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=True,
        push_notifications=True
    ),
    skills=[
        AgentSkill(
            id="skill_1",
            name="My Skill",
            description="Does something useful",
            tags=["tag1", "tag2"],
            examples=["Example 1", "Example 2"]
        )
    ],
    default_input_modes=["text/plain", "application/json"],
    default_output_modes=["text/plain", "application/json"]
)
```

### Messages

Messages are the primary communication unit:

```python
from a2a.types import Message, Part, TextPart, Role

message = Message(
    role=Role.user,  # or Role.agent
    parts=[Part(root=TextPart(text="Hello, agent!"))],
    message_id="msg_123",
    context_id="conversation_abc"  # For multi-turn conversations
)
```

### Tasks

Tasks represent units of work with states:

```python
from a2a.types import Task, TaskStatus, TaskState

task = Task(
    id="task_456",
    context_id="conversation_abc",
    status=TaskStatus(
        state=TaskState.working,  # pending, working, completed, failed, canceled, input_required
        message=message
    )
)
```

---

## Installation

```bash
# Install the official A2A SDK
pip install a2a-sdk[http-server]

# Or for specific transport:
pip install a2a-sdk[grpc]
pip install a2a-sdk[all]
```

**Dependencies:**
```python
# requirements.txt
a2a-sdk[http-server]>=0.3.16
httpx>=0.24.0
uvicorn>=0.23.0  # For HTTP server
pydantic>=2.0.0
```

---

## Agent Card

### Creating an Agent Card

```python
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

agent_card = AgentCard(
    name="Weather Agent",
    description="Provides weather forecasts and current conditions",
    url="https://weather-agent.example.com",
    version="1.0.0",
    
    capabilities=AgentCapabilities(
        streaming=True,
        push_notifications=True,
        state_transition_history=False
    ),
    
    skills=[
        AgentSkill(
            id="get_weather",
            name="Get Weather",
            description="Get current weather for a location",
            tags=["weather", "forecast"],
            examples=[
                "What's the weather in Paris?",
                "Get current conditions for Tokyo"
            ]
        )
    ],
    
    default_input_modes=["text/plain", "application/json"],
    default_output_modes=["text/plain", "application/json"],
    
    preferred_transport="JSONRPC",  # or "GRPC"
    protocol_version="0.3.0"
)
```

### Exposing the Agent Card

The agent card is automatically exposed at `/.well-known/agent-card.json`:

```python
# Clients can fetch it:
GET https://your-agent.com/.well-known/agent-card.json

# Returns:
{
  "name": "Weather Agent",
  "description": "Provides weather forecasts...",
  "url": "https://weather-agent.example.com",
  "version": "1.0.0",
  ...
}
```

---

## Building a Server

### Step 1: Implement AgentExecutor

The `AgentExecutor` is the core logic of your agent:

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Task, TaskStatus, TaskState
from a2a.utils.message import new_agent_text_message

class WeatherAgent(AgentExecutor):
    """
    Agent that provides weather information.
    """
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute the agent logic.
        
        Args:
            context: Contains the incoming message and task information
            event_queue: Queue for sending status updates and results
        """
        # Extract user message
        if not context.message or not context.message.parts:
            return
        
        user_text = context.message.parts[0].root.text
        
        # Send "working" status
        await event_queue.put(Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(
                state=TaskState.working,
                message=new_agent_text_message(
                    "Fetching weather data...",
                    context.context_id,
                    context.task_id
                )
            )
        ))
        
        # Do actual work (e.g., call weather API)
        weather_data = await self.get_weather(user_text)
        
        # Send completed task
        await event_queue.put(Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(
                state=TaskState.completed,
                message=new_agent_text_message(
                    f"Weather: {weather_data}",
                    context.context_id,
                    context.task_id
                )
            )
        ))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation."""
        await event_queue.put(Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.canceled)
        ))
    
    async def get_weather(self, location: str) -> str:
        """Fetch weather data (implement your logic here)."""
        return f"Sunny, 72°F in {location}"
```

### Step 2: Create Server Application

```python
import logging
import click
import httpx
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication  # or A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    InMemoryTaskStore,  # or DatabaseTaskStore
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore
)
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
def main(host, port):
    """Start the A2A agent server."""
    
    # Define agent card
    agent_card = AgentCard(
        name="Weather Agent",
        description="Provides weather information",
        url=f"http://{host}:{port}",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=[AgentSkill(id="weather", name="Weather", description="Get weather data")],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"]
    )
    
    # Create task store
    task_store = InMemoryTaskStore()
    
    # Create push notification components
    httpx_client = httpx.AsyncClient()
    push_config_store = InMemoryPushNotificationConfigStore()
    push_sender = BasePushNotificationSender(
        httpx_client=httpx_client,
        config_store=push_config_store
    )
    
    # Create request handler
    handler = DefaultRequestHandler(
        agent_executor=WeatherAgent(),
        task_store=task_store,
        push_config_store=push_config_store,
        push_sender=push_sender
    )
    
    # Build application
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler
    )
    
    app = server.build()
    
    logger.info(f"Starting Weather Agent on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
```

### Exposed Endpoints

The server automatically creates these endpoints:

- `GET /.well-known/agent-card.json` - Agent card
- `POST /` - JSON-RPC endpoint (default)
- `POST /message/send` - Send message (non-streaming)
- `POST /message/stream` - Send message (streaming)
- `POST /tasks/get` - Get task status
- `POST /tasks/cancel` - Cancel a task

---

## Building a Client

### Step 1: Connect to an Agent

```python
import asyncio
import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import AgentCard, Message

async def connect_to_agent():
    """Connect to a remote A2A agent."""
    
    async with httpx.AsyncClient(timeout=30.0) as httpx_client:
        # Resolve agent card
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url="https://agent.example.com"
        )
        
        # Fetch the agent card
        agent_card = await resolver.get_agent_card()
        
        print(f"Connected to: {agent_card.name}")
        print(f"Description: {agent_card.description}")
        print(f"Skills: {[skill.name for skill in agent_card.skills]}")
        
        # Create client
        client_config = ClientConfig(httpx_client=httpx_client)
        client = await ClientFactory.connect(
            agent=agent_card,
            client_config=client_config
        )
        
        return client

asyncio.run(connect_to_agent())
```

### Step 2: Send Messages

```python
from a2a.types import Message, Part, TextPart, Role
from uuid import uuid4

async def send_message_example(client):
    """Send a message to the agent."""
    
    # Create message
    message_data = {
        'role': 'user',
        'parts': [{'kind': 'text', 'text': 'What is the weather in Paris?'}],
        'message_id': uuid4().hex,
        'metadata': {'user_id': 'user_123'}
    }
    message = Message(**message_data)
    
    # Send and process response
    async for item in client.send_message(message):
        if isinstance(item, tuple):
            task, update = item
            
            # Extract response text
            if hasattr(update, 'status') and update.status:
                if hasattr(update.status, 'message') and update.status.message:
                    msg = update.status.message
                    if hasattr(msg, 'parts') and msg.parts:
                        for part in msg.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                print(part.root.text)
```

---

## Message Types

### 1. Text Message

```python
from a2a.types import Message, Part, TextPart, Role

message = Message(
    role=Role.user,
    parts=[
        Part(root=TextPart(text="Hello, agent!"))
    ],
    message_id="msg_001"
)
```

### 2. File Message (URI)

```python
from a2a.types import Message, Part, FilePart, FileWithUri, Role

message = Message(
    role=Role.user,
    parts=[
        Part(root=TextPart(text="Analyze this document:")),
        Part(root=FilePart(
            file=FileWithUri(
                uri="https://example.com/document.pdf",
                name="document.pdf",
                mime_type="application/pdf"
            )
        ))
    ],
    message_id="msg_002"
)
```

### 3. File Message (Bytes)

```python
from a2a.types import Message, Part, FilePart, FileWithBytes, Role

# Read file bytes
with open("report.pdf", "rb") as f:
    file_bytes = f.read()

message = Message(
    role=Role.user,
    parts=[
        Part(root=FilePart(
            file=FileWithBytes(
                bytes=file_bytes,
                name="report.pdf",
                mime_type="application/pdf"
            )
        ))
    ],
    message_id="msg_003"
)
```

### 4. Structured Data Message

```python
from a2a.types import Message, Part, DataPart, Role

message = Message(
    role=Role.user,
    parts=[
        Part(root=TextPart(text="Filter results:")),
        Part(root=DataPart(
            data={
                "filters": ["category:sales", "date:2024"],
                "limit": 100,
                "sort_by": "date_desc"
            },
            mime_type="application/json"
        ))
    ],
    message_id="msg_004"
)
```

### 5. Multi-Modal Message

```python
from a2a.types import Message, Part, TextPart, FilePart, DataPart, FileWithUri, Role

message = Message(
    role=Role.user,
    parts=[
        Part(root=TextPart(text="Analyze this report with these filters:")),
        Part(root=FilePart(
            file=FileWithUri(
                uri="https://storage.example.com/report.pdf",
                name="Q4_Report.pdf",
                mime_type="application/pdf"
            )
        )),
        Part(root=DataPart(
            data={
                "analysis_type": "financial",
                "focus_areas": ["revenue", "expenses"]
            }
        ))
    ],
    message_id="msg_005",
    metadata={"priority": "high"}
)
```

---

## Task Management

### Task States

```python
from a2a.types import TaskState

# Available states:
TaskState.pending       # Task created, not started
TaskState.working       # Task in progress
TaskState.completed     # Task finished successfully
TaskState.failed        # Task failed with error
TaskState.canceled      # Task was canceled
TaskState.input_required # Task needs human input
```

### Sending Task Updates

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Task, TaskStatus, TaskState, TaskStatusUpdateEvent
from a2a.utils.message import new_agent_text_message

class MyAgent(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Step 1: Send "working" status
        await event_queue.put(TaskStatusUpdateEvent(
            task_id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.working),
            final=False
        ))
        
        # Step 2: Do work
        result = await self.do_work()
        
        # Step 3: Send intermediate progress
        await event_queue.put(TaskStatusUpdateEvent(
            task_id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(
                state=TaskState.working,
                message=new_agent_text_message(
                    "50% complete",
                    context.context_id,
                    context.task_id
                )
            ),
            final=False
        ))
        
        # Step 4: Send completion
        await event_queue.put(Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(
                state=TaskState.completed,
                message=new_agent_text_message(
                    f"Result: {result}",
                    context.context_id,
                    context.task_id
                )
            )
        ))
```

### Requesting Human Input

```python
class InteractiveAgent(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_text = context.message.parts[0].root.text
        
        # Check if we need clarification
        if "budget" not in user_text.lower():
            # Request input from user
            await event_queue.put(Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.input_required,
                    message=new_agent_text_message(
                        "What is your budget for this project?",
                        context.context_id,
                        context.task_id
                    )
                )
            ))
            # Don't mark as final - wait for user response
            return
        
        # Continue with task...
        await event_queue.put(Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(
                state=TaskState.completed,
                message=new_agent_text_message(
                    "Project plan created!",
                    context.context_id,
                    context.task_id
                )
            )
        ))
```

---

## Streaming

### Server-Side Streaming

```python
import asyncio
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Task, TaskStatus, TaskState, TaskStatusUpdateEvent
from a2a.utils.message import new_agent_text_message

class StreamingAgent(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Start with working status
        await event_queue.put(TaskStatusUpdateEvent(
            task_id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.working),
            final=False
        ))
        
        # Stream progress updates
        for i in range(1, 6):
            await asyncio.sleep(1)
            
            await event_queue.put(TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=new_agent_text_message(
                        f"Processing... {i*20}% complete",
                        context.context_id,
                        context.task_id
                    )
                ),
                final=False
            ))
        
        # Final completion
        await event_queue.put(Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(
                state=TaskState.completed,
                message=new_agent_text_message(
                    "Processing complete!",
                    context.context_id,
                    context.task_id
                )
            )
        ))
```

### Client-Side Streaming

```python
import asyncio
from a2a.client import ClientFactory, ClientConfig
from a2a.types import Message, Part, TextPart, Role, TaskState

async def stream_example():
    async with httpx.AsyncClient() as httpx_client:
        # Connect
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8000")
        agent_card = await resolver.get_agent_card()
        
        client_config = ClientConfig(httpx_client=httpx_client)
        client = await ClientFactory.connect(agent=agent_card, client_config=client_config)
        
        # Create message
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="Process this data"))],
            message_id=uuid4().hex
        )
        
        # Stream responses
        async for event in client.send_message(message):
            if isinstance(event, tuple):
                task, update = event
                
                print(f"State: {task.status.state}")
                
                if task.status.message:
                    for part in task.status.message.parts:
                        if hasattr(part.root, 'text'):
                            print(f"Update: {part.root.text}")
                
                if task.status.state == TaskState.completed:
                    print("Task completed!")
                    break

asyncio.run(stream_example())
```

---

## File Handling

### Sending Files (URI)

```python
from a2a.types import Message, Part, FilePart, FileWithUri, TextPart, Role

async def send_file_by_url(client):
    """Send a file reference via URL."""
    
    message = Message(
        role=Role.user,
        parts=[
            Part(root=TextPart(text="Please analyze this PDF:")),
            Part(root=FilePart(
                file=FileWithUri(
                    uri="https://storage.googleapis.com/my-bucket/report.pdf",
                    name="annual_report.pdf",
                    mime_type="application/pdf"
                )
            ))
        ],
        message_id=uuid4().hex
    )
    
    async for response in client.send_message(message):
        # Process response
        pass
```

### Sending Files (Bytes)

```python
from a2a.types import Message, Part, FilePart, FileWithBytes, Role

async def send_file_bytes(client, file_path: str):
    """Send a file as bytes."""
    
    # Read file
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    message = Message(
        role=Role.user,
        parts=[
            Part(root=FilePart(
                file=FileWithBytes(
                    bytes=file_bytes,
                    name="document.pdf",
                    mime_type="application/pdf"
                )
            ))
        ],
        message_id=uuid4().hex
    )
    
    async for response in client.send_message(message):
        # Process response
        pass
```

### Receiving Files

```python
from a2a.types import FilePart, FileWithUri, FileWithBytes

async def process_file_response(task):
    """Process file from agent response."""
    
    if task.status.message:
        for part in task.status.message.parts:
            if isinstance(part.root, FilePart):
                file = part.root.file
                
                if isinstance(file, FileWithUri):
                    # Download from URI
                    print(f"File available at: {file.uri}")
                    # Use httpx to download
                    async with httpx.AsyncClient() as client:
                        response = await client.get(file.uri)
                        file_bytes = response.content
                
                elif isinstance(file, FileWithBytes):
                    # File is included as bytes
                    file_bytes = file.bytes
                    
                    # Save to disk
                    with open(file.name, 'wb') as f:
                        f.write(file_bytes)
                    print(f"Saved file: {file.name}")
```

---

## Artifacts

### Sending Artifacts

```python
from a2a.types import Artifact, Part, TextPart, TaskArtifactUpdateEvent

async def send_artifact(event_queue, task_id, context_id):
    """Send an artifact to the client."""
    
    artifact = Artifact(
        artifact_id="artifact_123",
        name="analysis_results.json",
        description="Analysis results",
        parts=[
            Part(root=DataPart(
                data={
                    "total_items": 1000,
                    "processed": 850,
                    "errors": 5
                }
            ))
        ]
    )
    
    await event_queue.put(TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=artifact,
        append=False,  # Replace existing artifact
        last_chunk=True  # Final chunk
    ))
```

### Streaming Large Artifacts in Chunks

```python
from a2a.types import Artifact, Part, TextPart, TaskArtifactUpdateEvent

async def stream_large_file(event_queue, task_id, context_id):
    """Stream a large file in chunks."""
    
    artifact_id = "large_file_001"
    chunks = ["chunk 1\n", "chunk 2\n", "chunk 3\n"]
    
    for i, chunk in enumerate(chunks):
        artifact = Artifact(
            artifact_id=artifact_id,
            name="large_data.csv",
            description="Large dataset",
            parts=[Part(root=TextPart(text=chunk))]
        )
        
        await event_queue.put(TaskArtifactUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            artifact=artifact,
            append=True,  # Append to existing artifact
            last_chunk=(i == len(chunks) - 1)  # Mark last chunk
        ))
        
        await asyncio.sleep(0.1)  # Rate limiting
```

---

## Authentication

### Client-Side Authentication

```python
from a2a.client import ClientFactory, ClientConfig
from a2a.client.auth import CredentialService, AuthInterceptor
from a2a.client.middleware import ClientCallContext
from typing import Optional

class APIKeyCredentialService(CredentialService):
    """Credential service for API key authentication."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def get_credentials(
        self,
        security_scheme_name: str,
        context: Optional[ClientCallContext] = None
    ) -> Optional[str]:
        """Return credentials for the security scheme."""
        if security_scheme_name == "api_key":
            return self.api_key
        return None


async def connect_with_auth():
    """Connect to agent with authentication."""
    
    # Setup credentials
    credential_service = APIKeyCredentialService(api_key="sk-abc123xyz")
    auth_interceptor = AuthInterceptor(credential_service)
    
    # Configure client with auth
    config = ClientConfig(streaming=True)
    
    factory = ClientFactory(
        client_config=config,
        interceptors=[auth_interceptor]
    )
    
    # Connect
    client = await factory.connect_and_create('https://secure-agent.example.com')
    
    return client
```

### Server-Side Authentication

```python
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import InvalidParamsError
from a2a.utils.errors import ServerError

class AuthenticatedAgent(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Validate authentication
        metadata = context.message.metadata if context.message else {}
        api_key = metadata.get('api_key')
        
        if not api_key or not self.validate_api_key(api_key):
            raise ServerError(error=InvalidParamsError(message="Invalid API key"))
        
        # Continue with execution...
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate the API key."""
        # Implement your validation logic
        return api_key == "valid-key"
```

---

## Push Notifications

### Configuring Push Notifications (Client)

```python
from a2a.types import TaskPushNotificationConfig

# When sending a message, configure callback URL
config = TaskPushNotificationConfig(
    callback_url="https://my-app.com/callback",
    headers={"Authorization": "Bearer my-token"}
)

# The agent will POST to this URL when task completes
```

### Handling Push Notifications (Server)

```python
from a2a.server.tasks import BasePushNotificationSender, InMemoryPushNotificationConfigStore
import httpx

# Setup push notification sender
httpx_client = httpx.AsyncClient()
push_config_store = InMemoryPushNotificationConfigStore()
push_sender = BasePushNotificationSender(
    httpx_client=httpx_client,
    config_store=push_config_store
)

# Include in request handler
handler = DefaultRequestHandler(
    agent_executor=my_agent,
    task_store=task_store,
    push_config_store=push_config_store,
    push_sender=push_sender
)
```

### Receiving Callbacks

```python
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

async def handle_task_callback(request):
    """Handle callback from agent when task completes."""
    
    payload = await request.json()
    
    task_id = payload.get('task_id')
    status = payload.get('status')
    result = payload.get('result')
    
    print(f"Task {task_id} completed with status: {status}")
    print(f"Result: {result}")
    
    # Process the result...
    
    return JSONResponse({"status": "acknowledged"})


# Add to your app
routes = [
    Route("/callback", handle_task_callback, methods=["POST"])
]

app = Starlette(routes=routes)
```

---

## Error Handling

### Server-Side Errors

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Task, TaskStatus, TaskState, InvalidParamsError, InternalError
from a2a.utils.errors import ServerError
from a2a.utils.message import new_agent_text_message

class RobustAgent(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        try:
            # Validate input
            if not context.message or not context.message.parts:
                raise ServerError(error=InvalidParamsError(message="No message provided"))
            
            user_text = context.message.parts[0].root.text
            
            # Validate content length
            if len(user_text) > 10000:
                raise ServerError(error=InvalidParamsError(message="Message too long"))
            
            # Process
            result = await self.process(user_text)
            
            # Complete
            await event_queue.put(Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=new_agent_text_message(result, context.context_id, context.task_id)
                )
            ))
            
        except ServerError:
            # Re-raise A2A protocol errors
            raise
        
        except Exception as e:
            # Handle unexpected errors
            await event_queue.put(Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message(
                        f"Internal error: {str(e)}",
                        context.context_id,
                        context.task_id
                    )
                )
            ))
```

### Client-Side Error Handling

```python
from a2a.client.errors import A2AClientError, A2AClientJSONRPCError

async def handle_errors(client, message):
    """Handle errors when calling agent."""
    
    try:
        async for event in client.send_message(message):
            if isinstance(event, tuple):
                task, update = event
                
                if task.status.state == TaskState.failed:
                    error_msg = task.status.message.parts[0].root.text
                    print(f"Task failed: {error_msg}")
                    
    except A2AClientJSONRPCError as e:
        print(f"JSON-RPC Error: {e.code} - {e.message}")
        
    except A2AClientError as e:
        print(f"A2A Client Error: {e}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
```

---

## Production Deployment

### Cloud Run Example

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/my-agent', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/my-agent']
  
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'my-agent'
      - '--image=gcr.io/$PROJECT_ID/my-agent'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--port=8080'
      - '--memory=512Mi'
      - '--cpu=1'
      - '--timeout=300'
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Cloud Run provides PORT env var
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run server
CMD ["sh", "-c", "python __main__.py --host 0.0.0.0 --port ${PORT}"]
```

### Environment Variables

```bash
# Cloud Run environment variables
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-key
GOOGLE_API_KEY=your-gemini-key

# Agent URLs (for inter-agent communication)
PROJECT_DECOMP_AGENT_URL=https://project-decomp-agent.run.app
SELECT_FREELANCER_AGENT_URL=https://freelancer-agent.run.app

# Public URL (for agent card)
MY_AGENT_PUBLIC_URL=https://my-agent.run.app
```

---

## Complete Example: FlashBrain A2A Integration

### Server (FlashBrain Orchestrator)

```python
# __main__.py
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities

from FlashBrain import FlashBrainReActAgent
from agent_executor import FlashBrainAgentExecutor

# Initialize agent
flashbrain_agent = FlashBrainReActAgent()
agent_executor = FlashBrainAgentExecutor(agent=flashbrain_agent)

# Create agent card
agent_card = AgentCard(
    name="FlashBrain Orchestrator",
    description="AI orchestrator for project management",
    url="https://flashbrain.example.com",
    version="2.0.0",
    capabilities=AgentCapabilities(streaming=True)
)

# Create server
handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=InMemoryTaskStore())
server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

# Run
app = server.build()
uvicorn.run(app, host='0.0.0.0', port=8010)
```

### Agent Executor

```python
# agent_executor.py
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message, new_task

class FlashBrainAgentExecutor(AgentExecutor):
    def __init__(self, agent):
        self.agent = agent
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        # Stream responses from agent
        async for item in self.agent.stream(query, context_id=task.context_id):
            is_complete = item.get('is_task_complete', False)
            requires_input = item.get('require_user_input', False)
            content = item.get('content', '')
            
            if requires_input:
                # User input needed
                await updater.update_status(
                    TaskState.input_required,
                    new_agent_text_message(content, task.context_id, task.id),
                    final=True
                )
                break
            elif is_complete:
                # Task complete
                await updater.add_artifact([Part(root=TextPart(text=content))], name='response')
                await updater.complete()
                break
            else:
                # In progress
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(content, task.context_id, task.id),
                    final=False
                )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        # Handle cancellation
        pass
```

### Client (Calling FlashBrain)

```python
# client.py
import asyncio
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import Message, Part, TextPart, Role
from uuid import uuid4

async def call_flashbrain():
    async with httpx.AsyncClient() as httpx_client:
        # Connect
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url="https://flashbrain.example.com"
        )
        agent_card = await resolver.get_agent_card()
        
        client_config = ClientConfig(httpx_client=httpx_client)
        client = await ClientFactory.connect(agent=agent_card, client_config=client_config)
        
        # Send message
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="Create a project plan for my app"))],
            message_id=uuid4().hex,
            metadata={'client_id': 'user_123'}
        )
        
        # Process response
        async for event in client.send_message(message):
            if isinstance(event, tuple):
                task, update = event
                
                if task.status.message:
                    text = task.status.message.parts[0].root.text
                    print(text)
                
                if task.status.state == "input_required":
                    # Agent needs input!
                    print("Agent is asking for clarification")
                    # Collect user input and send as next message
                elif task.status.state == "completed":
                    print("Task completed!")
                    break

asyncio.run(call_flashbrain())
```

---

## Best Practices

### 1. Always Handle Task States

```python
async for event in client.send_message(message):
    if isinstance(event, tuple):
        task, update = event
        
        state = task.status.state
        
        if state == TaskState.pending:
            print("Task queued...")
        elif state == TaskState.working:
            print("Task in progress...")
        elif state == TaskState.input_required:
            print("Waiting for user input...")
        elif state == TaskState.completed:
            print("Task complete!")
        elif state == TaskState.failed:
            print("Task failed!")
        elif state == TaskState.canceled:
            print("Task canceled")
```

### 2. Use Context IDs for Multi-Turn Conversations

```python
# First message
context_id = str(uuid4())
message1 = Message(role=Role.user, parts=[...], context_id=context_id)
await client.send_message(message1)

# Follow-up message - SAME context_id
message2 = Message(role=Role.user, parts=[...], context_id=context_id)
await client.send_message(message2)
```

### 3. Implement Proper Cleanup

```python
async def agent_interaction():
    client = None
    try:
        client = await ClientFactory.connect(...)
        # Do work
    finally:
        if client:
            await client.close()
```

### 4. Use Structured Metadata

```python
message = Message(
    role=Role.user,
    parts=[Part(root=TextPart(text="Query"))],
    message_id=uuid4().hex,
    metadata={
        'user_id': 'user_123',
        'session_id': 'session_456',
        'priority': 'high',
        'request_type': 'project_planning'
    }
)
```

### 5. Validate Inputs

```python
class ValidatingAgent(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Validate message exists
        if not context.message or not context.message.parts:
            raise ServerError(error=InvalidParamsError(message="No message provided"))
        
        # Validate content type
        part = context.message.parts[0]
        if not isinstance(part.root, TextPart):
            raise ServerError(error=InvalidParamsError(message="Text input required"))
        
        # Continue...
```

---

## Summary

**A2A Protocol enables:**
- ✅ Agent-to-agent communication
- ✅ Streaming responses
- ✅ Multi-modal messages (text, files, data)
- ✅ Task management and state tracking
- ✅ Human-in-the-loop workflows
- ✅ Push notifications
- ✅ Authentication
- ✅ Dynamic agent discovery

**Core Components:**
- `AgentCard`: Metadata about the agent
- `Message`: Communication unit
- `Task`: Work unit with state
- `AgentExecutor`: Server-side logic
- `ClientFactory`: Client-side connection

**Use Cases:**
- Multi-agent orchestration
- Human-in-the-loop workflows
- Long-running async tasks
- File processing pipelines
- Distributed AI systems

For more details, see the official A2A documentation at https://a2a-protocol.org

