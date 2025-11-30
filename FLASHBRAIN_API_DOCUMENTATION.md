# FlashBrain Orchestrator - API Documentation

## Overview

FlashBrain is an AI orchestrator agent that manages complex workflows including project planning, team selection, and financial operations. This document explains how to integrate a user interface (web, mobile, or desktop) with the FlashBrain agent.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Connection Methods](#connection-methods)
3. [Required Parameters](#required-parameters)
4. [Message Format](#message-format)
5. [Response Format](#response-format)
6. [Authentication](#authentication)
7. [Example Implementations](#example-implementations)
8. [Error Handling](#error-handling)
9. [Best Practices](#best-practices)

---

## Quick Start

### Minimal Example (Python)

```python
import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import AgentCard, Message

async def chat_with_flashbrain(user_message: str, client_id: str, project_id: str = None):
    """Send a message to FlashBrain and receive streaming responses."""
    
    # 1. Connect to FlashBrain
    base_url = "https://flashbrain-orchestrator-935893169019.us-central1.run.app"
    # For local: base_url = "http://localhost:8010"
    
    async with httpx.AsyncClient(timeout=660.0) as httpx_client:
        # Fetch agent card
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        
        # Create A2A client
        client_config = ClientConfig(httpx_client=httpx_client)
        client = await ClientFactory.connect(agent=agent_card, client_config=client_config)
        
        # 2. Create message
        context_id = str(uuid4())  # Unique conversation ID
        
        message = Message(
            role='user',
            context_id=context_id,
            parts=[{'kind': 'text', 'text': user_message}],
            message_id=uuid4().hex,
            metadata={
                'client_id': client_id,      # Required
                'project_id': project_id      # Optional
            }
        )
        
        # 3. Send and stream responses
        async for item in client.send_message(message):
            if isinstance(item, tuple):
                task, update = item
                if hasattr(update, 'status') and update.status:
                    if hasattr(update.status, 'message') and update.status.message:
                        msg = update.status.message
                        if hasattr(msg, 'parts') and msg.parts:
                            for part in msg.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    print(part.root.text, end='', flush=True)

# Usage
asyncio.run(chat_with_flashbrain(
    user_message="Create a project plan for my e-commerce website",
    client_id="9a76be62-0d44-4a34-913d-08dcac008de5",
    project_id="058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc"
))
```

---

## Connection Methods

FlashBrain supports the **Agent-to-Agent (A2A) Protocol v0.3.0** for communication.

### Supported Transports

1. **JSON-RPC over HTTP** (Preferred)
   - URL: `https://flashbrain-orchestrator-935893169019.us-central1.run.app`
   - Protocol: A2A v0.3.0
   - Features: Streaming, Push Notifications

2. **WebSocket** (Optional)
   - URL: `wss://flashbrain-orchestrator-935893169019.us-central1.run.app/ws`
   - Use for real-time bidirectional communication

### Agent Card

Fetch the agent capabilities before connecting:

```http
GET /.well-known/agent-card.json
```

**Response:**
```json
{
  "name": "FlashBrain Orchestrator",
  "version": "1.0.0",
  "protocolVersion": "0.3.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true
  },
  "defaultInputModes": ["text", "application/json", "application/pdf", ...],
  "defaultOutputModes": ["text", "application/json", ...],
  "skills": [
    {
      "id": "flashbrain_orchestrator",
      "name": "FlashBrain Orchestrator",
      "description": "Orchestrates project planning, team selection, and financial operations"
    }
  ]
}
```

---

## Required Parameters

### 1. `client_id` (REQUIRED)

- **Type**: UUID string
- **Purpose**: Identifies the authenticated user/organization
- **Used For**: 
  - Authentication and authorization
  - Database access control (RLS policies)
  - Tracking user-specific data
- **Example**: `"9a76be62-0d44-4a34-913d-08dcac008de5"`
- **Where to Get**: From your authentication system (maps to `customers.id` in Supabase)

### 2. `context_id` (REQUIRED)

- **Type**: UUID string
- **Purpose**: Identifies the conversation thread
- **Used For**:
  - Conversation continuity (message history)
  - State persistence across multiple requests
  - Multi-turn conversations
- **Example**: `"f77fb319-f01f-46a5-8d99-d031b625cca0"`
- **Where to Get**: 
  - Generate new UUID for new conversations
  - Reuse same UUID to continue existing conversation
- **Storage**: Maps to `brain_conversations.id` in Supabase

### 3. `project_id` (OPTIONAL)

- **Type**: UUID string
- **Purpose**: Links the conversation to a specific project
- **Used For**:
  - Project-specific operations (decomposition, team selection)
  - Fetching project data from database
  - Tracking which project is being discussed
- **Example**: `"058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc"`
- **Where to Get**: 
  - From your project management system
  - Created automatically when user requests project planning
- **Storage**: Maps to `project_decomposition.project_id` in Supabase

### 4. `job_id` (INTERNAL - Auto-Generated)

- **Type**: UUID string
- **Purpose**: Tracks async long-running operations
- **Used For**:
  - Polling job status
  - Webhook callbacks
  - Resuming suspended workflows
- **Example**: `"c08b5f41-b582-4967-88e8-7ed8e7dfbd02"`
- **Where to Get**: Automatically created by FlashBrain (don't pass this as input)
- **Storage**: Maps to `brain_jobs.id` in Supabase

---

## Message Format

### A2A Message Structure

```python
{
    "role": "user",               # Always "user" for input
    "context_id": "uuid-string",  # Conversation identifier
    "parts": [                    # Message content (can be multimodal)
        {
            "kind": "text",
            "text": "Your message here"
        }
    ],
    "message_id": "unique-hex",   # Unique message identifier
    "metadata": {                 # Optional metadata
        "client_id": "uuid-string",     # REQUIRED
        "project_id": "uuid-string"     # OPTIONAL
    }
}
```

### Supported Content Types

**Input:**
- `text/plain` - Plain text messages
- `application/json` - Structured data
- `application/pdf` - PDF documents
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document` - Word docs
- `text/markdown` - Markdown formatted text
- `image/jpeg`, `image/png`, `image/webp` - Images

**Output:**
- Same as input types

---

## Response Format

### Streaming Events

FlashBrain sends responses as a stream of events:

```python
{
    'is_task_complete': bool,      # True when task is fully done
    'require_user_input': bool,    # True if waiting for user
    'content': str,                # The actual message content
    'streaming': bool              # (Optional) True for partial/streaming tokens
}
```

### Event Types

**1. Status Updates (in-progress)**
```python
{
    'is_task_complete': False,
    'require_user_input': False,
    'content': '🎯 Routing your request...'
}
```

**2. Streaming Content (partial)**
```python
{
    'is_task_complete': False,
    'require_user_input': False,
    'content': 'Project management is the...',
    'streaming': True  # May be present
}
```

**3. Final Response (complete)**
```python
{
    'is_task_complete': True,
    'require_user_input': False,
    'content': 'Complete response text here...'
}
```

**4. Error / User Input Required**
```python
{
    'is_task_complete': False,
    'require_user_input': True,
    'content': 'An error occurred: ...'
}
```

### Multi-Intent Responses

For requests with multiple tasks, you'll receive results sequentially:

```python
# Task 1
{'content': '**Task 1/2**: What are agile principles?\n\n[Answer here...]'}

# Task 2
{'content': '**Task 2/2**: Create project plan\n\n🚀 Starting your project plan...'}

# Task 2 completion (after async job finishes)
{'content': '✅ Project plan is ready! Now finding freelancers...'}

# Task 2 result
{'content': '🔍 Found 5 qualified developers...'}
```

---

## Authentication

### Client ID Validation

FlashBrain validates `client_id` against the `customers` table in Supabase:

```sql
-- Your database should have:
CREATE TABLE customers (
    id UUID PRIMARY KEY,
    name TEXT,
    email TEXT,
    -- other fields...
);
```

**Security:**
- Client ID is used in Row Level Security (RLS) policies
- Only data belonging to that client is accessible
- Invalid client IDs result in 401 Unauthorized

### API Key (For Direct HTTP Calls)

If calling sub-agents directly (not recommended), include:

```http
POST /endpoint
Content-Type: application/json
x-api-key: {client_id}

{
  "project_id": "...",
  "data": {...}
}
```

---

## Example Implementations

### 1. React/TypeScript Frontend

```typescript
import { v4 as uuidv4 } from 'uuid';

interface FlashBrainMessage {
  is_task_complete: boolean;
  require_user_input: boolean;
  content: string;
  streaming?: boolean;
}

class FlashBrainClient {
  private baseUrl: string;
  private clientId: string;
  
  constructor(clientId: string, baseUrl: string = 'https://flashbrain-orchestrator-935893169019.us-central1.run.app') {
    this.baseUrl = baseUrl;
    this.clientId = clientId;
  }
  
  async *sendMessage(
    message: string, 
    contextId: string,
    projectId?: string
  ): AsyncGenerator<FlashBrainMessage> {
    
    const payload = {
      role: 'user',
      context_id: contextId,
      parts: [{ kind: 'text', text: message }],
      message_id: uuidv4().replace(/-/g, ''),
      metadata: {
        client_id: this.clientId,
        project_id: projectId
      }
    };
    
    const response = await fetch(`${this.baseUrl}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const event = JSON.parse(line);
          yield event as FlashBrainMessage;
        } catch (e) {
          // Skip malformed JSON
        }
      }
    }
  }
}

// Usage in React component
function ChatInterface() {
  const [messages, setMessages] = useState<string[]>([]);
  const clientId = useAuth().userId; // From your auth system
  const contextId = useConversationId(); // Generate or load from storage
  const client = new FlashBrainClient(clientId);
  
  async function sendMessage(text: string) {
    let currentMessage = '';
    
    for await (const event of client.sendMessage(text, contextId)) {
      if (event.streaming) {
        // Streaming token - update in place
        currentMessage = event.content;
        setMessages(prev => [...prev.slice(0, -1), currentMessage]);
      } else {
        // Complete message - add to history
        setMessages(prev => [...prev, event.content]);
      }
      
      if (event.is_task_complete) {
        break; // Done
      }
    }
  }
  
  return (
    <div>
      {messages.map((msg, i) => <div key={i}>{msg}</div>)}
      <input onSubmit={e => sendMessage(e.target.value)} />
    </div>
  );
}
```

### 2. Python Client (Simple)

```python
import asyncio
import httpx
from uuid import uuid4

async def send_to_flashbrain(message: str, client_id: str, project_id: str = None):
    """Simple Python client for FlashBrain."""
    
    base_url = "https://flashbrain-orchestrator-935893169019.us-central1.run.app"
    context_id = str(uuid4())  # New conversation
    
    payload = {
        "role": "user",
        "context_id": context_id,
        "parts": [{"kind": "text", "text": message}],
        "message_id": uuid4().hex,
        "metadata": {
            "client_id": client_id,
            "project_id": project_id
        }
    }
    
    async with httpx.AsyncClient(timeout=660.0) as client:
        async with client.stream("POST", f"{base_url}/messages", json=payload) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        event = json.loads(line)
                        print(event['content'])
                        
                        if event['is_task_complete']:
                            break
                    except:
                        pass

# Usage
asyncio.run(send_to_flashbrain(
    message="Help me plan my project",
    client_id="9a76be62-0d44-4a34-913d-08dcac008de5"
))
```

### 3. cURL (Testing)

```bash
curl -X POST https://flashbrain-orchestrator-935893169019.us-central1.run.app/messages \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "context_id": "test-123",
    "parts": [
      {
        "kind": "text",
        "text": "What is project management?"
      }
    ],
    "message_id": "msg-456",
    "metadata": {
      "client_id": "9a76be62-0d44-4a34-913d-08dcac008de5"
    }
  }' \
  --no-buffer
```

---

## Required Parameters

### Parameter Reference Table

| Parameter | Required | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `client_id` | ✅ Yes | UUID | - | Authenticated user/organization ID |
| `context_id` | ✅ Yes | UUID | - | Conversation thread ID |
| `message` | ✅ Yes | string | - | User's input text |
| `project_id` | ❌ No | UUID | `null` | Current project being discussed |
| `message_id` | ✅ Yes | string | - | Unique message identifier |

### Parameter Details

#### `client_id`

**Required:** Yes (Always)

**Purpose:** Identifies the authenticated user or organization making the request.

**Usage:**
```python
metadata={
    'client_id': 'uuid-from-your-auth-system'
}
```

**How to Get:**
- **Web App**: From your authentication system (Auth0, Firebase, Supabase Auth, etc.)
- **Mobile App**: From user login flow
- **API**: From your user database

**Example Flow:**
```javascript
// After user logs in
const user = await auth.currentUser();
const clientId = user.id; // Use this for all FlashBrain requests
```

**Security:**
- FlashBrain uses this for Row Level Security (RLS)
- User can only access their own projects and data
- Invalid `client_id` returns 401 Unauthorized

#### `context_id`

**Required:** Yes (Always)

**Purpose:** Maintains conversation continuity across multiple messages.

**Usage:**
```python
# New conversation
context_id = str(uuid4())

# Continue existing conversation
context_id = "f77fb319-f01f-46a5-8d99-d031b625cca0"  # Saved from previous interaction
```

**How to Get:**
- **New Conversation**: Generate a new UUID
- **Continue Conversation**: Load from browser localStorage, database, or session storage

**Example Flow (Web):**
```javascript
// Start new conversation
let contextId = localStorage.getItem('currentConversation');
if (!contextId) {
  contextId = crypto.randomUUID();
  localStorage.setItem('currentConversation', contextId);
}

// Send message with this contextId
await sendToFlashBrain(message, contextId);
```

**Storage:**
- Each `context_id` maps to a record in `brain_conversations` table
- Message history stored in `brain_messages` table
- FlashBrain remembers previous messages in the same context

#### `project_id`

**Required:** No (Optional)

**Purpose:** Links the conversation to a specific project for context-aware responses.

**When to Include:**
- User is discussing an existing project
- User wants to update/modify a project
- User asks about project-specific data (team, budget, timeline)

**When to Omit:**
- General questions ("What is project management?")
- Creating a brand new project (FlashBrain will generate one)
- Questions not related to a specific project

**Example:**
```python
# User selects project from dropdown
selected_project = "058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc"

# Include in metadata
metadata = {
    'client_id': client_id,
    'project_id': selected_project  # Optional but helpful
}
```

**Auto-Creation:**
If user says "Create a project plan for X" without providing `project_id`, FlashBrain will:
1. Create a new project in `project_decomposition` table
2. Return the new `project_id` in the response
3. Use that `project_id` for subsequent operations

#### `message_id`

**Required:** Yes (For deduplication)

**Purpose:** Uniquely identifies each message to prevent duplicates.

**Usage:**
```python
import uuid
message_id = uuid.uuid4().hex  # Generates: "a1b2c3d4e5f6..."
```

**Format:** Any unique string (typically UUID without hyphens)

---

## Message Format

### Full Message Structure

```python
from a2a.types import Message

message = Message(
    role='user',                    # Always 'user' for client messages
    context_id='uuid-string',       # Required: conversation ID
    parts=[                         # Required: message content
        {
            'kind': 'text',         # Content type
            'text': 'Your message'  # Actual content
        }
    ],
    message_id='unique-id',         # Required: message identifier
    metadata={                      # Optional but recommended
        'client_id': 'uuid',        # Required for auth
        'project_id': 'uuid'        # Optional for context
    }
)
```

### Multimodal Messages

FlashBrain supports multiple content types in a single message:

```python
parts=[
    {
        'kind': 'text',
        'text': 'Here is my project proposal:'
    },
    {
        'kind': 'file',
        'mimeType': 'application/pdf',
        'data': base64_encoded_pdf,
        'name': 'proposal.pdf'
    },
    {
        'kind': 'image',
        'mimeType': 'image/jpeg',
        'data': base64_encoded_image,
        'name': 'mockup.jpg'
    }
]
```

---

## Response Format

### A2A Response Structure

Responses are streamed as a series of **Task Updates**:

```python
(Task, TaskStatusUpdate)
```

**Task:**
```python
{
    'id': 'task-uuid',
    'context_id': 'conversation-uuid',
    'state': 'working' | 'completed' | 'input_required' | 'failed'
}
```

**TaskStatusUpdate:**
```python
{
    'status': {
        'state': 'working',
        'message': {
            'parts': [
                {
                    'root': {
                        'text': 'Response content here'
                    }
                }
            ]
        }
    }
}
```

### Response States

| State | Meaning | Action |
|-------|---------|--------|
| `working` | Agent is processing | Display to user, keep waiting |
| `completed` | Task finished successfully | Display final result |
| `input_required` | Needs user input | Prompt user for more info |
| `failed` | Task failed with error | Display error, allow retry |

---

## Authentication

### Flow Diagram

```
User Login
    ↓
Your Auth System (Auth0/Firebase/etc)
    ↓
Returns user object with ID
    ↓
Map to Supabase customers.id
    ↓
Use as client_id in FlashBrain requests
    ↓
FlashBrain validates against customers table
    ↓
RLS policies enforce data access
```

### Example Auth Integration

```javascript
// 1. User logs in via your system
const user = await auth.signIn(email, password);

// 2. Map to Supabase customer ID
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const { data: customer } = await supabase
  .from('customers')
  .select('id')
  .eq('email', user.email)
  .single();

const clientId = customer.id;

// 3. Store for FlashBrain requests
sessionStorage.setItem('flashbrain_client_id', clientId);

// 4. Use in all requests
await flashbrain.sendMessage(text, {
  client_id: clientId,
  context_id: conversationId
});
```

---

## Example Implementations

### Web App (JavaScript/Fetch API)

```html
<!DOCTYPE html>
<html>
<head>
    <title>FlashBrain Chat</title>
</head>
<body>
    <div id="chat"></div>
    <input id="input" type="text" />
    <button onclick="sendMessage()">Send</button>

    <script>
        const CLIENT_ID = '9a76be62-0d44-4a34-913d-08dcac008de5';
        const CONTEXT_ID = crypto.randomUUID();
        const BASE_URL = 'https://flashbrain-orchestrator-935893169019.us-central1.run.app';
        
        async function sendMessage() {
            const input = document.getElementById('input').value;
            const chat = document.getElementById('chat');
            
            // Add user message to UI
            chat.innerHTML += `<div><strong>You:</strong> ${input}</div>`;
            
            // Prepare FlashBrain message
            const payload = {
                role: 'user',
                context_id: CONTEXT_ID,
                parts: [{ kind: 'text', text: input }],
                message_id: crypto.randomUUID().replace(/-/g, ''),
                metadata: {
                    client_id: CLIENT_ID,
                    project_id: null
                }
            };
            
            // Stream response
            const response = await fetch(`${BASE_URL}/messages`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantDiv = document.createElement('div');
            assistantDiv.innerHTML = '<strong>FlashBrain:</strong> ';
            chat.appendChild(assistantDiv);
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(l => l.trim());
                
                for (const line of lines) {
                    try {
                        const event = JSON.parse(line);
                        assistantDiv.innerHTML += event.content;
                        
                        if (event.is_task_complete) {
                            break;
                        }
                    } catch (e) {}
                }
            }
        }
    </script>
</body>
</html>
```

### Mobile App (React Native)

```typescript
import { v4 as uuidv4 } from 'uuid';

export class FlashBrainService {
  private clientId: string;
  private baseUrl: string;
  
  constructor(userId: string) {
    this.clientId = userId;
    this.baseUrl = 'https://flashbrain-orchestrator-935893169019.us-central1.run.app';
  }
  
  async sendMessage(
    message: string,
    contextId: string,
    onChunk: (text: string) => void,
    onComplete: () => void
  ) {
    const payload = {
      role: 'user',
      context_id: contextId,
      parts: [{ kind: 'text', text: message }],
      message_id: uuidv4().replace(/-/g, ''),
      metadata: {
        client_id: this.clientId
      }
    };
    
    const response = await fetch(`${this.baseUrl}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(l => l.trim());
      
      for (const line of lines) {
        try {
          const event = JSON.parse(line);
          onChunk(event.content);
          
          if (event.is_task_complete) {
            onComplete();
            return;
          }
        } catch (e) {}
      }
    }
  }
}

// Usage in component
const flashbrain = new FlashBrainService(user.id);

await flashbrain.sendMessage(
  "Create a project plan",
  conversationId,
  (text) => setStreamingText(text),      // Update UI with each chunk
  () => setIsComplete(true)               // Mark as done
);
```

### Backend API Proxy (Node.js/Express)

```javascript
const express = require('express');
const { v4: uuidv4 } = require('uuid');
const fetch = require('node-fetch');

const app = express();
app.use(express.json());

app.post('/api/chat', async (req, res) => {
  const { message, userId, projectId } = req.body;
  
  // Map your user ID to Supabase client_id
  const clientId = await getUserClientId(userId);
  const contextId = req.session.flashbrainContext || uuidv4();
  req.session.flashbrainContext = contextId;
  
  const payload = {
    role: 'user',
    context_id: contextId,
    parts: [{ kind: 'text', text: message }],
    message_id: uuidv4().replace(/-/g, ''),
    metadata: {
      client_id: clientId,
      project_id: projectId
    }
  };
  
  const response = await fetch(
    'https://flashbrain-orchestrator-935893169019.us-central1.run.app/messages',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }
  );
  
  // Stream response to client
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  for await (const chunk of response.body) {
    res.write(`data: ${chunk}\n\n`);
  }
  
  res.end();
});

app.listen(3000);
```

---

## Error Handling

### Common Errors

#### 1. Authentication Error (401)

```json
{
  "is_task_complete": false,
  "require_user_input": true,
  "content": "Authentication failed: Invalid client_id"
}
```

**Cause:** `client_id` is missing, invalid, or not in `customers` table

**Fix:** 
- Verify `client_id` exists in Supabase `customers` table
- Check that `client_id` is included in metadata
- Ensure user is properly authenticated

#### 2. Connection Timeout

```json
{
  "is_task_complete": false,
  "require_user_input": true,
  "content": "An error occurred: couldn't get a connection after 10.00 sec"
}
```

**Cause:** Database connection pool exhausted

**Fix:**
- Increase client timeout to 660s (11 minutes)
- Check FlashBrain server health
- Verify Supabase connection limits

#### 3. Missing Parameters

```json
{
  "is_task_complete": false,
  "require_user_input": true,
  "content": "Error: client_id is required"
}
```

**Cause:** Required parameter not provided

**Fix:** Ensure all required parameters are included in metadata

### Error Response Format

All errors follow this structure:

```python
{
    'is_task_complete': False,
    'require_user_input': True,  # True indicates error state
    'content': 'Error description here'
}
```

### Retry Logic

```python
async def send_with_retry(message, client_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            async for event in send_to_flashbrain(message, client_id):
                yield event
                
                if event['require_user_input'] and 'error' in event['content'].lower():
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        break  # Retry
                    else:
                        raise Exception(event['content'])
                        
                if event['is_task_complete']:
                    return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
```

---

## Best Practices

### 1. Conversation Management

```python
# Good: Reuse context_id for multi-turn conversations
context_id = load_or_create_conversation_id()

await send_message("Create a project plan", context_id)
# Later in the same conversation...
await send_message("Add more developers", context_id)  # FlashBrain remembers context
```

```python
# Bad: New context_id for every message
await send_message("Create a project plan", uuid4())
await send_message("Add more developers", uuid4())  # No memory of previous request
```

### 2. Timeout Configuration

**Client Timeout:** Always > FlashBrain's max job duration

```python
# Good
httpx.AsyncClient(timeout=660.0)  # 11 minutes (covers 10 min max job)

# Bad
httpx.AsyncClient(timeout=30.0)  # Will timeout during long jobs
```

### 3. Handle Async Jobs

For long-running tasks (project decomposition), FlashBrain may:
1. Start the job and return status
2. Poll internally (Option 1 implementation)
3. Stream final result when complete

**Your UI should:**
- Show progress indicators
- Keep connection alive (don't timeout)
- Handle "still working..." messages

### 4. Project ID Management

```python
# Track current project in your app state
current_project = None

# When user creates project
response = await send_message("Create project plan for X", client_id)
if "project_id" in response:
    current_project = extract_project_id(response)
    save_to_session(current_project)

# Use for subsequent requests
await send_message("Find developers", client_id, project_id=current_project)
```

### 5. Context Persistence

```javascript
// Store conversation history
const conversationStore = {
  save: (contextId, messages) => {
    localStorage.setItem(`chat_${contextId}`, JSON.stringify(messages));
  },
  
  load: (contextId) => {
    return JSON.parse(localStorage.getItem(`chat_${contextId}`) || '[]');
  },
  
  list: () => {
    // Get all conversations for sidebar
    return Object.keys(localStorage)
      .filter(k => k.startsWith('chat_'))
      .map(k => k.replace('chat_', ''));
  }
};
```

---

## Advanced Features

### 1. Multi-Intent Requests

FlashBrain can handle multiple tasks in one request:

```python
message = """Please help me with:
1. Create a project plan for my e-commerce site
2. Find React developers for it
3. Estimate the budget
"""

# FlashBrain will:
# 1. Detect 3 separate tasks
# 2. Execute them in priority order
# 3. Handle dependencies (Task 2 waits for Task 1)
# 4. Stream all results back in sequence
```

**Response:**
```
🎯 Routing your request...
📋 Detected 3 tasks to complete...
⚙️ Processing task 1/3: Create project plan
🚀 Starting your project plan... This will take about 5 minutes.
[5 minutes later]
✅ Project plan is ready! Now finding freelancers...
⚙️ Processing task 2/3: Find React developers
🔍 Searching for freelancers...
Found 5 qualified React developers...
⚙️ Processing task 3/3: Estimate budget
Based on the data, estimated budget is $45,000
```

### 2. Push Notifications (Future)

Enable push notifications to receive updates when long jobs complete:

```python
# Subscribe to notifications
await client.subscribe_to_push_notifications(
    context_id=context_id,
    webhook_url="https://your-app.com/webhooks/flashbrain"
)

# Later, when job completes, receive webhook:
@app.post('/webhooks/flashbrain')
async def handle_notification(payload):
    context_id = payload['context_id']
    result = payload['result']
    
    # Notify user via WebSocket, email, etc.
    await notify_user(context_id, result)
```

### 3. File Uploads

Send documents for analysis:

```python
import base64

with open('proposal.pdf', 'rb') as f:
    pdf_data = base64.b64encode(f.read()).decode()

message = Message(
    role='user',
    context_id=context_id,
    parts=[
        {
            'kind': 'text',
            'text': 'Analyze this project proposal'
        },
        {
            'kind': 'file',
            'mimeType': 'application/pdf',
            'data': pdf_data,
            'name': 'proposal.pdf'
        }
    ],
    metadata={'client_id': client_id}
)
```

---

## Testing

### Local Testing

**1. Start FlashBrain locally:**
```bash
cd "Agents/FlashBrain Orchestrator"
python -m __main__ --host localhost --port 8010
```

**2. Test with Python script:**
```bash
python test_multi_intent.py
```

**3. Test with cURL:**
```bash
curl -X POST http://localhost:8010/messages \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "context_id": "test-123",
    "parts": [{"kind": "text", "text": "Hello"}],
    "message_id": "msg-456",
    "metadata": {"client_id": "9a76be62-0d44-4a34-913d-08dcac008de5"}
  }'
```

### Production Testing

Use the Cloud Run URL:
```
https://flashbrain-orchestrator-935893169019.us-central1.run.app
```

---

## Environment-Specific URLs

### Development
```
http://localhost:8010
```

### Staging (if applicable)
```
https://flashbrain-staging-xyz.run.app
```

### Production
```
https://flashbrain-orchestrator-935893169019.us-central1.run.app
```

**Recommendation:** Store as environment variable in your frontend:

```javascript
const FLASHBRAIN_URL = process.env.REACT_APP_FLASHBRAIN_URL || 'http://localhost:8010';
```

---

## Rate Limits & Quotas

### Current Limits

- **Requests per minute**: 60 per client_id
- **Concurrent connections**: 10 per client_id
- **Message size**: 10 MB max
- **Conversation history**: 50 messages (older messages summarized)

### Handling Rate Limits

If you receive a 429 error:

```python
{
    'is_task_complete': False,
    'require_user_input': True,
    'content': 'Rate limit exceeded. Please try again in 60 seconds.'
}
```

**Implement exponential backoff:**
```python
async def send_with_backoff(message, client_id):
    for delay in [1, 2, 4, 8]:
        try:
            return await send_message(message, client_id)
        except RateLimitError:
            await asyncio.sleep(delay)
    raise Exception("Rate limit exceeded after retries")
```

---

## Monitoring & Observability

### Track These Metrics

1. **Response Time**: Time from request to first chunk
2. **Completion Time**: Time to `is_task_complete: true`
3. **Error Rate**: Percentage of `require_user_input: true` responses
4. **Token Usage**: Track from response metadata (if available)

### Example Logging

```python
import time

start_time = time.time()
first_chunk_time = None
chunks_received = 0

async for event in send_to_flashbrain(message, client_id):
    chunks_received += 1
    
    if first_chunk_time is None:
        first_chunk_time = time.time() - start_time
        logger.info(f"Time to first chunk: {first_chunk_time:.2f}s")
    
    if event['is_task_complete']:
        total_time = time.time() - start_time
        logger.info(f"Total time: {total_time:.2f}s, chunks: {chunks_received}")
```

---

## Troubleshooting

### Problem: No response received

**Check:**
1. Is `client_id` valid? (exists in `customers` table)
2. Is FlashBrain server running?
3. Are you using correct URL (HTTP for local, HTTPS for Cloud Run)?
4. Check server logs for errors

### Problem: Response timeout

**Check:**
1. Client timeout >= 660s
2. Network connectivity
3. FlashBrain server not overloaded

### Problem: "Response length: 0 chars"

**Check:**
1. Streaming event handler is correctly parsing responses
2. A2A client is properly initialized
3. Server logs show response being generated (line 479: "POST brain_messages")

**Debug:** Add logging to see raw events:
```python
async for item in client.send_message(message):
    print(f"Raw event: {item}")  # See what's actually being received
```

---

## Summary Checklist

Before integrating FlashBrain into your UI, ensure you have:

- [ ] `client_id` from your authentication system
- [ ] Logic to generate/reuse `context_id` for conversations
- [ ] Timeout configured to 660s (11 minutes)
- [ ] Streaming response handler implemented
- [ ] Error handling for `require_user_input: true` events
- [ ] Progress indicators for long-running tasks
- [ ] (Optional) `project_id` tracking for project-specific operations

---

## Support

For issues or questions:
- Check server logs at Cloud Run console
- Review `brain_conversations` and `brain_messages` tables in Supabase
- Test with `test_multi_intent.py` to verify server functionality

---

## Appendix: Complete Parameter Reference

```python
{
    # Message envelope
    "role": "user",                          # Fixed value
    "context_id": "uuid-string",             # Required: conversation ID
    "message_id": "unique-string",           # Required: message deduplication
    
    # Content
    "parts": [
        {
            "kind": "text" | "file" | "image",
            "text": "message content",       # For kind="text"
            "data": "base64-encoded",        # For kind="file"/"image"
            "mimeType": "mime/type",         # For kind="file"/"image"
            "name": "filename"               # For kind="file"/"image"
        }
    ],
    
    # Metadata
    "metadata": {
        "client_id": "uuid-string",          # Required: auth
        "project_id": "uuid-string"          # Optional: context
    }
}
```

