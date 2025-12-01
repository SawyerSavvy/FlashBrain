# Digital Twin – FlashBrain ReAct Orchestrator

Current architecture (per `Agents/FlashBrain Orchestrator/FlashBrain.py`) is a single ReAct agent with tools to reach remote A2A agents and the database.

## Core Agent
- **FlashBrainReActAgent (ReAct, LangGraph)**
  - LLM: `ChatGoogleGenerativeAI` (`gemini-2.5-flash`, temperature 0.1).
  - System prompt: `FLASHBRAIN_SYSTEM_PROMPT` + dynamic agent roster.
  - Checkpointing: `AsyncPostgresSaver` using `psycopg_pool.AsyncConnectionPool` (requires `SUPABASE_TRANSACTION_POOLER`).
  - Persistence: Supabase REST client for `brain_conversations`/`brain_messages` (if `SUPABASE_URL`/`SUPABASE_SERVICE_ROLE_KEY` set).
  - Inputs: `messages` (includes optional context system message with `project_id`/`client_id`), `thread_id` (context_id).
  - Outputs: Streamed AI messages (non-tool content), final AI completion; saves to Supabase when available.

## Tools / Sub-Agents
- **send_message (generic A2A caller)**
  - Discovers remote agents from `AGENTS_URLS` using `A2ACardResolver`, builds `RemoteAgentConnections`.
  - Inputs: `agent_name`, `task`, optional `project_id`, `client_id`.
  - Action: Sends A2A `SendMessageRequest` to the selected remote agent and returns artifact text.
  - Output: Text response (concatenated artifacts) or an error string if agent not found.

> Note: Additional tools can be added to `self.tools`; currently the A2A send_message tool is the primary expansion surface.

## External Integrations
- **Remote A2A Agents** (dynamic from `AGENTS_URLS`)
  - Each discovered agent contributes name/description to the system prompt and is callable via `send_message`.
- **Supabase/Postgres**
  - Checkpoints: Postgres via `AsyncPostgresSaver`.
  - Conversation log: Supabase REST tables `brain_conversations`, `brain_messages` (optional, gated by env).

## Workflow Map
1) Frontend/A2A client sends user text (plus optional `project_id`/`client_id`) to `FlashBrainReActAgent.stream(...)` with a `thread_id`.
2) Agent prepends context system message (project/client) and passes messages to the ReAct graph with checkpointing.
3) ReAct agent reasons, may call `send_message` tool to invoke remote A2A agents, and streams AI messages back (`stream_mode="messages"`), skipping tool-call payloads.
4) `stream()` yields incremental content to the caller; final message is saved to Supabase (if configured) and emitted with `is_task_complete=True`.
5) State is persisted via the Postgres checkpointer keyed by `thread_id`.

## Inputs & Outputs
- Inputs: User text, `context_id`/`thread_id`, optional `project_id`, `client_id`.
- Streamed Outputs: AI text chunks (non-tool content); final completion marker.
- Stored Artifacts: Conversation history in Supabase; graph state in Postgres.

## Observability / Considerations
- Streaming depends on Gemini chunking; monitor chunk counts in logs to verify incremental delivery.
- Remote agent discovery is runtime-dynamic via `AGENTS_URLS`; errors there reduce available tools but keep the core agent alive.
- Supabase is optional for history; Postgres checkpointer is required (agent raises if missing).

---

## Project Decomp Agent (`Agents/Project Decomp Graph/project_decomp_agent.py`)
- **Purpose:** Extract project requirements, persist summary, and trigger project decomposition orchestrator.
- **State/Input:** `messages` (conversation), `project_id`, `client_id` (api key header), `job_id` (async tracking), `exist` flag.
- **Outputs:** `extracted_info` summary, Supabase update status, orchestrator response message; streaming status texts.
- **Tools/Subnodes:**
  - `extract_information`: LLM (`gemini-2.5-flash-lite`) produces summary of requirements/updates.
  - `upload_to_supabase`: Writes summary (and `job_id`) to Supabase `project_decomposition`.
  - `call_orchestrator`: HTTP POST to `PROJECT_DECOMP_ORCHESTRATOR_URL` `/project` with `project_id`; uses `client_id` as `x-api-key`.
- **Streaming:** Emits progress messages (“Starting…”, “Writing to Database…”, “Creating Project Plan…”); final status currently minimal.

## Select Freelancer Agent (`Agents/Select Freelancer Graph/select_freelancer_agent.py`)
- **Purpose:** Update project phases/roles from conversation, persist changes, and call freelancer matching/re-optimization.
- **State/Input:** `messages`, `project_id`, `client_id`; fetched `existing_phases`, `existing_roles`.
- **Outputs:** `extracted_updates` (phases/roles deltas), Supabase update result, `api_response` from freelancer service; streamed status.
- **Tools/Subnodes:**
  - `fetch_project_data`: Reads phases/roles from Supabase (`project_phases`, `project_phase_roles`).
  - `extract_requirements`: LLM (`gemini-2.5-flash-lite` with structured output `RequirementsOutput`) compares conversation vs existing state to produce deltas and reasoning.
  - `upload_to_supabase`: Applies phase/role updates back to Supabase (best-effort matching by ids/titles).
  - `call_freelancer_service`: Calls `SELECT_FREELANCER_URL` via HTTP; chooses `/reoptimize-targets` when deltas present else `/match-all-freelancers`.
- **Streaming:** Emits stepwise progress (“Fetching project data…”, “Analyzing…”, “Extracting requirements…”, “Updating database…”, “Finding personal…”), then final summary or error/need-input.

---

## Architecture Map (Connections)
```
User / A2A client
     |
     v
FlashBrainReActAgent (Gemini-Flash, ReAct)
     |---------------> Postgres (checkpoints via AsyncPostgresSaver)
     |---------------> Supabase REST (conversation logs; optional)
     |
     |--- send_message (A2A tool) --- Project Decomp Agent
     |                                |- LLM extraction
     |                                |- Supabase update: project_decomposition
     |                                `- HTTP -> PROJECT_DECOMP_ORCHESTRATOR_URL
     |
     `--- send_message (A2A tool) --- Select Freelancer Agent
                                      |- Supabase read: project_phases, project_phase_roles
                                      |- LLM requirements (structured)
                                      |- Supabase update: phases/roles
                                      `- HTTP -> SELECT_FREELANCER_URL
```
Notes:
- No WebSocket layer in the current flow; clients use A2A or direct stream/invoke.
- Tool calls are hidden from streamed output unless their text is returned; streaming delivers non-tool AI content back to the caller.
