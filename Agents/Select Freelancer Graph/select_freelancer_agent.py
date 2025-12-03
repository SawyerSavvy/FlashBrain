"""
Select Freelancer ReAct Agent

A simplified ReAct agent for managing freelancers and project roles.
Supports full CRUD operations: Create, Read, Update, Delete.

Architecture:
    - Uses LangChain's create_agent() for automatic ReAct loop
    - Tools for all freelancer/role operations
    - LLM naturally maps user intent to correct operations
    - Much simpler than custom graph pipeline
"""

from typing import Optional, List, Dict, Any
import os
import json
import requests
from dotenv import load_dotenv
import logging

load_dotenv(override=True)

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from supabase import create_client, Client
except ImportError:
    print("Warning: supabase package not found. Please install it with `pip install supabase`.")
    create_client = None
    Client = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- System Prompt ---
SYSTEM_PROMPT = """You are a Project Role Management Assistant.

You help users manage project phases and roles with these capabilities:
- **Add role slots** to project phases
- **Update role requirements** (skills, budget, description, specialized name)
- **Assign freelancers** to role slots (set freelancer_id)
- **Remove role slots** from project
- **Get role information** (requirements, assignments)
- **Get phase information** (phase details, tasks)
- **Update phase details** (title, description, tasks, budget)
- **Match freelancers** to roles using AI matching service

When the user asks to do something, use the appropriate tool. Be conversational and helpful.

**CRITICAL: Use Context Clues and Tools to Infer UUIDs and Identifiers**

Users will often refer to entities by their names, descriptions, or context rather than UUIDs. You MUST use available tools and context clues to automatically infer the required UUIDs (role_slot_id, phase_id, project_id, etc.) instead of asking the user.

**General Pattern for Inferring Identifiers:**

1. **Role Names → role_slot_id**: When a user mentions a role by name (e.g., "Security Engineer", "Backend Developer"):
   - Use `find_role_slot_by_name(project_id, role_name)` to find the role_slot_id
   - If exactly one match: use it automatically
   - If multiple matches: use the most relevant one based on context (phase, specialized name, etc.)
   - Never ask the user for role_slot_id if you can find it via tools

2. **Phase Names/Numbers → phase_id**: When a user mentions a phase:
   - Use `get_phase_info(project_id)` to get all phases
   - Match by phase_title, phase_number, or description
   - Use the matching phase_id automatically

3. **Project Context → project_id**: 
   - Extract project_id from conversation context (provided in SystemMessage or previous messages)
   - Use `get_project_data(project_id)` to understand the project structure

4. **Role Slots → role_slot_id**:
   - Use `get_role_info(project_id)` to see all role slots
   - Match by role_name, specialized_role_name, phase, or other context clues
   - Use `find_role_slot_by_name` for name-based matching

**Workflow Examples:**

- User: "The Security Engineer needs AI experience"
  → Infer: Use `find_role_slot_by_name(project_id, "Security Engineer")` to get role_slot_id
  → Action: Call `update_role_requirements(role_slot_id=<found_id>, role_skills=["AI", "experience"])`
  → Don't ask: Never ask "What is the role_slot_id?" - find it yourself!

- User: "Update Phase 1's budget to 30%"
  → Infer: Use `get_phase_info(project_id)` to find phase_id where phase_number=1
  → Action: Call `update_phase(phase_id=<found_id>, phase_budget_pct=0.3)`

- User: "What skills does the Backend Developer role need?"
  → Infer: Use `find_role_slot_by_name(project_id, "Backend Developer")` or `get_role_info(project_id)`
  → Action: Extract and present the role_skills from the result

- User: "Assign freelancer abc-123 to the Frontend Developer in Phase 2"
  → Infer: 
    1. Use `get_phase_info(project_id)` to find phase_id for phase_number=2
    2. Use `find_role_slot_by_name(project_id, "Frontend Developer", phase_id=<found>)` to get role_slot_id
  → Action: Call `assign_freelancer_to_role(role_slot_id=<found_id>, freelancer_id="abc-123")`

**Key Principles:**

1. **Always infer first**: Use tools to find UUIDs before asking the user
2. **Use context**: Leverage phase numbers, role names, descriptions, and other context clues
3. **Be smart about ambiguity**: If multiple matches exist, use the most contextually relevant one
4. **Only ask when necessary**: Only ask the user for clarification if:
   - No matches are found AND you've exhausted all search options
   - Multiple equally valid matches exist AND context doesn't help disambiguate
   - The user's request is genuinely ambiguous

5. **Chain tool calls**: Don't hesitate to call multiple tools in sequence to gather information:
   - `get_project_data` → understand project structure
   - `get_phase_info` → find phases
   - `get_role_info` → see all roles
   - `find_role_slot_by_name` → find specific role slots

Important: You work with project_phases and project_phase_roles tables ONLY.
You cannot create or modify freelancers themselves (they exist in a separate system).

Examples:
- "Add a Backend Developer role to Phase 1" → use get_phase_info to find phase_id, then add_role_slot
- "The Security Engineer needs AI experience" → use find_role_slot_by_name, then update_role_requirements
- "Assign freelancer abc-123 to the Backend Developer role" → use find_role_slot_by_name, then assign_freelancer_to_role
- "What skills does the Security Engineer role need?" → use find_role_slot_by_name or get_role_info
- "Find freelancers for this project" → use match_freelancers
"""


class SelectFreelancerReActAgent:
    """
    Simplified Select Freelancer agent using create_agent.

    Replaces the complex 4-node pipeline with a ReAct agent that
    intelligently chooses tools based on user requests.
    """

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/json']

    def __init__(self):
        """Initialize the ReAct agent with tools."""
        logger.info("Initializing Select Freelancer ReAct Agent")

        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase_client = None

        if self.supabase_url and self.supabase_key and create_client:
            try:
                self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1
        )

        # Create tools
        self.tools = self._create_tools()

        # Create ReAct agent using create_agent
        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT
        )

        logger.info(f"ReAct agent created with {len(self.tools)} tools")

    def _create_tools(self) -> List:
        """Create all tools for project role management."""
        return [
            self._create_add_role_slot_tool(),
            self._create_update_role_requirements_tool(),
            self._create_remove_role_slot_tool(),
            self._create_get_all_roles_tool(),
            self._create_get_role_info_tool(),
            self._create_get_phase_info_tool(),
            self._create_get_project_data_tool(),
            self._create_update_phase_tool(),
            self._create_match_freelancers_tool(),
        ]

    # --- Tool Definitions ---

    def _create_add_role_slot_tool(self):
        """Tool for adding a new role slot to a phase."""
        supabase = self.supabase_client

        @tool
        def add_role_slot(
            project_id: str,
            phase_id: str,
            role_id: str,
            role_slot: int,
            role_skills: Optional[List[str]] = None,
            role_description: Optional[str] = None,
            specialized_role_name: Optional[str] = None,
            role_budget_pct_phase: Optional[float] = None,
            supplied_by: str = "platform"
        ) -> str:
            """
            Add a new role slot to a project phase.

            Args:
                project_id: UUID of the project
                phase_id: UUID of the phase
                role_id: UUID of the role type (from roles table)
                role_slot: Slot number for this role (e.g., 1 for first Backend Developer)
                role_skills: List of required skills (optional)
                role_description: Description of the role requirements (optional)
                specialized_role_name: Specialized name for this role slot (optional)
                role_budget_pct_phase: Budget percentage within phase (optional)
                supplied_by: Who supplies this role - 'client' or 'platform' (default: platform)

            Returns:
                Confirmation message with role slot ID
            """
            if not supabase:
                return "Error: Supabase not configured"

            try:
                insert_data = {
                    "project_id": project_id,
                    "phase_id": phase_id,
                    "role_id": role_id,
                    "role_slot": role_slot,
                    "supplied_by": supplied_by
                }

                # Add optional fields
                if role_skills:
                    insert_data["role_skills"] = role_skills
                if role_description:
                    insert_data["role_description"] = {"description": role_description}
                if specialized_role_name:
                    insert_data["specialized_role_name"] = specialized_role_name
                if role_budget_pct_phase is not None:
                    insert_data["role_budget_pct_phase"] = role_budget_pct_phase

                result = supabase.table("project_phase_roles").insert(insert_data).execute()

                role_slot_id = result.data[0]['id'] if result.data else "unknown"
                return f"✅ Added role slot #{role_slot} with ID {role_slot_id}. Skills: {', '.join(role_skills) if role_skills else 'Not specified'}"

            except Exception as e:
                logger.error(f"Failed to add role slot: {e}")
                return f"❌ Failed to add role slot: {str(e)}"

        return add_role_slot

    def _create_update_role_requirements_tool(self):
        """Tool for updating role requirements."""
        supabase = self.supabase_client

        @tool
        def update_role_requirements(
            role_slot_id: str,
            role_skills: Optional[List[str]] = None,
            role_description: Optional[str] = None,
            specialized_role_name: Optional[str] = None,
            role_budget_pct_phase: Optional[float] = None,
            role_time_weeks: Optional[float] = None,
            role_week_hours: Optional[float] = None
        ) -> str:
            """
            Update requirements for a project role slot.

            Args:
                role_slot_id: UUID of the project_phase_roles record
                role_skills: List of required skills (optional)
                role_description: Description of the role (optional)
                specialized_role_name: Specialized name for this role (optional)
                role_budget_pct_phase: Budget percentage for this role (optional)
                role_time_weeks: Estimated time in weeks (optional)
                role_week_hours: Hours per week (optional)

            Returns:
                Confirmation message
            """
            if not supabase:
                return "Error: Supabase not configured"

            # Build update dict (only non-None values)
            update_data = {}
            if role_skills is not None:
                update_data["role_skills"] = role_skills
            if role_description is not None:
                update_data["role_description"] = {"description": role_description}
            if specialized_role_name is not None:
                update_data["specialized_role_name"] = specialized_role_name
            if role_budget_pct_phase is not None:
                update_data["role_budget_pct_phase"] = role_budget_pct_phase
            if role_time_weeks is not None:
                update_data["role_time_weeks"] = role_time_weeks
            if role_week_hours is not None:
                update_data["role_week_hours"] = role_week_hours

            if not update_data:
                return "No updates provided"

            try:
                supabase.table("project_phase_roles").update(update_data).eq("id", role_slot_id).execute()

                updates = ", ".join([f"{k}={v}" for k, v in update_data.items()])
                return f"✅ Updated role requirements: {updates}"

            except Exception as e:
                logger.error(f"Failed to update role: {e}")
                return f"❌ Failed to update role: {str(e)}"

        return update_role_requirements

    def _create_remove_role_slot_tool(self):
        """Tool for removing a role slot."""
        supabase = self.supabase_client

        @tool
        def remove_role_slot(role_slot_id: str) -> str:
            """
            Remove a role slot from a project phase.

            Args:
                role_slot_id: UUID of the project_phase_roles record to remove

            Returns:
                Confirmation message
            """
            if not supabase:
                return "Error: Supabase not configured"

            try:
                supabase.table("project_phase_roles").delete().eq("id", role_slot_id).execute()
                return f"✅ Removed role slot {role_slot_id}"

            except Exception as e:
                logger.error(f"Failed to remove role slot: {e}")
                return f"❌ Failed to remove role slot: {str(e)}"

        return remove_role_slot

    def _create_assign_freelancer_tool(self):
        """Tool for assigning a freelancer to a role slot."""
        supabase = self.supabase_client

        @tool
        def assign_freelancer_to_role(
            role_slot_id: str,
            freelancer_id: str
        ) -> str:
            """
            Assign a freelancer to a specific role slot.

            Args:
                role_slot_id: UUID of the project_phase_roles record
                freelancer_id: UUID of the freelancer to assign

            Returns:
                Confirmation message
            """
            if not supabase:
                return "Error: Supabase not configured"

            try:
                supabase.table("project_phase_roles").update({
                    "freelancer_id": freelancer_id
                }).eq("id", role_slot_id).execute()

                return f"✅ Assigned freelancer {freelancer_id} to role slot {role_slot_id}"

            except Exception as e:
                logger.error(f"Failed to assign freelancer: {e}")
                return f"❌ Failed to assign freelancer: {str(e)}"

        return assign_freelancer_to_role

    def _create_get_phase_info_tool(self):
        """Tool for getting phase information."""
        supabase = self.supabase_client

        @tool
        def get_phase_info(project_id: str, phase_id: Optional[str] = None) -> str:
            """
            Get information about project phases.

            Args:
                project_id: UUID of the project
                phase_id: UUID of specific phase (optional, returns all if not provided)

            Returns:
                Phase information as JSON string
            """
            if not supabase:
                return "Error: Supabase not configured"

            try:
                query = supabase.table("project_phases").select(
                    "id, phase_title, phase_description, tasks, phase_number, "
                    "phase_budget_pct, predicted_time_weeks, predicted_time_hours, "
                    "predicted_time_months, approval_status, created_at, updated_at"
                ).eq("project_id", project_id)

                if phase_id:
                    query = query.eq("id", phase_id)

                result = query.order("phase_number").execute()

                if result.data:
                    return json.dumps(result.data, indent=2, default=str)
                else:
                    return f"No phases found for project {project_id}"

            except Exception as e:
                logger.error(f"Failed to get phase info: {e}")
                return f"❌ Error: {str(e)}"

        return get_phase_info

    def _create_update_phase_tool(self):
        """Tool for updating phase details."""
        supabase = self.supabase_client

        @tool
        def update_phase(
            phase_id: str,
            phase_title: Optional[str] = None,
            phase_description: Optional[str] = None,
            tasks: Optional[List[str]] = None,
            phase_budget_pct: Optional[float] = None
        ) -> str:
            """
            Update details of a project phase.

            Args:
                phase_id: UUID of the phase
                phase_title: New title for the phase (optional)
                phase_description: New description (optional)
                tasks: New list of tasks (optional)
                phase_budget_pct: Budget percentage for this phase (optional)

            Returns:
                Confirmation message
            """
            if not supabase:
                return "Error: Supabase not configured"

            # Build update dict
            update_data = {}
            if phase_title is not None:
                update_data["phase_title"] = phase_title
            if phase_description is not None:
                update_data["phase_description"] = phase_description
            if tasks is not None:
                update_data["tasks"] = tasks
            if phase_budget_pct is not None:
                update_data["phase_budget_pct"] = phase_budget_pct

            if not update_data:
                return "No updates provided"

            try:
                supabase.table("project_phases").update(update_data).eq("id", phase_id).execute()

                updates = ", ".join([f"{k}={v}" for k, v in update_data.items()])
                return f"✅ Updated phase: {updates}"

            except Exception as e:
                logger.error(f"Failed to update phase: {e}")
                return f"❌ Failed to update phase: {str(e)}"

        return update_phase

    def _create_get_all_roles_tool(self):
        """Tool for getting all available roles from the roles table."""
        supabase = self.supabase_client

        @tool
        def get_all_roles() -> str:
            """
            Get all available roles from the roles table.

            Returns:
                List of all roles with their IDs, names, and industries as JSON string
            """
            if not supabase:
                return "Error: Supabase not configured"

            try:
                result = supabase.table("roles").select("id, name, Industry").execute()

                if result.data:
                    return json.dumps(result.data, indent=2, default=str)
                else:
                    return "No roles found in the roles table"

            except Exception as e:
                logger.error(f"Failed to get roles: {e}")
                return f"❌ Error: {str(e)}"

        return get_all_roles

    def _create_get_role_info_tool(self):
        """Tool for getting role information."""
        supabase = self.supabase_client

        @tool
        def get_role_info(project_id: str, phase_id: Optional[str] = None) -> str:
            """
            Get information about project roles.

            Args:
                project_id: UUID of the project
                phase_id: UUID of specific phase (optional, returns all if not provided)

            Returns:
                Role information as JSON string
            """
            if not supabase:
                return "Error: Supabase not configured"

            try:
                query = supabase.table("project_phase_roles").select(
                    "id, phase_id, role_id, role_skills, role_description, "
                    "role_budget_pct_phase, freelancer_id, roles(name)"
                ).eq("project_id", project_id)

                if phase_id:
                    query = query.eq("phase_id", phase_id)

                result = query.execute()

                if result.data:
                    # Flatten role name
                    roles = []
                    for r in result.data:
                        role_data = r.copy()
                        if "roles" in role_data and role_data["roles"]:
                            role_data["role_name"] = role_data["roles"].get("name")
                            del role_data["roles"]
                        roles.append(role_data)

                    return json.dumps(roles, indent=2, default=str)
                else:
                    return f"No roles found for project {project_id}"

            except Exception as e:
                logger.error(f"Failed to get role info: {e}")
                return f"❌ Error: {str(e)}"

        return get_role_info

    def _create_get_project_data_tool(self):
        """Tool for getting comprehensive project data."""
        supabase = self.supabase_client

        @tool
        def get_project_data(
            project_id: str,
            client_id: Optional[str] = None,
            phase_id: Optional[str] = None,
            role_slot_id: Optional[str] = None
        ) -> str:
            """
            Get comprehensive project data including phases, roles, and project report.

            This retrieves:
            - final_project_report from project_decomposition
            - All project_phases (or specific phase if phase_id provided)
            - All project_phase_roles (filtered by phase_id or role_slot_id if provided)

            Args:
                project_id: UUID of the project (required)
                client_id: UUID of the client (optional, for validation)
                phase_id: UUID of specific phase to filter (optional)
                role_slot_id: UUID of specific role slot (optional)

            Returns:
                Comprehensive project data as JSON
            """
            if not supabase:
                return "Error: Supabase not configured"

            try:
                result = {
                    "project_id": project_id,
                    "final_project_report": None,
                    "phases": [],
                    "role_slots": []
                }

                # 1. Get final_project_report from project_decomposition
                project_decomp = supabase.table("project_decomposition").select(
                    "final_project_report, project_name, project_topic, project_budget, "
                    "predicted_cost, status, select_freelancer_status"
                ).eq("project_id", project_id)

                if client_id:
                    project_decomp = project_decomp.eq("client_id", client_id)

                project_result = project_decomp.execute()

                if project_result.data:
                    result["final_project_report"] = project_result.data[0].get("final_project_report")
                    result["project_name"] = project_result.data[0].get("project_name")
                    result["project_topic"] = project_result.data[0].get("project_topic")
                    result["project_budget"] = project_result.data[0].get("project_budget")
                    result["predicted_cost"] = project_result.data[0].get("predicted_cost")
                    result["status"] = project_result.data[0].get("status")
                    result["select_freelancer_status"] = project_result.data[0].get("select_freelancer_status")

                # 2. Get project_phases
                phases_query = supabase.table("project_phases").select(
                    "id, phase_title, phase_description, tasks, phase_number, "
                    "phase_budget_pct, predicted_time_weeks, predicted_time_hours, "
                    "approval_status"
                ).eq("project_id", project_id)

                if phase_id:
                    phases_query = phases_query.eq("id", phase_id)

                phases_result = phases_query.order("phase_number").execute()

                if phases_result.data:
                    result["phases"] = phases_result.data

                # 3. Get project_phase_roles
                roles_query = supabase.table("project_phase_roles").select(
                    "id, phase_id, role_id, role_slot, freelancer_id, "
                    "role_skills, role_description, specialized_role_name, "
                    "role_budget_pct_phase, role_time_weeks, role_week_hours, "
                    "supplied_by, pre_selection_freelancer_ids, recommended_freelancers, "
                    "roles(name)"
                ).eq("project_id", project_id)

                if phase_id:
                    roles_query = roles_query.eq("phase_id", phase_id)

                if role_slot_id:
                    roles_query = roles_query.eq("id", role_slot_id)

                roles_result = roles_query.execute()

                if roles_result.data:
                    # Flatten role name
                    role_slots = []
                    for r in roles_result.data:
                        role_data = r.copy()
                        if "roles" in role_data and role_data["roles"]:
                            role_data["role_name"] = role_data["roles"].get("name")
                            del role_data["roles"]
                        role_slots.append(role_data)

                    result["role_slots"] = role_slots

                return json.dumps(result, indent=2, default=str)

            except Exception as e:
                logger.error(f"Failed to get project data: {e}")
                return f"❌ Error: {str(e)}"

        return get_project_data

    def _create_match_freelancers_tool(self):
        """Tool for matching freelancers using external service."""

        @tool
        def match_freelancers(
            project_id: str,
            phase_ids: Optional[List[str]] = None,
            role_ids: Optional[List[str]] = None,
            max_results: int = 20
        ) -> str:
            """
            Find and recommend freelancers for project roles using AI matching.

            Args:
                project_id: UUID of the project
                phase_ids: List of phase IDs to re-optimize (optional)
                role_ids: List of role IDs to re-optimize (optional)
                max_results: Maximum number of results per role

            Returns:
                Matching results
            """
            service_url = os.getenv("SELECT_FREELANCER_URL", "")
            if not service_url:
                return "❌ Error: SELECT_FREELANCER_URL not configured"

            try:
                if phase_ids or role_ids:
                    # Re-optimize specific targets
                    endpoint = f"{service_url.rstrip('/')}/reoptimize-targets"
                    payload = {
                        "project_id": project_id,
                        "phase_ids": phase_ids or [],
                        "phase_role_ids": role_ids or [],
                        "max_results": max_results
                    }
                    action = "Re-optimization"
                else:
                    # Match all freelancers
                    endpoint = f"{service_url.rstrip('/')}/match-all-freelancers"
                    payload = {
                        "project_id": project_id,
                        "max_results": max_results
                    }
                    action = "Match all"

                response = requests.post(endpoint, json=payload, timeout=30)

                if response.status_code == 200:
                    return f"✅ {action} successful: {json.dumps(response.json(), indent=2)}"
                else:
                    return f"❌ {action} failed: {response.status_code} - {response.text}"

            except requests.Timeout:
                return "❌ Matching service timeout"
            except Exception as e:
                logger.error(f"Failed to match freelancers: {e}")
                return f"❌ Error: {str(e)}"

        return match_freelancers

    # --- Stream Method (for A2A protocol) ---

    async def stream(
        self,
        query: str,
        context_id: str = "default",
        project_id: str = None,
        client_id: str = None
    ):
        """
        Stream responses from the ReAct agent.

        Args:
            query: User's message/query
            context_id: Conversation thread ID
            project_id: Current project ID context
            client_id: Client ID for authentication

        Yields:
            Dict with streaming response data:
                - is_task_complete: bool
                - require_user_input: bool
                - content: str
        """
        config = {"configurable": {"thread_id": context_id}}

        try:
            # Build message list with context
            messages = [HumanMessage(content=query)]

            # Add project context to messages if provided
            context_info = []
            if project_id:
                context_info.append(f"project_id: {project_id}")
            if client_id:
                context_info.append(f"client_id: {client_id}")

            if context_info:
                messages.insert(0, SystemMessage(content=f"[Context: {', '.join(context_info)}]"))

            # Stream from ReAct agent
            final_response = ""

            async for event in self.graph.astream(
                {"messages": messages},
                config=config,
                stream_mode="messages"
            ):
                if isinstance(event, tuple):
                    message, metadata = event

                    # Only stream AI messages (not tool calls)
                    if hasattr(message, 'content') and message.content:
                        if not hasattr(message, 'tool_calls') or not message.tool_calls:
                            content = message.content

                            # Handle dict/list content (Gemini format)
                            if isinstance(content, list):
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and 'text' in item:
                                        text_parts.append(item['text'])
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                content = '\n'.join(text_parts)
                            elif isinstance(content, dict):
                                content = content.get('text', str(content))

                            final_response = content
                            yield {
                                'is_task_complete': False,
                                'require_user_input': False,
                                'content': content
                            }

            # Final completion signal (don't repeat content - already streamed)
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': ""  # Empty - content already streamed above
            }

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
        project_id: str = None,
        client_id: str = None
    ) -> Dict[str, Any]:
        """
        Synchronous invocation of the agent.

        Args:
            message: User message
            project_id: Current project ID context
            client_id: Client ID for authentication

        Returns:
            Dict with response
        """
        # Handle JSON payload parsing
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    message = parsed.get("message", message)
                    project_id = parsed.get("project_id", project_id)
                    client_id = parsed.get("client_id", client_id)
            except json.JSONDecodeError:
                pass

        messages = [HumanMessage(content=message)]

        # Add context
        if project_id or client_id:
            context_info = []
            if project_id:
                context_info.append(f"project_id: {project_id}")
            if client_id:
                context_info.append(f"client_id: {client_id}")
            messages.insert(0, SystemMessage(content=f"[Context: {', '.join(context_info)}]"))

        result = self.graph.invoke({"messages": messages})

        # Extract final AI message
        final_message = ""
        if result.get("messages"):
            final_message = result["messages"][-1].content if result["messages"] else ""

        return {
            "response": final_message
        }


# --- Testing ---
async def test():
    """Test the ReAct agent."""
    agent = SelectFreelancerReActAgent()

    project_id = "058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc"
    client_id = "9a76be62-0d44-4a34-913d-08dcac008de5"

    test_queries = [
        "What roles are available in the project?",
        "The Security Engineer needs to have experience in AI Agents",
        "Find freelancers for all roles in this project",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}\n")

        async for item in agent.stream(query, project_id=project_id, client_id=client_id):
            if item.get('content'):
                print(item['content'], end='', flush=True)
            if item.get('is_task_complete'):
                print("\n[Task Complete]")


if __name__ == "__main__":
    import asyncio

    # For A2A server deployment
    port = int(os.getenv("PORT", 8012))
    print(f"Starting Select Freelancer ReAct Agent on port {port}...")

    from agent_executor import run_server
    run_server(SelectFreelancerReActAgent(), port=port)

    # For testing
    # asyncio.run(test())
