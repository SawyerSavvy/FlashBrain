from typing import Annotated, List, Literal, Optional, TypedDict, Union, Dict, Any
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

try:
    from supabase import create_client, Client
except ImportError:
    print("Warning: supabase package not found. Please install it with `pip install supabase`.")
    create_client = None
    Client = None

# --- State Definition ---
class SelectFreelancerState(TypedDict):
    messages: List[BaseMessage]
    project_id: str
    extracted_updates: Optional[Dict[str, Any]]
    api_response: Optional[str]
    existing_phases: List[Dict[str, Any]]
    existing_roles: List[Dict[str, Any]]
    reasoning: Optional[str]
    error: Optional[bool]

# --- Pydantic Models for Structured Output ---
class PhaseUpdate(BaseModel):
    phase_id: Optional[str] = Field(None, description="UUID of the phase if known, else None")
    phase_title: Optional[str] = Field(None, description="Title of the phase")
    description: Optional[str] = Field(None, description="Description of the phase")
    tasks: Optional[List[str]] = Field(None, description="List of tasks in the phase")

class RoleUpdate(BaseModel):
    role_id: Optional[str] = Field(None, description="UUID of the role if known, else None")
    phase_id: Optional[str] = Field(None, description="UUID of the phase this role belongs to")
    role_name: Optional[str] = Field(None, description="Name of the role")
    description: Optional[str] = Field(None, description="Detailed description of the role")
    role_skills: Optional[List[str]] = Field(None, description="List of required skills")
    budget_pct: Optional[float] = Field(None, description="Budget percentage for this role")

class RequirementsOutput(BaseModel):
    phases: Optional[List[PhaseUpdate]] = Field(default_factory=list, description="List of phase updates")
    roles: Optional[List[RoleUpdate]] = Field(default_factory=list, description="List of role updates")
    reasoning: str = Field(..., description="Explanation of the changes made, citing specific user requests and existing state context.")

# --- Node 0: Fetch Project Data ---
def fetch_project_data(state: SelectFreelancerState) -> Dict[str, Any]:
    """
    Fetches existing project phases and roles from Supabase.
    """
    project_id = state.get("project_id")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not project_id or not supabase_url or not supabase_key or not create_client:
        print("Missing project_id or Supabase credentials.")
        return {"messages": AIMessage(content="Supabase not configured or unavailable."), "existing_phases": [], "existing_roles": []}
        
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Fetch Phases
        phases_res = supabase.table("project_phases").select("*").eq("project_id", project_id).execute()
        existing_phases = phases_res.data if phases_res.data else []
        
        # Fetch Roles
        # Join with roles table to get the role name
        roles_res = supabase.table("project_phase_roles").select("*, roles(name)").eq("project_id", project_id).execute()
        
        existing_roles = []
        if roles_res.data:
            for r in roles_res.data:
                # Flatten the role name
                role_data = r.copy()
                if "roles" in role_data and role_data["roles"]:
                    role_data["role_name"] = role_data["roles"].get("name")
                    del role_data["roles"] # Clean up nested object
                existing_roles.append(role_data)
        
        return {"existing_phases": existing_phases, "existing_roles": existing_roles}
        
    except Exception as e:
        print(f"Failed to fetch project data: {e}")
        return {"messages": AIMessage(content="Failed to fetch project data."), "error": True, "existing_phases": [], "existing_roles": []}

# --- Node 1: Extract Requirements ---
def extract_requirements(state: SelectFreelancerState) -> Dict[str, Any]:
    """
    Extracts freelancer role requirements and phase details from the conversation.
    """
    messages = state["messages"]
    conversation_history = ""
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        if isinstance(msg, SystemMessage):
            role = "System"
        conversation_history += f"{role}: {msg.content}\n"
    
    project_id = state.get("project_id", "Unknown")
    existing_phases = state.get("existing_phases", [])
    existing_roles = state.get("existing_roles", [])
    
    # Format existing state for the prompt
    existing_state_str = json.dumps({
        "phases": existing_phases,
        "roles": existing_roles
    }, indent=2, default=str)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(RequirementsOutput)
    
    prompt = f"""
    Analyze the following conversation history and extract updates for project phases and freelancer roles (Project ID: {project_id}).
    
    Current Project State:
    {existing_state_str}
    
    Compare the conversation request with the Current Project State.
    Identify what needs to be ADDED or UPDATED. If there are no updates, return empty phases and roles lists. 

    Only return the fields that are being updated. Do not return fields that are not being updated. For example, if the descritipion is not being changed, it should not be returned. 

    Provide a clear reasoning for your decisions.
    
    Conversation History:
    {conversation_history}
    """
    
    try:
        response = structured_llm.invoke(prompt)
        # response is an instance of RequirementsOutput
        extracted_updates = response.model_dump()
        reasoning = extracted_updates.pop("reasoning", "")
        
        return {"extracted_updates": extracted_updates, "reasoning": reasoning}
    except Exception as e:
        print(f"Extraction failed: {e}")
        return {"extracted_updates": {"phases": [], "roles": []}, "reasoning": f"Error: {e}"}

# --- Node 2: Upload to Supabase ---
def upload_to_supabase(state: SelectFreelancerState) -> Dict[str, Any]:
    """
    Updates project_phases and project_phase_roles in Supabase.
    """
    extracted_updates = state.get("extracted_updates", {})
    phases = extracted_updates.get("phases", [])
    roles = extracted_updates.get("roles", [])
    project_id = state.get("project_id")
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key or not create_client:
        print("Supabase not configured or unavailable.")
        return {}
        
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Update Phases
        for phase in phases:
            phase_id = phase.get("phase_id")
            phase_title = phase.get("phase_title")
            
            # If we don't have phase_id, try to find it by title and project_id
            if not phase_id and phase_title and project_id:
                res = supabase.table("project_phases").select("id").eq("project_id", project_id).eq("phase_title", phase_title).execute()
                if res.data:
                    phase_id = res.data[0]["id"]
            
            if phase_id:
                update_data = {}
                if phase.get("description"):
                    update_data["phase_description"] = phase.get("description")
                if phase.get("tasks"):
                    # Convert list of strings to jsonb structure if needed, or just store as is if schema allows
                    # Schema says tasks is jsonb. Assuming simple list is okay or dict.
                    update_data["tasks"] = phase.get("tasks")
                
                if update_data:
                    supabase.table("project_phases").update(update_data).eq("id", phase_id).execute()
                    print(f"Updated phase {phase_id}")

        # Update Roles
        for role in roles:
            role_id = role.get("role_id")
            role_name = role.get("role_name")
            phase_id = role.get("phase_id") # Might be needed to find role
            
            # If we don't have role_id, try to find it. 
            # This is harder without exact mapping, but let's assume we might find it by specialized_role_name if it exists
            # or we skip if we can't identify the role.
            # For now, we proceed only if we have role_id or can find it.
            
            if not role_id and role_name and project_id:
                 # Try to find by specialized_role_name? Or maybe we can't easily.
                 # Let's skip lookup for now to avoid complexity and assume ID is provided or we skip.
                 pass

            if role_id:
                update_data = {
                    k:v for k,v in role.items() if v is not None
                }

                if update_data:
                    supabase.table("project_phase_roles").update(update_data).eq("id", role_id).execute()
                    print(f"Updated role {role_id}")
                    
        return {}
        
    except Exception as e:
        print(f"Supabase update failed: {e}")
        return {}

# --- Node 3: Call Freelancer Service ---
def call_freelancer_service(state: SelectFreelancerState) -> Dict[str, Any]:
    """
    Calls the Freelancer Service to match or re-optimize.
    """
    extracted_updates = state.get("extracted_updates", {})
    roles = extracted_updates.get("roles", [])
    phases = extracted_updates.get("phases", [])
    project_id = state.get("project_id")
    
    service_url = os.getenv("SELECT_FREELANCER_URL", "")
    if not service_url:
        return {"api_response": "Error: SELECT_FREELANCER_URL not configured"}
        
    # Determine which endpoint to call
    # If specific roles/phases were updated, use /reoptimize-targets
    # Otherwise (or if explicitly requested), use /match-all-freelancers
    
    # Collect IDs for re-optimization
    phase_ids = [p.get("phase_id") for p in phases if p.get("phase_id")]
    role_ids = [r.get("role_id") for r in roles if r.get("role_id")]
    
    try:
        if phase_ids or role_ids:
            # Call /reoptimize-targets
            endpoint = f"{service_url.rstrip('/')}/reoptimize-targets"
            payload = {
                "project_id": project_id,
                "phase_ids": phase_ids,
                "phase_role_ids": role_ids,
                "max_results": 20
            }
            response = requests.post(endpoint, json=payload)
            action = "Re-optimization"
        else:
            # Call /match-all-freelancers
            endpoint = f"{service_url.rstrip('/')}/match-all-freelancers"
            payload = {
                "project_id": project_id,
                "max_results": 10
            }
            response = requests.post(endpoint, json=payload)
            action = "Match All"
            
        if response.status_code == 200:
            return {"api_response": f"{action} successful: {response.json()}"}
        else:
            return {"api_response": f"{action} failed: {response.status_code} - {response.text}"}
            
    except Exception as e:
        return {"api_response": f"Service call failed: {e}"}

# --- Build Graph ---
workflow = StateGraph(SelectFreelancerState)

workflow.add_node("fetch_project_data", fetch_project_data)
workflow.add_node("extract_requirements", extract_requirements)
workflow.add_node("upload_to_supabase", upload_to_supabase)
workflow.add_node("call_freelancer_service", call_freelancer_service)

workflow.add_edge(START, "fetch_project_data")
workflow.add_edge("fetch_project_data", "extract_requirements")
workflow.add_edge("extract_requirements", "upload_to_supabase")
workflow.add_edge("upload_to_supabase", "call_freelancer_service")
workflow.add_edge("call_freelancer_service", END)

select_freelancer_graph = workflow.compile()


# --- A2A Integration ---
from python_a2a import run_server, AgentCard, AgentSkill

select_freelancer_skill = AgentSkill(
    id='select_freelancer',
    name="select_freelancer",
    description="Updates project phase and roles in Supabase based on user's input",
    tags=["freelancer", "selection"],
)

AgentCard(
    name="Select Freelancer Agent",
    description="An agent that selects freelancers for a project.",
    url="http://localhost:8012",  # Required parameter
    version="0.1.0",
    capabilities={"streaming": False},
    skills=[select_freelancer_skill],
)

class SelectFreelancerAgent:

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    async def stream(self, query: str, context_id: str = "default", project_id: str = None):
        """
        Streams responses from the Select Freelancer Graph for A2A protocol.

        Yields:
            Dict with keys:
                - is_task_complete: bool
                - require_user_input: bool
                - content: str
        """
        messages = [HumanMessage(content=query)]
        state = {
            "messages": messages,
            "project_id": project_id
        }

        try:
            # Yield initial status
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': 'Fetching project data...'
            }

            # Execute the graph
            result = select_freelancer_graph.invoke(state)

            extracted_updates = result.get("extracted_updates", {})
            api_response = result.get("api_response", "")
            error = result.get("error", False)

            # Build response message
            if error:
                yield {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': f"Error occurred while processing: {api_response}"
                }
            elif api_response:
                phases = extracted_updates.get("phases", [])
                roles = extracted_updates.get("roles", [])
                summary = f"Updated {len(phases)} phase(s) and {len(roles)} role(s). {api_response}"

                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': summary
                }
            else:
                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': "Freelancer selection completed successfully."
                }

        except Exception as e:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"An error occurred during freelancer selection: {str(e)}"
            }

    def invoke(self, message: str, project_id: str) -> Dict[str, Any]:
        """
        Invokes the Select Freelancer Graph.
        """
        # Handle JSON payload parsing if input is a string (A2A compat)
        if isinstance(message, str):
            try:
                import json
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    message = parsed.get("message", message)
                    project_id = parsed.get("project_id", project_id)
            except:
                pass

        messages = [HumanMessage(content=message)]
        state = {
            "messages": messages,
            "project_id": project_id
        }
        result = select_freelancer_graph.invoke(state)
        return {
            "extracted_updates": result.get("extracted_updates"),
            "api_response": result.get("api_response")
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8012))
    print(f"Starting Select Freelancer Agent on port {port}...")
    run_server(SelectFreelancerAgent(), port=port)

