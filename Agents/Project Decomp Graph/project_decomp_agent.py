from typing import Annotated, List, Literal, Optional, TypedDict, Union, Dict, Any

import os
import time
import logging

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- A2A Integration ---
from a2a.types import AgentCard, AgentSkill
import json

from supabase import create_client, Client

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Define the state for this subgraph
class ProjectDecompState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] #Conversation history
    project_id: Optional[str] #Project_id value
    client_id: Optional[str] #Client_id value
    job_id: Optional[str] #Job ID from FlashBrain orchestrator for async tracking
    exist: Optional[bool] #Project_id exists
    extracted_info: Optional[str] #History to send to A2A
    orchestrator_response: Optional[str] #Response from the A2A 
    job_status: Optional[str] #Status of the async job 

class ProjectDecompAgent:

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ProjectDecompState)

        workflow.add_node("extract_information", self.extract_information)
        workflow.add_node("upload_to_supabase", self.upload_to_supabase)
        workflow.add_node("call_orchestrator", self.call_orchestrator)

        workflow.add_edge(START, "extract_information")
        workflow.add_edge("extract_information", "upload_to_supabase")
        workflow.add_edge("upload_to_supabase", "call_orchestrator")
        workflow.add_edge("call_orchestrator", END)

        return workflow.compile()

    # --- Node 1: Extract Information using LLM ---
    def extract_information(self, state: ProjectDecompState) -> Dict[str, Any]:
        """
        Uses an LLM to extract important information from the conversation
        before sending it to the decomposition orchestrator.
        """
        # Get the full conversation history
        messages = state["messages"]
        conversation_history = ""
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            if isinstance(msg, SystemMessage):
                role = "System"
            conversation_history += f"{role}: {msg.content}\n"
        
        project_id = state.get("project_id", "Unknown")
        exist = state.get("exist", False)
        
        # Initialize the LLM (using a fast model for extraction)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        
        prompt = f"""
        Analyze the following conversation history and extract the key requirements, 
        constraints, and goals for the project (Project ID: {project_id}). 
        Format the output as a clear narrative extracting all details and descriptions of the project. Be sure to include all details.  
        
        Focus on information relevant to the project. Ignore unrelated chit-chat. Do not include project_id in the output only a detailed description. 

        Do not add additional information or comments to the output. Include the details and conversation pertaining to the project, nothing else and nothing more. 
        
        If the project_id already exists (Exist: {exist}), focus on extracting the updates or changes requested.
        
        Conversation History:
        {conversation_history}
        """
        
        try:
            response = llm.invoke(prompt)
            return {"extracted_info": {"summary": response.content, "original_request": conversation_history}}
        except Exception as e:
            print(f"Extraction failed: {e}")
            return {"extracted_info": {"summary": "Extraction failed.", "original_request": conversation_history}}

    # --- Node 1.5: Upload to Supabase ---
    def upload_to_supabase(self, state: ProjectDecompState) -> Dict[str, Any]:
        """
        Uploads the extracted summary and job_id to Supabase.
        This ensures the webhook can send job_id back to the orchestrator when the project completes.
        """
        extracted_info = state.get("extracted_info", {})
        summary = extracted_info.get("summary", "")
        job_id = state.get("job_id")
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            return {"messages": [SystemMessage(content="ERROR: Supabase credentials not found. Skipping upload.")]}

        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            
            # Update Project_Decomposition table with summary AND job_id
            data = {
                "input": summary,
            }
            
            # Add job_id if provided (for async orchestration tracking)
            if job_id:
                data["job_id"] = job_id
                logger.info(f"Saving job_id {job_id} to project_decomposition table for project {state['project_id']}")
            
            response = supabase.table("project_decomposition").update(data).eq("project_id", state["project_id"]).execute()

            return {"messages": [SystemMessage(content="Successfully uploaded to Supabase.")]}

        except Exception as e:
            return {"messages": [SystemMessage(content=f"ERROR: {e} Failed to upload to Supabase. Skipping upload. {state['project_id']}")]}

    # --- Node 2: Call Project Decomposition Orchestrator via HTTP ---
    def call_orchestrator(self, state: ProjectDecompState) -> Dict[str, Any]:
        """
        Sends the project_id to the Project Decomposition Orchestrator via HTTP POST.
        """
        project_id = state.get("project_id")
        
        if not project_id:
            return {
                "messages": [AIMessage(content="Error: Project ID is missing in state.")],
                "orchestrator_response": "Project ID missing"
            }
        
        # Get URL and API Key
        orchestrator_url = os.getenv("PROJECT_DECOMP_ORCHESTRATOR_URL", "")
        
        if not orchestrator_url:
            return {
                "messages": [AIMessage(content="Configuration Error: Project Decomposition Orchestrator URL is missing.")],
                "orchestrator_response": "URL not configured"
            }

        if not orchestrator_url.endswith("/project"):
            target_url = f"{orchestrator_url.rstrip('/')}/project"
        else:
            target_url = orchestrator_url

        import requests
        
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": state.get("client_id")
            }
            payload = {"project_id": project_id}
            
            time1 = time.time()
            response = requests.post(target_url, json=payload, headers=headers)
            time2 = time.time()
            print(f"Time taken: {time2 - time1}")
            
            if response.status_code == 202:
                return {
                    "messages": [AIMessage(content=f"Project decomposition started successfully (Status 202).")],
                    "orchestrator_response": "Started"
                }
            else:
                return {
                    "messages": [AIMessage(content=f"Failed to start decomposition. Status: {response.status_code}, Response: {response.text}")],
                    "orchestrator_response": f"Failed: {response.status_code}"
                }
            
        except Exception as e:
            print(f"Project Decomposition Orchestrator call failed: {e}")
            return {
                "messages": [AIMessage(content=f"Failed to contact Project Decomposition Orchestrator: {e}")],
                "orchestrator_response": f"Error: {e}"
            }

    async def stream(self, query: str, context_id: str = "default", project_id: str = None, client_id: str = None, job_id: str = None, exist: bool = False):
        """
        Streams responses from the Project Decomposition Graph for A2A protocol.

        Args:
            query: User's project description
            context_id: Conversation context ID
            project_id: Project ID (if existing project)
            client_id: Client ID for authentication
            job_id: Job ID from FlashBrain orchestrator (for async tracking)
            exist: Whether the project already exists

        Yields:
            Dict with keys:
                - is_task_complete: bool
                - require_user_input: bool
                - content: str
        """
        messages = [HumanMessage(content=query)]
        inputs = {
            "messages": messages,
            "project_id": project_id,
            "client_id": client_id,
            "job_id": job_id,
            "exist": exist
        }
        
        config = {"configurable": {"thread_id": context_id}}

        try:
            # Yield initial status
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': 'Starting project decomposition...'
            }

            # Execute the graph with streaming
            final_state = {}
            for event in self.graph.stream(inputs, stream_mode = 'messages', subgraphs= True):
                message = event[1]
                #print(event[1][1]['langgraph_node'])
                
                if isinstance(message[0], SystemMessage):
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': "Writing to Database..."
                    }
                elif isinstance(message[0], AIMessage) and message[1]['langgraph_node'] == 'call_orchestrator':
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': "Creating Project Plan... This may take a few minutes."
                    }
                
                        
                
            # Final result processing
            job_status = final_state.get("job_status", "unknown")
            orchestrator_response = final_state.get("orchestrator_response", "")

            # Check if we need user input based on status
            '''
            if job_status == "failed":
                yield {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': f"Project decomposition failed: {orchestrator_response}"
                }
            elif job_status == "completed":
                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': f"Project decomposition completed successfully. Status: {orchestrator_response}"
                }
            else:
                yield {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': f"Project decomposition job status: {job_status}. Response: {orchestrator_response}"
                }
            '''

        except Exception as e:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f"An error occurred during project decomposition: {str(e)}"
            }

    def invoke(self, query: str, project_id: str, client_id: str = None, job_id: str = None, exist: bool = False) -> Dict[str, Any]:
        """
        Invokes the Project Decomposition Graph.
        """
        # Handle JSON payload parsing if input is a string (A2A compat)
        if isinstance(query, str):
            try:
                parsed = json.loads(query)
                if isinstance(parsed, dict):
                    query = parsed.get("query", query)
                    project_id = parsed.get("project_id", project_id)
                    client_id = parsed.get("client_id", client_id)
                    job_id = parsed.get("job_id", job_id)
                    exist = parsed.get("exist", exist)
            except:
                pass

        messages = [HumanMessage(content=query)]
        state = {
            "messages": messages,
            "project_id": project_id,
            "client_id": client_id,
            "job_id": job_id,
            "exist": exist
        }
        result = self.graph.invoke(state)
        return {
            "job_status": result.get("job_status"),
            "orchestrator_response": result.get("orchestrator_response")
        }


async def test():
    agent = ProjectDecompAgent()

    text = """The project, named SkillLink - Micro Mentorship Platform, is a two-sided marketplace platform designed to connect users with experts for short, on-demand mentoring sessions.

**Project Description:**
SkillLink aims to be a real-time mentorship platform facilitating 15-30 minute micro-mentoring sessions. These sessions can be conducted via video, voice, or chat, providing users with quick guidance for skill development, career advice, and tackling specific challenges.

**Business Objectives:**
The platform intends to generate revenue through several streams:
*   Session commissions.
*   A premium subscription model.
*   Monetization of featured mentor listings.

**Target Audience:**
The platform is intended for college students, early-career professionals, self-learners, and entrepreneurs.

**Key Requirements:**
*   A real-time mentor matching system.
*   Communication capabilities supporting video, voice, and chat.
*   A time-based pricing system for sessions.
*   Functionality for session recording.
*   A system for mentor reviews and ratings.
*   A feature to follow or bookmark mentors.
*   An integrated calendar and reminder system.
*   User profiles and authentication.
*   Integration with a payment processing system.

**Technical Architecture:**
The project will involve a mobile application (iOS and Android) and a web application. The complexity is rated as high.

**Project Specifications:**
*   **Timeline:** 2 months
*   **Budget:** $15,000
*   **Priority:** High

**Deliverables:**
*   A mobile application for both iOS and Android.
*   A web platform.

**Additional Details:**
Success metrics, risk factors, strategic recommendations, scalability planning, and a maintenance plan are yet to be defined or developed."""

    print("Calling agent...")
    async for event in agent.stream(project_id = "058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc", query = text, client_id = "9a76be62-0d44-4a34-913d-08dcac008de5", exist = False):
        print(event)

if __name__ == "__main__":
    import asyncio
    #port = int(os.getenv("PORT", 8011))
    #print(f"Starting Project Decomposition Agent on port {port}...")
    #run_server(ProjectDecompAgent(), port=port)
    asyncio.run(test())
