"""
FlashBrain Orchestrator Server Entry Point

This module starts the FlashBrain Orchestrator A2A server.
Uses the simplified ReAct agent architecture with dynamic tool calling.
"""

import logging
import os
import sys

import click
import httpx
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage

from FlashBrain_ReAct import FlashBrainReActAgent
from agent_executor import FlashBrainAgentExecutor

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8010)
def main(host, port):
    """Starts the FlashBrain Orchestrator server."""
    try:
        # Initialize the FlashBrain ReAct Agent
        logger.info("Initializing FlashBrain ReAct Agent...")
        flashbrain_agent = FlashBrainReActAgent()
        agent_executor = FlashBrainAgentExecutor(agent=flashbrain_agent)
        supported_content_types = FlashBrainReActAgent.SUPPORTED_CONTENT_TYPES
        logger.info("FlashBrain ReAct Agent initialized successfully")
        
        capabilities = AgentCapabilities(streaming=True, push_notifications=True)

        flashbrain_skill = AgentSkill(
            id='flashbrain_orchestrator',
            name='FlashBrain Orchestrator',
            description='Orchestrates the entire FlashBrain system, routing requests to specialized agents.',
            tags=['orchestrator', 'brain', 'routing'],
            examples=[
                'Help me plan my project',
                'I need to select freelancers for my project',
                'What are the costs for this project?',
                'What do I need to do today?',
                'Hello, how are you today?'
            ],
        )

        # Use ORCHESTRATOR_PUBLIC_URL env var for Cloud Run, fallback to host:port for local
        public_url = os.getenv('ORCHESTRATOR_PUBLIC_URL', f'http://{host}:{port}')

        agent_card = AgentCard(
            name='FlashBrain Orchestrator',
            description='Intelligent AI orchestrator using ReAct pattern with dynamic tool calling for project management, team selection, and financial operations.',
            url=public_url,
            version='2.0.0',
            default_input_modes=supported_content_types,
            default_output_modes=supported_content_types,
            capabilities=capabilities,
            skills=[flashbrain_skill],
        )

        # Create request handler
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        
        class ProjectCompletePayload(BaseModel):
            """Payload for project completion callback."""
            project_id: str
            job_id: str
            status: str
            result: Optional[dict] = None
            error: Optional[str] = None

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        # Define callback endpoint handler for async job completion
        async def handle_project_complete(request):
            """
            Callback endpoint for long-running project decomposition jobs.
            Called by sub-agents (e.g., Project Decomp agent) when async jobs finish.
            """
            try:
                # Parse JSON payload
                payload = await request.json()
                callback_data = ProjectCompletePayload(**payload)
                logger.info(f"Received callback for project {callback_data.project_id}, job {callback_data.job_id}, status {callback_data.status}")
                
                # Update job status in database
                if flashbrain_agent.supabase_client:
                    try:
                        flashbrain_agent.supabase_client.table("brain_jobs").update({
                            "status": callback_data.status,
                            "result": callback_data.result,
                            "error": callback_data.error,
                            "updated_at": "now()"
                        }).eq("id", callback_data.job_id).execute()
                        logger.info(f"Updated job {callback_data.job_id} status to {callback_data.status}")
                    except Exception as e:
                        logger.error(f"Failed to update job status: {e}")
                
                # Acknowledge the callback
                from starlette.responses import JSONResponse
                return JSONResponse({
                    "status": "acknowledged",
                    "job_id": callback_data.job_id,
                    "message": "Callback received and processed"
                })
            
            except Exception as e:
                logger.error(f"Callback processing failed: {e}")
                from starlette.responses import JSONResponse
                return JSONResponse({"error": str(e)}, status_code=500)
        
        # Build the Starlette app and add custom route
        starlette_app = server.build()
        from starlette.routing import Route
        starlette_app.routes.append(
            Route("/callbacks/project-complete", handle_project_complete, methods=["POST"])
        )

        logger.info(f'Starting FlashBrain Orchestrator on {host}:{port}')
        uvicorn.run(starlette_app, host=host, port=port)

    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
