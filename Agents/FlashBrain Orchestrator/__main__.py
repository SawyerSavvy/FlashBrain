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
from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

from FlashBrain import FlashBrainReActAgent
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

        class ChatStreamRequest(BaseModel):
            """Request payload for chat streaming endpoint."""
            message: str
            conversation_id: str = "default"
            metadata: Optional[Dict[str, Any]] = None

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        # Define chat streaming endpoint for direct frontend access
        async def handle_chat_stream(request):
            """
            Direct streaming endpoint for frontend clients.
            Bypasses A2A protocol for real-time chatbot interaction.
            """
            import json
            from starlette.responses import StreamingResponse

            try:
                # Parse request
                body = await request.json()
                chat_request = ChatStreamRequest(**body)

                # Extract values from metadata (with backward compatibility)
                metadata = chat_request.metadata or {}
                client_id = metadata.get('client_id')
                project_id = metadata.get('project_id')
                ai_persona = metadata.get('ai_persona')

                logger.info(f"Chat stream request: conversation_id={chat_request.conversation_id}, client_id={client_id}, metadata={metadata}")

                # Stream generator
                async def event_generator():
                    try:
                        async for event in flashbrain_agent.stream(
                            query=chat_request.message,
                            context_id=chat_request.conversation_id,
                            client_id=client_id,
                            project_id=project_id,
                            ai_persona=ai_persona
                        ):
                            # Send as Server-Sent Events format
                            yield f"data: {json.dumps(event)}\n\n"
                    except Exception as e:
                        logger.error(f"Stream error: {e}", exc_info=True)
                        error_event = {
                            'is_task_complete': True,
                            'require_user_input': False,
                            'content': f"Error: {str(e)}"
                        }
                        yield f"data: {json.dumps(error_event)}\n\n"

                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"  # Disable nginx buffering
                    }
                )

            except Exception as e:
                logger.error(f"Chat stream endpoint error: {e}", exc_info=True)
                from starlette.responses import JSONResponse
                return JSONResponse({"error": str(e)}, status_code=500)

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
        
        # Build the Starlette app and add custom routes
        starlette_app = server.build()
        from starlette.routing import Route
        from starlette.responses import JSONResponse
        
        # Add health check endpoint (required for Cloud Run)
        async def health_check(request):
            """Health check endpoint for Cloud Run."""
            return JSONResponse({
                "status": "healthy",
                "service": "FlashBrain Orchestrator"
            })
        
        starlette_app.routes.append(
            Route("/health", health_check, methods=["GET"])
        )
        starlette_app.routes.append(
            Route("/chat/stream", handle_chat_stream, methods=["POST"])
        )
        starlette_app.routes.append(
            Route("/callbacks/project-complete", handle_project_complete, methods=["POST"])
        )

        logger.info(f'Starting FlashBrain Orchestrator on {host}:{port}')
        uvicorn.run(starlette_app, host=host, port=port)

    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)
    finally:
        # Ensure cleanup if possible (though uvicorn captures SIGINT)
        pass


if __name__ == '__main__':
    main()
