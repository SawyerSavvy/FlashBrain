import logging
import os
import sys
from typing import Optional

import click
import httpx
import uvicorn
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

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

from project_decomp_agent import ProjectDecompAgent
from agent_executor import ProjectDecompAgentExecutor

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8011)
def main(host, port):
    """Starts the Project Decomposition Agent server."""
    try:
        # Check required environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'PROJECT_DECOMP_ORCHESTRATOR_URL']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.warning(f'Missing environment variables: {", ".join(missing_vars)}. Some functionality may be limited.')

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)

        project_decomp_skill = AgentSkill(
            id='project_decomposition',
            name='Project Decomposition',
            description='Decomposes a project into phases and tasks, manages project structure in database.',
            tags=['project', 'decomposition', 'planning'],
            examples=[
                'Break down my web application project',
                'Decompose this mobile app into phases',
                'Create a project plan for my e-commerce platform'
            ],
        )

        # Use PROJECT_DECOMP_PUBLIC_URL env var for Cloud Run, fallback to host:port for local
        public_url = os.getenv('PROJECT_DECOMP_PUBLIC_URL', f'http://{host}:{port}')

        agent_card = AgentCard(
            name='Project Decomposition Agent',
            description='An agent that decomposes projects into manageable phases and tasks, storing them in Supabase.',
            url=public_url,
            version='1.0.0',
            default_input_modes=ProjectDecompAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=ProjectDecompAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[project_decomp_skill],
        )

        # Create request handler
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        request_handler = DefaultRequestHandler(
            agent_executor=ProjectDecompAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        # Get agent instance for async endpoint
        project_decomp_agent = ProjectDecompAgent()
        
        # Define async job handler
        async def handle_start_async(request):
            """
            Async endpoint for long-running project decomposition.
            Accepts job_id and callback_url, executes in background, POSTs result to callback.
            """
            import asyncio
            from pydantic import BaseModel
            
            class StartAsyncPayload(BaseModel):
                message: str
                job_id: str
                callback_url: str
                project_id: Optional[str] = None
                client_id: Optional[str] = None
            
            try:
                # Parse payload
                payload_dict = await request.json()
                payload = StartAsyncPayload(**payload_dict)
                logger.info(f"[START_ASYNC] Received job {payload.job_id} with callback {payload.callback_url}")
                
                # Start background task
                async def execute_and_callback():
                    try:
                        # Execute the graph
                        result = await project_decomp_agent.graph.ainvoke({
                            "messages": [HumanMessage(content=payload.message)],
                            "project_id": payload.project_id,
                            "client_id": payload.client_id
                        })
                        
                        # Extract project_id from result
                        project_id = result.get("project_id")
                        
                        # Send callback to orchestrator
                        async with httpx.AsyncClient() as client:
                            await client.post(
                                payload.callback_url,
                                json={
                                    "project_id": project_id,
                                    "job_id": payload.job_id,
                                    "status": "COMPLETED",
                                    "result": {"summary": "Project plan created successfully"}
                                },
                                timeout=10.0
                            )
                        logger.info(f"[START_ASYNC] Job {payload.job_id} completed, callback sent")
                    
                    except Exception as e:
                        logger.error(f"[START_ASYNC] Job {payload.job_id} failed: {e}")
                        # Send failure callback
                        try:
                            async with httpx.AsyncClient() as client:
                                await client.post(
                                    payload.callback_url,
                                    json={
                                        "project_id": None,
                                        "job_id": payload.job_id,
                                        "status": "FAILED",
                                        "error": str(e)
                                    },
                                    timeout=10.0
                                )
                        except Exception as callback_error:
                            logger.error(f"[START_ASYNC] Failed to send error callback: {callback_error}")
                
                # Fire and forget
                asyncio.create_task(execute_and_callback())
                
                # Return immediately
                from starlette.responses import JSONResponse
                return JSONResponse({
                    "status": "started",
                    "job_id": payload.job_id,
                    "project_id": payload.project_id
                })
            
            except Exception as e:
                logger.error(f"[START_ASYNC] Failed to start job: {e}")
                from starlette.responses import JSONResponse
                return JSONResponse({"error": str(e)}, status_code=500)
        
        # Build app and add custom route
        starlette_app = server.build()
        from starlette.routing import Route
        starlette_app.routes.append(
            Route("/start-async", handle_start_async, methods=["POST"])
        )

        logger.info(f'Starting Project Decomposition Agent on {host}:{port}')
        uvicorn.run(starlette_app, host=host, port=port)

    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
