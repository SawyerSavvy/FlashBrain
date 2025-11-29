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

from FlashBrain import FlashBrainAgent
from agent_executor import FlashBrainAgentExecutor

from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8010)
def main(host, port):
    """Starts the FlashBrain Orchestrator server."""
    try:
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
            description='The central orchestrator for the FlashBrain system, managing project planning, team selection, and financial operations.',
            url=public_url,
            version='1.0.0',
            default_input_modes=FlashBrainAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=FlashBrainAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[flashbrain_skill],
        )

        # Get the agent instance (create if needed)
        flashbrain_agent = FlashBrainAgent()

        # Create request handler
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        request_handler = DefaultRequestHandler(
            agent_executor=FlashBrainAgentExecutor(agent=flashbrain_agent),
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
        
        # Define callback endpoint handler
        async def handle_project_complete(request):
            """
            Callback endpoint for long-running project decomposition jobs.
            Called by Project Decomp agent when a job finishes.
            """
            try:
                # Parse JSON payload
                payload = await request.json()
                callback_data = ProjectCompletePayload(**payload)
                logger.info(f"Received callback for project {callback_data.project_id}, job {callback_data.job_id}, status {callback_data.status}")
                
                # Update job status in database
                if flashbrain_agent.supabase_client:
                    # Retry logic for database operations (handle stale connections) Tries to update the connection max_retires times before giving up
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            flashbrain_agent.supabase_client.table("brain_jobs").update({
                                "status": callback_data.status,
                                "result": callback_data.result,
                                "error": callback_data.error,
                                "updated_at": "now()"
                            }).eq("id", callback_data.job_id).execute()
                            break  # Success, exit retry loop
                        except Exception as e:
                            logger.warning(f"Database update attempt {attempt + 1}/{max_retries} failed: {e}")
                            if attempt == max_retries - 1:
                                # Final attempt failed
                                logger.error(f"Failed to update job status after {max_retries} attempts: {e}")
                                from starlette.responses import JSONResponse
                                return JSONResponse({"error": "Failed to update job"}, status_code=500)
                            # Recreate Supabase client for retry
                            try:
                                from supabase import create_client
                                flashbrain_agent.supabase_client = create_client(
                                    flashbrain_agent.supabase_url,
                                    flashbrain_agent.supabase_key
                                )
                                logger.info("Recreated Supabase client for retry")
                            except Exception as recreate_error:
                                logger.error(f"Failed to recreate Supabase client: {recreate_error}")
                
                # Find thread_id for this job
                for attempt in range(max_retries):
                    try:
                        job_record = flashbrain_agent.supabase_client.table("brain_jobs")\
                            .select("thread_id")\
                            .eq("id", callback_data.job_id)\
                            .single()\
                            .execute()
                        
                        thread_id = job_record.data["thread_id"]
                        break  # Success
                    except Exception as e:
                        logger.warning(f"Thread lookup attempt {attempt + 1}/{max_retries} failed: {e}")
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to find thread_id after {max_retries} attempts: {e}")
                            from starlette.responses import JSONResponse
                            return JSONResponse({"error": "Job not found"}, status_code=404)
                
                # Resume the workflow
                try:
                    config = {"configurable": {"thread_id": thread_id}}
                    
                    # Get the current state from the checkpointer
                    current_state = await flashbrain_agent.graph.aget_state(config)
                    logger.info(f"[CALLBACK] Current state values: {list(current_state.values.keys())}")
                    
                    # Check if there are suspended tasks to resume
                    suspended_tasks = current_state.values.get("suspended_tasks")
                    waiting_project_id = current_state.values.get("waiting_for_project_id")
                    
                    logger.info(f"[CALLBACK] Suspended tasks: {len(suspended_tasks) if suspended_tasks else 0}")
                    logger.info(f"[CALLBACK] Waiting for project: {waiting_project_id}")
                    
                    if suspended_tasks and waiting_project_id == callback_data.project_id:
                        # Update the state to activate suspended tasks and route to task_executor
                        await flashbrain_agent.graph.aupdate_state(
                            config,
                            {
                                "waiting_for_project_id": None,
                                "suspended_tasks": None,
                                "active_job_id": None,
                                "pending_tasks": suspended_tasks,
                                "current_task_idx": 0,
                                "next_step": "task_executor",  # Route directly to task_executor
                                "messages": [AIMessage(content=f"✅ Project plan is ready! Now continuing with your other tasks...")]
                            },
                            as_node="check_dependency"  # Resume as if coming from check_dependency node
                        )
                        
                        # Now invoke with None to continue from the updated state
                        # This will pick up from task_executor node (next_step was set above)
                        result = await flashbrain_agent.graph.ainvoke(None, config=config)
                        
                        logger.info(f"Successfully resumed {len(suspended_tasks)} suspended tasks for thread {thread_id}")
                        from starlette.responses import JSONResponse
                        return JSONResponse({"status": "resumed", "tasks_resumed": len(suspended_tasks)})
                    else:
                        # No suspended tasks, just notify user
                        logger.info(f"[CALLBACK] No suspended tasks to resume for thread {thread_id}")
                        from starlette.responses import JSONResponse
                        return JSONResponse({"status": "completed", "message": "No tasks to resume"})
                
                except Exception as e:
                    logger.error(f"Failed to resume workflow: {e}")
                    from starlette.responses import JSONResponse
                    return JSONResponse({"error": str(e)}, status_code=500)
            
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
