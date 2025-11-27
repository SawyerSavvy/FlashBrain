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

        logger.info(f'Starting Project Decomposition Agent on {host}:{port}')
        uvicorn.run(server.build(), host=host, port=port)

    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
