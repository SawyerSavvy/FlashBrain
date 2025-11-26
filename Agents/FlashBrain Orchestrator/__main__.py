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

load_dotenv()

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

        # Create request handler
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        request_handler = DefaultRequestHandler(
            agent_executor=FlashBrainAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        logger.info(f'Starting FlashBrain Orchestrator on {host}:{port}')
        uvicorn.run(server.build(), host=host, port=port)

    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
