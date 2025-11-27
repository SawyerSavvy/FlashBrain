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

from summarization_agent import SummarizationAgent
from agent_executor import SummarizationAgentExecutor

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8013)
def main(host, port):
    """Starts the Summarization Agent server."""
    try:
        capabilities = AgentCapabilities(streaming=True, push_notifications=True)

        summarization_skill = AgentSkill(
            id='summarize_agent',
            name='Summarize Conversation',
            description='Summarizes a list of messages, condensing conversation history while preserving key information.',
            tags=['summarization', 'conversation', 'memory'],
            examples=[
                'Summarize this conversation history',
                'Condense these messages',
                'Create a summary of our discussion'
            ],
        )

        # Use SUMMARIZATION_PUBLIC_URL env var for Cloud Run, fallback to host:port for local
        public_url = os.getenv('SUMMARIZATION_PUBLIC_URL', f'http://{host}:{port}')

        agent_card = AgentCard(
            name='Summarization Agent',
            description='An agent that summarizes conversation history, helping manage context length.',
            url=public_url,
            version='1.0.0',
            default_input_modes=SummarizationAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=SummarizationAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[summarization_skill],
        )

        # Create request handler
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        request_handler = DefaultRequestHandler(
            agent_executor=SummarizationAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        logger.info(f'Starting Summarization Agent on {host}:{port}')
        uvicorn.run(server.build(), host=host, port=port)

    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
