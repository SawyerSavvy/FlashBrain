"""
Test client for Select Freelancer A2A Agent
"""
import logging
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Message,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

async def main() -> None:
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_url = 'http://localhost:8012'

    # Increase timeout to 30 seconds to handle long-running agent tasks
    async with httpx.AsyncClient(timeout=30.0) as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        # Fetch Public Agent Card and Initialize Client
        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(
                f'Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}'
            )
            _public_card = await resolver.get_agent_card()
            logger.info('Successfully fetched public agent card:')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )

            if _public_card.supports_authenticated_extended_card:
                try:
                    logger.info(
                        '\nPublic card supports authenticated extended card. '
                        'Attempting to fetch from: '
                        f'{base_url}{EXTENDED_AGENT_CARD_PATH}'
                    )
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info(
                        'Successfully fetched authenticated extended agent card:'
                    )
                    logger.info(
                        _extended_card.model_dump_json(
                            indent=2, exclude_none=True
                        )
                    )
                    final_agent_card_to_use = _extended_card
                    logger.info(
                        '\nUsing AUTHENTICATED EXTENDED agent card for client '
                        'initialization.'
                    )
                except Exception as e_extended:
                    logger.warning(
                        f'Failed to fetch extended agent card: {e_extended}. '
                        'Will proceed with public card.',
                        exc_info=True,
                    )
            elif _public_card:
                logger.info(
                    '\nPublic card does not indicate support for an extended card. Using public card.'
                )

        except Exception as e:
            logger.error(
                f'Critical error fetching public agent card: {e}', exc_info=True
            )
            raise RuntimeError(
                'Failed to fetch the public agent card. Cannot continue.'
            ) from e

        # Initialize A2A Client using ClientFactory
        client_config = ClientConfig(httpx_client=httpx_client)
        client = await ClientFactory.connect(
            agent=final_agent_card_to_use,
            client_config=client_config
        )
        logger.info('A2AClient initialized via ClientFactory.')

        # Helper to print responses
        async def print_responses(iterator):
            async for item in iterator:
                if isinstance(item, tuple):
                    task, update = item
                    print(f"Update: {update}")
                else:
                    print(f"Message: {item}")

        # Test 1: Update role requirements
        logger.info('\n=== Test 1: Update Role Requirements ===')
        message_data = {
            'role': 'user',
            'parts': [
                {
                    'kind': 'text',
                    'text': 'Update the Security Engineer role to require React and TypeScript experience',
                }
            ],
            'message_id': uuid4().hex,
            'metadata': {
                'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
                'client_id': '9a76be62-0d44-4a34-913d-08dcac008de5',
                'exist': True
            }
        }
        message = Message(**message_data)
        await print_responses(client.send_message(message))

        # Test 2: Add new project phase
        logger.info('\n=== Test 2: Add New Project Phase ===')
        phase_message_data = {
            'role': 'user',
            'parts': [
                {
                    'kind': 'text',
                    'text': 'Change the Security Engineer to a Quality Engineer',
                }
            ],
            'message_id': uuid4().hex,
            'metadata': {
                'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
                'client_id': '9a76be62-0d44-4a34-913d-08dcac008de5',
                'exist': True
            }
        }
        phase_message = Message(**phase_message_data)
        await print_responses(client.send_message(phase_message))

        # Test 3: Match freelancers for project
        logger.info('\n=== Test 3: Match Freelancers ===')
        match_message_data = {
            'role': 'user',
            'parts': [
                {
                    'kind': 'text',
                    'text': 'Add a AI Researcher role to the first project phase.',
                }
            ],
            'message_id': uuid4().hex,
            'metadata': {
                'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
                'client_id': '9a76be62-0d44-4a34-913d-08dcac008de5',
                'exist': True
            }
        }
        match_message = Message(**match_message_data)
        await print_responses(client.send_message(match_message))

        # Test 4: Streaming response (Same as Test 1 but explicitly showing streaming capability which is default now)
        logger.info('\n=== Test 4: Streaming Freelancer Selection (Implicit) ===')
        # Re-using message from Test 1
        await print_responses(client.send_message(message))


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

