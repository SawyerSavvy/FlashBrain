"""
Test client for FlashBrain Orchestrator A2A Agent
"""
import logging
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)


async def main() -> None:
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_url = 'http://localhost:8010'

    async with httpx.AsyncClient() as httpx_client:
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

        # Initialize A2A Client
        client = A2AClient(
            httpx_client=httpx_client, agent_card=final_agent_card_to_use
        )
        logger.info('A2AClient initialized.')

        # Test 1: Simple greeting
        logger.info('\n=== Test 1: Simple Greeting ===')
        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'Hello, how are you today?'}
                ],
                'message_id': uuid4().hex,
            },
        }
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        response = await client.send_message(request)
        print('\nResponse:')
        print(response.model_dump(mode='json', exclude_none=True))

        # Test 2: Project planning request
        logger.info('\n=== Test 2: Project Planning Request ===')
        planning_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': 'Help me plan and decompose my e-commerce project',
                    }
                ],
                'message_id': uuid4().hex,
            },
        }
        planning_request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**planning_payload)
        )

        planning_response = await client.send_message(planning_request)
        print('\nPlanning Response:')
        print(planning_response.model_dump(mode='json', exclude_none=True))

        # Test 3: Multiturn conversation
        logger.info('\n=== Test 3: Multiturn Conversation ===')
        first_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': 'I need help with my project budget',
                    }
                ],
                'message_id': uuid4().hex,
            },
        }

        first_request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**first_message_payload),
        )

        first_response = await client.send_message(first_request)
        print('\nFirst Turn Response:')
        print(first_response.model_dump(mode='json', exclude_none=True))

        task_id = first_response.root.result.id
        context_id = first_response.root.result.context_id

        second_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'user': 'text',
                        'text': 'The budget is $50,000 and timeline is 3 months',
                    }
                ],
                'message_id': uuid4().hex,
                'task_id': task_id,
                'context_id': context_id,
            },
        }

        second_request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**second_message_payload),
        )

        second_response = await client.send_message(second_request)
        print('\nSecond Turn Response:')
        print(second_response.model_dump(mode='json', exclude_none=True))

        # Test 4: Streaming response
        logger.info('\n=== Test 4: Streaming Response ===')
        streaming_request = SendStreamingMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        stream_response = client.send_message_streaming(streaming_request)

        print('\nStreaming Response:')
        async for chunk in stream_response:
            print(chunk.model_dump(mode='json', exclude_none=True))


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
