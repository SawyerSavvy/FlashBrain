"""
Test client for Summarization A2A Agent
"""
import logging
from typing import Any
from uuid import uuid4
import json

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

    base_url = 'http://localhost:8013'

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

        # Test 1: Summarize a conversation
        logger.info('\n=== Test 1: Summarize Conversation History ===')

        # Create sample conversation history
        messages = [
            {"role": "user", "content": "I want to build an e-commerce platform"},
            {"role": "assistant", "content": "Great! I can help you with that. What features do you need?"},
            {"role": "user", "content": "I need product catalog, shopping cart, and payment processing"},
            {"role": "assistant", "content": "Those are essential features. What's your timeline and budget?"},
            {"role": "user", "content": "I have 3 months and $50,000 budget"},
            {"role": "assistant", "content": "Perfect. I'll help you plan this project with those constraints."},
        ]

        messages_json = json.dumps({"messages": messages, "keep_last": 2})

        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': messages_json}
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

        # Test 2: Summarize with different keep_last value
        logger.info('\n=== Test 2: Summarize with Keep Last 3 ===')

        messages_json_2 = json.dumps({"messages": messages, "keep_last": 3})

        summarize_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': messages_json_2}
                ],
                'message_id': uuid4().hex,
            },
        }
        summarize_request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**summarize_payload)
        )

        summarize_response = await client.send_message(summarize_request)
        print('\nSummarize Response:')
        print(summarize_response.model_dump(mode='json', exclude_none=True))

        # Test 3: Simple text summarization
        logger.info('\n=== Test 3: Simple Text Summarization ===')

        simple_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': 'Summarize this conversation about building a mobile app with authentication and real-time features',
                    }
                ],
                'message_id': uuid4().hex,
            },
        }
        simple_request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**simple_payload)
        )

        simple_response = await client.send_message(simple_request)
        print('\nSimple Response:')
        print(simple_response.model_dump(mode='json', exclude_none=True))

        # Test 4: Streaming response
        logger.info('\n=== Test 4: Streaming Summarization ===')
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
