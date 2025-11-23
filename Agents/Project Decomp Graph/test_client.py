"""
Test client for Project Decomposition A2A Agent
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

    base_url = 'http://localhost:8011'

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

        # Test 1: Simple project decomposition
        logger.info('\n=== Test 1: Simple Project Decomposition ===')
        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': 'Break down a web application project for an e-commerce platform',
                    }
                ],
                'message_id': uuid4().hex,
                'metadata': { #Optional metadata associated with this part
                    'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
                    'client_id': '9a76be62-0d44-4a34-913d-08dcac008de5',
                    'exist': True
                }
            },
        }
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        response = await client.send_message(request)
        print('\nResponse:')
        print(response.model_dump(mode='json', exclude_none=True))

        # Test 2: Project decomposition with specific requirements
        logger.info('\n=== Test 2: Detailed Project Decomposition ===')
        detailed_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': '''Decompose a mobile app project with these requirements:
                        - User authentication
                        - Real-time messaging
                        - Payment processing
                        - Push notifications
                        Timeline: 3 months, Budget: $75,000''',
                    }
                ],
                'message_id': uuid4().hex,
            },
        }
        detailed_request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**detailed_payload)
        )

        detailed_response = await client.send_message(detailed_request)
        print('\nDetailed Response:')
        print(detailed_response.model_dump(mode='json', exclude_none=True))

        # Test 3: Streaming response
        logger.info('\n=== Test 3: Streaming Project Decomposition ===')
        streaming_request = SendStreamingMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        stream_response = client.send_message_streaming(streaming_request)

        print('\nStreaming Response:')
        async for chunk in stream_response:
            print(chunk.model_dump(mode='json', exclude_none=True))

        # Test 4: Project Decomposition with Metadata
        logger.info('\n=== Test 4: Project Decomposition with Metadata ===')
        metadata_payload: dict[str, Any] = {
            'message': { #Single message in the conversation between a user and a agent. 
                'role': 'user', #identifes the sender of the message
                'parts': [ # RootModel[Union[TextPart, FilePart, DataPart]] below is an example of a TextPart
                    {
                        'kind': 'text', #Type of this part
                        'text': 'Update the existing project plan with new timeline', #String content of the text part
                    }
                ],
                'message_id': uuid4().hex,
                'metadata': { #Optional metadata associated with this part
                    'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
                    'client_id': '9a76be62-0d44-4a34-913d-08dcac008de5',
                    'exist': True
                }
            },
        }
        metadata_request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**metadata_payload)
        )

        metadata_response = await client.send_message(metadata_request)
        print('\nMetadata Response:')
        print(metadata_response.model_dump(mode='json', exclude_none=True))


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
