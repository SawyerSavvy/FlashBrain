"""
Test client for FlashBrain Orchestrator A2A Agent
"""
import logging
from typing import Any
from uuid import uuid4
import json

import httpx

from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)


async def main() -> None:
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # For local testing
    base_url = 'http://localhost:8010'
    # For Cloud Run testing
    #base_url = 'https://flashbrain-orchestrator-barrxvhqyq-uc.a.run.app'

    async with httpx.AsyncClient(timeout=60.0) as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        # Fetch Public Agent Card and Initialize Client
        final_agent_card_to_use: AgentCard | None = None

        try:
            try:
                _public_card = await resolver.get_agent_card()
            except Exception as e:
                logger.warning(f"Standard fetch failed: {e}. Trying fallback to /a2a/agent.json")
                # Fallback to /a2a/agent.json
                resp = await httpx_client.get(f"{base_url}/a2a/agent.json")
                resp.raise_for_status()
                _public_card = AgentCard.model_validate(resp.json())

            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
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

        # Helper to print responses
        async def print_responses(iterator):
            async for item in iterator:
                if isinstance(item, tuple):
                    task, update = item
                    # Extract text from the update
                    if hasattr(update, 'status') and update.status:
                        if hasattr(update.status, 'message') and update.status.message:
                            msg = update.status.message
                            if hasattr(msg, 'parts') and msg.parts:
                                for part in msg.parts:
                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                        print(f"[{update.status.state.value}] {part.root.text}")
                else:
                    # item is a Message object
                    if hasattr(item, 'parts') and item.parts:
                        for part in item.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                print(f"[message] {part.root.text}")

        # Test 1: Simple greeting
        context_id = 'f77fb319-f01f-46a5-8d99-d031b625cca0'
        logger.info('\n=== Test 1: Simple Greeting ===')
        message_data = {
            'role': 'user',
            'context_id': context_id,  # Set context_id in the message
            'parts': [
                {
                    'kind': 'text',
                    'text': """The project, named SkillLink - Micro Mentorship Platform, is a two-sided marketplace platform designed to connect users with experts for short, on-demand mentoring sessions.

**Project Description:**
SkillLink aims to be a real-time mentorship platform facilitating 15-30 minute micro-mentoring sessions. These sessions can be conducted via video, voice, or chat, providing users with quick guidance for skill development, career advice, and tackling specific challenges.

**Business Objectives:**
The platform intends to generate revenue through several streams:
*   Session commissions.
*   A premium subscription model.
*   Monetization of featured mentor listings.

**Target Audience:**
The platform is intended for college students, early-career professionals, self-learners, and entrepreneurs.

**Key Requirements:**
*   A real-time mentor matching system.
*   Communication capabilities supporting video, voice, and chat.
*   A time-based pricing system for sessions.
*   Functionality for session recording.
*   A system for mentor reviews and ratings.
*   A feature to follow or bookmark mentors.
*   An integrated calendar and reminder system.
*   User profiles and authentication.
*   Integration with a payment processing system.

**Technical Architecture:**
The project will involve a mobile application (iOS and Android) and a web application. The complexity is rated as high.

**Project Specifications:**
*   **Timeline:** 2 months
*   **Budget:** $15,000
*   **Priority:** High

**Deliverables:**
*   A mobile application for both iOS and Android.
*   A web platform.

**Additional Details:**
Success metrics, risk factors, strategic recommendations, scalability planning, and a maintenance plan are yet to be defined or developed.""",
                }
            ],
            'message_id': uuid4().hex,
            'metadata': {
                'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
                'user_id': '9a76be62-0d44-4a34-913d-08dcac008de5'
            }
        }
        message = Message(**message_data)
        #await print_responses(client.send_message(message))

        # Test 2: Continue conversation (should remember the name "Cole")
        logger.info('\n=== Test 2: Conversation Continuity Test ===')
        planning_message_data = {
            'role': 'user',
            'context_id': context_id,  # Use the SAME context_id to continue the conversation
            'parts': [
                {
                    'kind': 'text',
                    'text': 'Create a project plan for SkillLink.',
                }
            ],
            'message_id': uuid4().hex,
            'metadata': {
                'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
                'user_id': '9a76be62-0d44-4a34-913d-08dcac008de5'
            }
        }
        planning_message = Message(**planning_message_data)
        # Use the SAME context_id to continue the conversation
        await print_responses(client.send_message(planning_message))


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
