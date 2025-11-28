"""
Test multi-intent orchestration functionality.
Tests both single-intent (backward compatibility) and multi-intent scenarios.
"""
import asyncio
import logging
from uuid import uuid4
import httpx

from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import AgentCard, Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_client(httpx_client: httpx.AsyncClient, base_url: str = "http://localhost:8010"):
    """Initialize A2A client with proper agent card resolution."""
    # Initialize A2ACardResolver
    resolver = A2ACardResolver(
        httpx_client=httpx_client,
        base_url=base_url,
    )

    # Fetch Public Agent Card
    try:
        try:
            agent_card = await resolver.get_agent_card()
        except Exception as e:
            logger.warning(f"Standard fetch failed: {e}. Trying fallback to /a2a/agent.json")
            # Fallback to /a2a/agent.json
            resp = await httpx_client.get(f"{base_url}/a2a/agent.json")
            resp.raise_for_status()
            agent_card = AgentCard.model_validate(resp.json())

        logger.info('Using PUBLIC agent card for client initialization.')

    except Exception as e:
        logger.error(f'Failed to fetch agent card: {e}', exc_info=True)
        raise RuntimeError('Failed to fetch the public agent card.') from e

    # Initialize A2A Client using ClientFactory
    client_config = ClientConfig(httpx_client=httpx_client)
    client = await ClientFactory.connect(
        agent=agent_card,
        client_config=client_config
    )

    return client


async def print_responses(iterator):
    """Helper to print responses from A2A agent."""
    full_response = ""
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
                                text = part.root.text
                                print(f"[{update.status.state.value}] {text}")
                                full_response += text
        else:
            # item is a Message object
            if hasattr(item, 'parts') and item.parts:
                for part in item.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        text = part.root.text
                        print(f"[message] {text}")
                        full_response += text
    return full_response


async def test_single_intent(httpx_client: httpx.AsyncClient):
    """Test backward compatibility with single-intent messages."""
    print("\n=== Test 1: Single Intent (Backward Compatibility) ===")

    client = await setup_client(httpx_client)

    # Use proper UUIDs for context_id
    context_id = str(uuid4())

    message_data = {
        'role': 'user',
        'context_id': context_id,
        'parts': [
            {
                'kind': 'text',
                'text': 'What is project management?',
            }
        ],
        'message_id': uuid4().hex,
        'metadata': {
            'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
            'user_id': '9a76be62-0d44-4a34-913d-08dcac008de5'
        }
    }
    message = Message(**message_data)

    response = await print_responses(client.send_message(message))
    print(f"\n✅ Single intent test completed. Response length: {len(response)} chars")


async def test_multi_intent(httpx_client: httpx.AsyncClient):
    """Test multi-intent with multiple separate questions."""
    print("\n=== Test 2: Multi-Intent (Multiple Questions) ===")

    client = await setup_client(httpx_client)

    # Use proper UUIDs for context_id
    context_id = str(uuid4())

    # This message should trigger multi-intent detection
    message_text = """Please help me with these tasks:
1. What are the key principles of agile methodology?
2. Create a project plan for building a mobile app called SkillLink that connects freelancers with clients."""

    message_data = {
        'role': 'user',
        'context_id': context_id,
        'parts': [
            {
                'kind': 'text',
                'text': message_text,
            }
        ],
        'message_id': uuid4().hex,
        'metadata': {
            'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
            'user_id': '9a76be62-0d44-4a34-913d-08dcac008de5'
        }
    }
    message = Message(**message_data)

    response = await print_responses(client.send_message(message))
    print(f"\n✅ Multi-intent test completed. Response length: {len(response)} chars")


async def test_decomp_then_freelancer(httpx_client: httpx.AsyncClient):
    """Test dependency: project decomp should run before freelancer selection."""
    print("\n=== Test 3: Project Decomp + Freelancer Selection ===")

    client = await setup_client(httpx_client)

    # Use proper UUIDs for context_id
    context_id = str(uuid4())

    message_text = "Create a project plan for an e-commerce website and then find React developers for it."

    message_data = {
        'role': 'user',
        'context_id': context_id,
        'parts': [
            {
                'kind': 'text',
                'text': message_text,
            }
        ],
        'message_id': uuid4().hex,
        'metadata': {
            'project_id': '058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc',
            'user_id': '9a76be62-0d44-4a34-913d-08dcac008de5'
        }
    }
    message = Message(**message_data)

    response = await print_responses(client.send_message(message))

    # Check if both tasks appear in response
    has_project_plan = "project" in response.lower() or "plan" in response.lower()
    has_freelancer = "freelancer" in response.lower() or "developer" in response.lower()
    print(f"\nContains project plan: {has_project_plan}")
    print(f"Contains freelancer info: {has_freelancer}")
    print(f"\n✅ Dependency test completed. Response length: {len(response)} chars")

async def main():
    """Run all tests."""
    async with httpx.AsyncClient(timeout=60.0) as httpx_client:
        try:
            await test_single_intent(httpx_client)
            await test_multi_intent(httpx_client)
            await test_decomp_then_freelancer(httpx_client)
            print("\n" + "="*50)
            print("✅ All tests completed successfully!")
            print("="*50)
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
