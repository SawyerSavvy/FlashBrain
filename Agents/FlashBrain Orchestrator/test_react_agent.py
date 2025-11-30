"""
Test script for FlashBrain ReAct Agent running as a local server.

This tests the agent via A2A protocol over HTTP, simulating real client usage.

Usage:
    1. Start the server in one terminal:
       cd "Agents/FlashBrain Orchestrator"
       python __main__.py --host localhost --port 8010
    
    2. Run this test in another terminal:
       python test_react_agent.py
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
    """
    Initialize A2A client for FlashBrain.
    
    Args:
        httpx_client: HTTP client
        base_url: FlashBrain server URL
    
    Returns:
        Connected A2A client
    """
    logger.info(f"Connecting to FlashBrain at {base_url}")
    
    # Fetch agent card
    resolver = A2ACardResolver(
        httpx_client=httpx_client,
        base_url=base_url,
    )
    
    try:
        agent_card = await resolver.get_agent_card()
        logger.info(f"Connected to: {agent_card.name} v{agent_card.version}")
    except Exception as e:
        logger.error(f"Failed to fetch agent card: {e}")
        raise
    
    # Create client connection
    client_config = ClientConfig(httpx_client=httpx_client)
    client = await ClientFactory.connect(
        agent=agent_card,
        client_config=client_config
    )
    
    return client


async def send_and_print(client, message_text: str, context_id: str, project_id: str = None, client_id: str = None):
    """
    Send a message to FlashBrain and print the streaming response.
    
    Args:
        client: A2A client
        message_text: The message to send
        context_id: Conversation context ID
        project_id: Optional project ID
        client_id: Optional client ID
    
    Returns:
        Full response text
    """
    message_data = {
        'role': 'user',
        'parts': [{'kind': 'text', 'text': message_text}],
        'message_id': uuid4().hex,
        'metadata': {
            'project_id': project_id,
            'client_id': client_id
        }
    }
    message = Message(**message_data)
    
    print(f"\n{'='*70}")
    print(f"Query: {message_text}")
    print(f"{'='*70}\n")
    
    full_response = ""
    async for item in client.send_message(message):
        if isinstance(item, tuple):
            task, update = item
            # Extract text from status update
            if hasattr(update, 'status') and update.status:
                if hasattr(update.status, 'message') and update.status.message:
                    msg = update.status.message
                    if hasattr(msg, 'parts') and msg.parts:
                        for part in msg.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                text = part.root.text
                                print(text, end='', flush=True)
                                full_response += text
        else:
            # Direct message
            if hasattr(item, 'parts') and item.parts:
                for part in item.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        text = part.root.text
                        print(text, end='', flush=True)
                        full_response += text
    
    print("\n")  # New line after response
    return full_response


async def test_simple_question(client, context_id):
    """Test 1: Simple question (no tools needed)."""
    print("\n" + "="*70)
    print("TEST 1: Simple General Knowledge Question")
    print("="*70)
    
    response = await send_and_print(
        client,
        "What is agile methodology?",
        context_id
    )
    
    print(f"✅ Test 1 Complete - Response length: {len(response)} chars")
    return response


async def test_tool_calling(client, context_id, project_id, client_id):
    """Test 2: Question that requires calling a remote agent."""
    print("\n" + "="*70)
    print("TEST 2: Tool Calling (send_message to remote agent)")
    print("="*70)
    
    response = await send_and_print(
        client,
        "Create a project plan for an e-commerce website with React frontend and Node.js backend",
        context_id,
        project_id=project_id,
        client_id=client_id
    )
    
    print(f"✅ Test 2 Complete - Response length: {len(response)} chars")
    return response


async def test_multi_intent(client, context_id, project_id, client_id):
    """Test 3: Multi-intent request (multiple tasks)."""
    print("\n" + "="*70)
    print("TEST 3: Multi-Intent Request")
    print("="*70)
    
    message_text = """Please help me with these tasks:
1. What are the key principles of agile methodology?
2. Create a project plan for a mobile app called SkillLink"""
    
    response = await send_and_print(
        client,
        message_text,
        context_id,
        project_id=project_id,
        client_id=client_id
    )
    
    # Check if both tasks appear in response
    has_agile = "agile" in response.lower()
    has_project = "project" in response.lower() or "skilllink" in response.lower()
    
    print(f"\nMentions agile: {has_agile}")
    print(f"Mentions project: {has_project}")
    print(f"✅ Test 3 Complete - Response length: {len(response)} chars")
    
    return response


async def test_conversation_memory(client, context_id):
    """Test 4: Conversation memory (follow-up questions)."""
    print("\n" + "="*70)
    print("TEST 4: Conversation Memory")
    print("="*70)
    
    # First message
    await send_and_print(
        client,
        "My budget is $50,000 and I spend $5,000 per month",
        context_id
    )
    
    # Follow-up (should remember context)
    response = await send_and_print(
        client,
        "What's my runway?",
        context_id
    )
    
    print(f"✅ Test 4 Complete - Response length: {len(response)} chars")
    return response


async def main():
    """Run all tests against local FlashBrain server."""
    print("\n" + "="*70)
    print("FlashBrain ReAct Agent Test Suite")
    print("="*70)
    print("\nMake sure FlashBrain is running:")
    print("  cd 'Agents/FlashBrain Orchestrator'")
    print("  python __main__.py --host localhost --port 8010")
    print("="*70)
    
    # Wait a moment for user to verify
    await asyncio.sleep(2)
    
    async with httpx.AsyncClient(timeout=120.0) as httpx_client:
        try:
            # Setup client
            client = await setup_client(httpx_client, "http://localhost:8010")
            
            # Generate test IDs
            context_id = str(uuid4())
            project_id = "058ed2ae-0bd6-4fc5-8fb5-0f0319a2fcbc"
            client_id = "9a76be62-0d44-4a34-913d-08dcac008de5"
            
            # Run tests
            await test_simple_question(client, context_id)
            await test_tool_calling(client, context_id, project_id, client_id)
            await test_multi_intent(client, context_id, project_id, client_id)
            await test_conversation_memory(client, context_id)
            
            # Summary
            print("\n" + "="*70)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

